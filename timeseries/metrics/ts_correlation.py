import torch
from torchmetrics import Metric


class PearsonCorrelation(Metric):
    """Computes the p x q correlation matrix between two latent spaces."""

    def __init__(self):
        super().__init__()
        self.add_state("n", default=torch.tensor(0))
        self.add_state("x_mean", default=torch.tensor(0))
        self.add_state("x_second", default=torch.tensor(0))

        self.add_state("y_mean", default=torch.tensor(0))
        self.add_state("y_second", default=torch.tensor(0))

        self.add_state("outer_product", default=torch.tensor(0))

    def update(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """
        Args:
            x: a tensor of shape B x p
            y: a tensor of shape B x p
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if y.dim() == 1:
            y = y.unsqueeze(0)
        if x.shape != y.shape:
            raise ValueError(f"The shape of x, {x.shape}, is different from y, {y.shape}")
        #########################################

        n = x.shape[0]
        a: torch.Tensor = self.n / (self.n + n)
        b: torch.Tensor = 1 / (self.n + n)

        self.x_mean = a * self.x_mean + b * x.sum(0)
        self.y_mean = a * self.y_mean + b * y.sum(0)

        self.x_second = a * self.x_second + b * x.T @ x
        self.y_second = a * self.y_second + b * y.T @ y

        self.outer_product = a * self.outer_product + b * x.T @ y

        self.n = self.n + n

    def compute(self) -> float:
        """
        Returns: float \in [0, 1]
        """
        x_std = self.x_second - self.x_mean.unsqueeze(1) @ self.x_mean.unsqueeze(0)
        y_std = self.y_second - self.y_mean.unsqueeze(1) @ self.y_mean.unsqueeze(0)
        den = (x_std * y_std).sqrt()

        cov = self.outer_product - self.x_mean.unsqueeze(1) @ self.y_mean.unsqueeze(0)
        corr = cov / den.clamp(1e-8)
        return corr.abs().max().item()


class CanonicalCorrelation(Metric):
    """Computes the first (highest) canonical correlation value."""

    def __init__(self):
        super().__init__()
        self.add_state("x_list", default=[], dist_reduce_effect="cat")
        self.add_state("y_list", default=[], dist_reduce_effect="cat")

    def update(self, x: torch.Tensor, y: torch.Tensor):
        self.x_list.append(x)
        self.y_list.append(y)

    def compute(self) -> float:
        X = torch.cat(self.x_list, dim=0)
        Y = torch.cat(self.y_list, dim=0)

        # Center and Orthogonalize
        X_c = X - X.mean(dim=0)
        Y_c = Y - Y.mean(dim=0)

        # QR decomposition for numerical stability in high dimensions
        Qx, _ = torch.linalg.qr(X_c)
        Qy, _ = torch.linalg.qr(Y_c)

        # Singular values of the product are the canonical correlations
        S = torch.linalg.svdvals(torch.mm(Qx.t(), Qy))
        return S[0]


class RVCoefficient(Metric):
    """Computes the RV coefficient (multivariate generalization of R^2)."""

    def __init__(self):
        super().__init__()
        self.add_state("x_list", default=[], dist_reduce_effect="cat")
        self.add_state("y_list", default=[], dist_reduce_effect="cat")

    def update(self, x: torch.Tensor, y: torch.Tensor):
        self.x_list.append(x)
        self.y_list.append(y)

    def compute(self) -> float:
        X = torch.cat(self.x_list, dim=0)
        Y = torch.cat(self.y_list, dim=0)

        X_c = X - X.mean(dim=0)
        Y_c = Y - Y.mean(dim=0)

        # Build Similarity Matrices
        SXX = torch.mm(X_c, X_c.t())
        SYY = torch.mm(Y_c, Y_c.t())

        num = torch.trace(torch.mm(SXX, SYY))
        den = torch.sqrt(torch.trace(torch.mm(SXX, SXX)) * torch.trace(torch.mm(SYY, SYY)))
        return num / (den + 1e-8)


class DistanceCorrelation(Metric):
    """Computes Distance Correlation (captures non-linear dependence)."""

    def __init__(self):
        super().__init__()
        self.add_state("x_list", default=[], dist_reduce_effect="cat")
        self.add_state("y_list", default=[], dist_reduce_effect="cat")

    def update(self, x: torch.Tensor, y: torch.Tensor):
        self.x_list.append(x)
        self.y_list.append(y)

    def compute(self) -> float:
        X = torch.cat(self.x_list, dim=0)
        Y = torch.cat(self.y_list, dim=0)
        n = X.size(0)

        def double_center(mat):
            return mat - mat.mean(dim=1, keepdim=True) - mat.mean(dim=0, keepdim=True) + mat.mean()

        # Compute Euclidean distances
        A = double_center(torch.cdist(X, X, p=2))
        B = double_center(torch.cdist(Y, Y, p=2))

        dcov2 = torch.sum(A * B) / (n * n)
        dvarX2 = torch.sum(A * A) / (n * n)
        dvarY2 = torch.sum(B * B) / (n * n)

        return torch.sqrt(dcov2 / (torch.sqrt(dvarX2 * dvarY2) + 1e-8))
