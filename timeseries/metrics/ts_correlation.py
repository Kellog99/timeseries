"""Streaming multivariate dependence metrics on paired observations."""


import torch
from torch import Tensor
from torchmetrics import PearsonCorrCoef

from timeseries.utils.typing import Metric
from timeseries.metric_register import MetricFactory


class PairedMetric(Metric):
    full_state_update = False
    x_list: list[Tensor]
    y_list: list[Tensor]

    def __init__(
            self
            ) -> None:
        super().__init__()
        self.add_state("x_list", default = [], dist_reduce_fx = "cat")
        self.add_state("y_list", default = [], dist_reduce_fx = "cat")

    def update(
            self,
            x: Tensor,
            y: Tensor
            ) -> None:
        x, y = x.detach().to(self.device, dtype = torch.float64), y.detach().to(self.device, dtype = torch.float64)
        x = x.unsqueeze(1) if x.ndim == 1 else x
        y = y.unsqueeze(1) if y.ndim == 1 else y
        if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0] or min(x.shape[1], y.shape[1]) == 0:
            raise ValueError("Expected two feature matrices with the same number of observations.")
        if not torch.isfinite(x).all() or not torch.isfinite(y).all():
            raise ValueError("Observations must be finite.")
        if self.x_list and (x.shape[1] != self.x_list[0].shape[1] or y.shape[1] != self.y_list[0].shape[1]):
            raise ValueError("Feature counts must remain constant across updates.")
        self.x_list.append(x)
        self.y_list.append(y)

    def values(
            self
            ) -> tuple[Tensor, Tensor]:
        if not isinstance(self.x_list, Tensor) and not self.x_list:
            raise ValueError("Update the metric before computing it.")
        x = self.x_list if isinstance(self.x_list, Tensor) else torch.cat(self.x_list)
        y = self.y_list if isinstance(self.y_list, Tensor) else torch.cat(self.y_list)
        if x.shape[0] < 2:
            raise ValueError("At least two paired observations are required.")
        return x, y


@MetricFactory.register(
    name = 'CanonicalCorrelation',
    description = 'Highest canonical correlation of paired feature matrices.',
    id = 'canonical', domain = 'dependence', actions = {"performance"},
)
class CanonicalCorrelation(PairedMetric):

    def compute(
            self
            ) -> Tensor:
        x, y = self.values()

        def basis(
                values: Tensor
                ) -> Tensor:
            centered = values - values.mean(0)
            u, _, _ = torch.linalg.svd(centered, full_matrices = False)
            return u[:, :int(torch.linalg.matrix_rank(centered).item())]

        qx, qy = basis(x), basis(y)
        if qx.shape[1] == 0 or qy.shape[1] == 0:
            return x.new_tensor(float("nan"))
        return torch.linalg.svdvals(qx.T @ qy)[0].clamp(0, 1)


@MetricFactory.register(
    name = 'RVCoefficient',
    description = 'RV coefficient of paired feature matrices.',
    id = 'rv', domain = 'dependence', actions = {"performance"},
)
class RVCoefficient(PairedMetric):

    def compute(
            self
            ) -> Tensor:
        x, y = self.values()
        x, y = x - x.mean(0), y - y.mean(0)
        xx, yy = x @ x.T, y @ y.T
        return (xx * yy).sum() / ((xx.square().sum() * yy.square().sum()).sqrt())


@MetricFactory.register(
    name = 'DistanceCorrelation',
    description = 'Distance correlation capturing nonlinear dependence.',
    id = 'distance', domain = 'dependence', actions = {"performance"},
)
class DistanceCorrelation(PairedMetric):

    def compute(
            self
            ) -> Tensor:
        x, y = self.values()

        def distances(
                values: Tensor
                ) -> Tensor:
            matrix = torch.cdist(values, values)
            return matrix - matrix.mean(1, keepdim = True) - matrix.mean(0, keepdim = True) + matrix.mean()

        a, b = distances(x), distances(y)
        covariance = (a * b).mean().clamp_min(0)
        denominator = (a.square().mean() * b.square().mean()).sqrt()
        return (covariance / denominator).clamp(0, 1).sqrt()


@MetricFactory.register(
    name = 'PearsonCorrelation',
    description = 'Pearson correlation of paired univariate observations.',
    id = 'pearson', domain = 'dependence', actions = {"performance"},
)
class PearsonCorrelation(PearsonCorrCoef):

    def __init__(
            self
            ) -> None:
        super().__init__()
