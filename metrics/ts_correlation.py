from typing import Literal

import torch
import torch.nn as nn


class TimeSeriesCorrelation(nn.Module):
    """
    Compute correlation metrics among multiple time series.

    Args:
        method: Correlation method ('pearson' or 'spearman')
        reduction: How to aggregate pairwise correlations
            - 'mean': Average of all pairwise correlations
            - 'min': Minimum correlation (weakest relationship)
            - 'max': Maximum correlation (strongest relationship)
            - 'none': Return full correlation matrix
        dim: Dimension along which series are arranged (default: 1)
    """

    def __init__(
            self,
            method: Literal['pearson', 'spearman'] = 'pearson',
            reduction: Literal['mean', 'min', 'max', 'none'] = 'mean',
            dim: int = 1
    ):
        super().__init__()
        self.method = method
        self.reduction = reduction
        self.dim = dim

    def _pearson_correlation(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Pearson correlation matrix."""
        # Center the data
        x_centered = x - x.mean(dim=-1, keepdim=True)

        # Compute covariance matrix
        cov = torch.matmul(x_centered, x_centered.transpose(-2, -1))

        # Compute standard deviations
        std = torch.sqrt(torch.sum(x_centered ** 2, dim=-1, keepdim=True))

        # Compute correlation matrix
        corr = cov / (torch.matmul(std, std.transpose(-2, -1)) + 1e-8)

        return corr

    def _spearman_correlation(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Spearman rank correlation matrix."""
        # Convert to ranks
        ranks = self._compute_ranks(x)

        # Compute Pearson correlation on ranks
        return self._pearson_correlation(ranks)

    def _compute_ranks(self, x: torch.Tensor) -> torch.Tensor:
        """Compute ranks along the time dimension."""
        # Get sorting indices
        sorted_indices = torch.argsort(x, dim=-1)

        # Create ranks tensor
        ranks = torch.zeros_like(x)

        # Assign ranks
        batch_size, n_series, n_timesteps = x.shape
        for i in range(batch_size):
            for j in range(n_series):
                ranks[i, j, sorted_indices[i, j]] = torch.arange(
                    n_timesteps, dtype=x.dtype, device=x.device
                )

        return ranks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute correlation among time series.

        Args:
            x: Input tensor of shape (batch, n_series, time_steps)
               or (n_series, time_steps) for single batch

        Returns:
            Correlation value(s) based on reduction method:
            - 'mean'/'min'/'max': Scalar or (batch,) tensor
            - 'none': (batch, n_series, n_series) correlation matrix
        """
        # Handle single batch case
        if x.ndim == 2:
            x = x.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        # Ensure correct dimension order
        if self.dim != 1:
            x = x.transpose(1, self.dim)

        # Compute correlation matrix
        if self.method == 'pearson':
            corr_matrix = self._pearson_correlation(x)
        elif self.method == 'spearman':
            corr_matrix = self._spearman_correlation(x)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Apply reduction
        if self.reduction == 'none':
            result = corr_matrix
        else:
            # Get upper triangle (excluding diagonal) for pairwise correlations
            batch_size, n_series = corr_matrix.shape[0], corr_matrix.shape[1]
            mask = torch.triu(torch.ones(n_series, n_series, device=x.device), diagonal=1).bool()

            if self.reduction == 'mean':
                result = torch.stack([
                    corr_matrix[i][mask].mean() for i in range(batch_size)
                ])
            elif self.reduction == 'min':
                result = torch.stack([
                    corr_matrix[i][mask].min() for i in range(batch_size)
                ])
            elif self.reduction == 'max':
                result = torch.stack([
                    corr_matrix[i][mask].max() for i in range(batch_size)
                ])
            else:
                raise ValueError(f"Unknown reduction: {self.reduction}")

        if squeeze_output and self.reduction != 'none':
            result = result.squeeze(0)

        return result
