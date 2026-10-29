"""Risk metrics on one-dimensional decimal simple returns (0.01 means 1%).

Sharpe/volatility use sample standard deviation. Sortino uses the root mean
square of negative target deviations over all observations. Ratios with a zero
denominator return signed infinity, or NaN when the numerator is also zero.
"""

from math import isfinite, sqrt
from numbers import Integral

import torch
from torch import Tensor

from timeseries.utils.typing import Metric
from timeseries.metric_register import MetricFactory


def as_returns(
        data: Tensor | list[float]
        ) -> Tensor:
    values = torch.as_tensor(data, dtype = torch.float64)
    if values.ndim != 1:
        raise ValueError("Expected a one-dimensional sequence of decimal returns.")
    if not torch.isfinite(values).all() or (values < -1).any():
        raise ValueError("Returns must be finite and greater than or equal to -1.")
    return values


def divide(
        numerator: Tensor,
        denominator: Tensor
        ) -> Tensor:
    # IEEE division preserves undefined and infinite results without hiding them.
    return numerator / denominator


def max_drawdown(
        returns: Tensor
        ) -> Tensor:
    wealth = torch.cat((returns.new_ones(1), (1 + returns).cumprod(0)))
    peaks = wealth.cummax(0).values
    return (1 - wealth / peaks).max()


class ReturnMetric(Metric):
    full_state_update = False
    returns: list[Tensor]

    def __init__(
            self
            ) -> None:
        super().__init__()
        self.add_state("returns", default = [], dist_reduce_fx = "cat")

    def update(
            self,
            data: Tensor | list[float]
            ) -> None:
        self.returns.append(as_returns(data).detach().to(self.device))

    def values(
            self
            ) -> Tensor:
        # TorchMetrics may concatenate list states during distributed sync.
        values = self.returns if isinstance(self.returns, Tensor) else (
            torch.cat(self.returns) if self.returns else torch.empty(0, device = self.device, dtype = torch.float64)
        )
        if values.numel() == 0:
            raise ValueError("Update the metric with at least one return before computing it.")
        return values


@MetricFactory.register(
    name = 'SharpeRatio',
    description = 'Annualized mean excess return divided by sample volatility.',
    id = 'sharpe', domain = 'risk', actions = {"performance"},
)
class SharpeRatio(ReturnMetric):

    def __init__(
            self,
            risk_free_rate: float = 0.0,
            periods_per_year: int = 252
            ) -> None:
        super().__init__()
        self.risk_free_rate = float(risk_free_rate)
        if not isfinite(self.risk_free_rate):
            raise ValueError("risk_free_rate must be finite.")
        if isinstance(periods_per_year, bool) or not isinstance(periods_per_year, Integral) or periods_per_year <= 0:
            raise ValueError("periods_per_year must be a positive integer.")
        self.periods_per_year = int(periods_per_year)

    def compute(
            self
            ) -> Tensor:
        excess = self.values() - self.risk_free_rate
        std = excess.std() if excess.numel() > 1 else excess.new_tensor(float("nan"))
        return divide(excess.mean(), std) * sqrt(self.periods_per_year)


@MetricFactory.register(
    name = 'OmegaRatio',
    description = 'Sum of gains above a threshold divided by shortfalls below it.',
    id = 'omega', domain = 'risk', actions = {"performance"},
)
class OmegaRatio(ReturnMetric):

    def __init__(
            self,
            threshold: float = 0.0
            ) -> None:
        super().__init__()
        self.threshold = float(threshold)
        if not isfinite(self.threshold):
            raise ValueError("threshold must be finite.")

    def compute(
            self
            ) -> Tensor:
        excess = self.values() - self.threshold
        return divide(excess.clamp_min(0).sum(), (-excess).clamp_min(0).sum())


@MetricFactory.register(
    name = 'SortinoRatio',
    description = 'Annualized mean target excess return divided by downside deviation.',
    id = 'sortino', domain = 'risk', actions = {"performance"},
)
class SortinoRatio(ReturnMetric):

    def __init__(
            self,
            target_return: float = 0.0,
            periods_per_year: int = 252
            ) -> None:
        super().__init__()
        self.target_return = float(target_return)
        if not isfinite(self.target_return):
            raise ValueError("target_return must be finite.")
        if isinstance(periods_per_year, bool) or not isinstance(periods_per_year, Integral) or periods_per_year <= 0:
            raise ValueError("periods_per_year must be a positive integer.")
        self.periods_per_year = int(periods_per_year)

    def compute(
            self
            ) -> Tensor:
        excess = self.values() - self.target_return
        downside = excess.clamp_max(0).square().mean().sqrt()
        return divide(excess.mean(), downside) * sqrt(self.periods_per_year)


@MetricFactory.register(
    name = 'Volatility',
    description = 'Annualized sample standard deviation of decimal returns.',
    id = 'volatility', domain = 'risk', actions = {"performance"},
)
class Volatility(ReturnMetric):

    def __init__(
            self,
            periods_per_year: int = 252
            ) -> None:
        super().__init__()
        if isinstance(periods_per_year, bool) or not isinstance(periods_per_year, Integral) or periods_per_year <= 0:
            raise ValueError("periods_per_year must be a positive integer.")
        self.periods_per_year = int(periods_per_year)

    def compute(
            self
            ) -> Tensor:
        values = self.values()
        return (values.std() if values.numel() > 1 else values.new_tensor(float("nan"))) * sqrt(self.periods_per_year)


@MetricFactory.register(
    name = 'MaxDrawdown',
    description = 'Largest compounded wealth decline from a previous peak, including initial capital.',
    id = 'max_drawdown', domain = 'risk', actions = {"performance"},
)
class MaxDrawdown(ReturnMetric):

    def compute(
            self
            ) -> Tensor:
        return max_drawdown(self.values())


@MetricFactory.register(
    name = 'CalmarRatio',
    description = 'Annualized compounded return divided by maximum drawdown.',
    id = 'calmar', domain = 'risk', actions = {"performance"},
)
class CalmarRatio(ReturnMetric):

    def __init__(
            self,
            periods_per_year: int = 252
            ) -> None:
        super().__init__()
        if isinstance(periods_per_year, bool) or not isinstance(periods_per_year, Integral) or periods_per_year <= 0:
            raise ValueError("periods_per_year must be a positive integer.")
        self.periods_per_year = int(periods_per_year)

    def compute(
            self
            ) -> Tensor:
        values = self.values()
        annual_return = (1 + values).prod().pow(self.periods_per_year / values.numel()) - 1
        return divide(annual_return, max_drawdown(values))


@MetricFactory.register(
    name = 'TotalReturn',
    description = 'Total compounded simple return.',
    id = 'total_return', domain = 'returns', actions = {"performance"},
)
class TotalReturn(ReturnMetric):

    def compute(
            self
            ) -> Tensor:
        return (1 + self.values()).prod() - 1
