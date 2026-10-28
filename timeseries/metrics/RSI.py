"""Historical module name retained for the windowed metrics API."""

from .register import MetricFactory

from typing import ClassVar

import torch
from torch import Tensor

from .configs import MovingAverageConfig, MovingSRConfig, MovingVarianceConfig, Period, WindowConfig
from .risk import ReturnMetric, divide


class WindowMetric(ReturnMetric):
    CONFIG_T: ClassVar[type[WindowConfig]] = MovingAverageConfig

    def __init__(self, period: Period = None) -> None:
        super().__init__()
        self.period = self.CONFIG_T(period=period).period

    def reduce(self, window: Tensor) -> Tensor:
        raise NotImplementedError

    def compute(self) -> list[float]:
        values = self.values()
        period = values.numel() if self.period is None or self.period == "max" else self.period
        return [self.reduce(window).item() for window in values.split(period)]

    def update(self, data: Tensor | list[float]) -> None:
        # Means and variances also accept arbitrary finite observations, not only returns.
        values = torch.as_tensor(data, dtype=torch.float64, device=self.device)
        if values.ndim != 1 or not torch.isfinite(values).all():
            raise ValueError("Expected a finite one-dimensional sequence.")
        self.returns.append(values.detach())


@MetricFactory.register(
    name='MovingAverage',
    description='Arithmetic mean in non-overlapping windows.',
    id='moving_average', domain='statistics', actions={"performance"},
)
class MovingAverage(WindowMetric):
    CONFIG_T: ClassVar[type[WindowConfig]] = MovingAverageConfig

    def reduce(self, window: Tensor) -> Tensor:
        return window.mean()


@MetricFactory.register(
    name='MovingVariance',
    description='Population variance in non-overlapping windows.',
    id='moving_variance', domain='statistics', actions={"performance"},
)
class MovingVariance(WindowMetric):
    CONFIG_T: ClassVar[type[WindowConfig]] = MovingVarianceConfig

    def reduce(self, window: Tensor) -> Tensor:
        return window.var(correction=0)


@MetricFactory.register(
    name='MovingSR',
    description='Unannualized Sharpe ratio in non-overlapping windows.',
    id='moving_sr', domain='risk', actions={"performance"},
)
class MovingSR(WindowMetric):
    CONFIG_T: ClassVar[type[WindowConfig]] = MovingSRConfig

    def update(self, data: Tensor | list[float]) -> None:
        ReturnMetric.update(self, data)

    def reduce(self, window: Tensor) -> Tensor:
        std = window.std() if window.numel() > 1 else window.new_tensor(float("nan"))
        return divide(window.mean(), std)
