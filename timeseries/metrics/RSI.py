from typing import Literal, Optional

import torch
from torchmetrics import Metric


class MovingAverage(Metric):
    def __init__(
            self,
            period: Optional[int | Literal["max"]] = None
    ):
        super(MovingAverage, self).__init__()
        """
        Args:
            period (int, optional): Defaults to 14. This represents the moving window to compute the sharpe ration in
        """
        if not period:
            period = "max"
        else:
            if isinstance(period, (int, float)):
                # the period must be an integer and positive
                period = abs(int(period))
            if isinstance(period, str) and period != "max":
                raise ValueError(f"The type of period, {type(period)}, is not invalid.")
        self.period = "max" if not period or period == "max" else period

        self.values = []

    def reset(self) -> None:
        self.values = []

    def update(self, data: list[float]) -> None:
        self.values.extend(data)

    def compute(self) -> list[float]:
        period = len(self.values) if self.period == "max" else self.period
        out: list[float] = []
        for i in range(len(self.values) // period):
            window: torch.Tensor = torch.tensor(self.values[i * period: (i + 1) * period])
            window_return = window[-1] - window[0]
            out.append((window_return / window.std().clamp(1e-5)).item())

        if len(self.values) > len(self.values) // period * period:
            window: torch.Tensor = torch.tensor(self.values[len(self.values) // period * period: -1])
            window_return = window[-1] - window[0]
            out.append((window_return / window.std().clamp(1e-5)).item())

        return out


class MovingVariance(Metric):
    def __init__(
            self,
            period: Optional[int | Literal["max"]] = None
    ):
        super(MovingVariance, self).__init__()
        """
        Args:
            period (int, optional): Defaults to 14. This represents the moving window to compute the sharpe ration in
        """
        if not period:
            period = "max"
        else:
            if isinstance(period, (int, float)):
                # the period must be an integer and positive
                period = abs(int(period))
            if isinstance(period, str) and period != "max":
                raise ValueError(f"The type of period, {type(period)}, is not invalid.")
        self.period = "max" if not period or period == "max" else period

        self.values = []

    def reset(self) -> None:
        self.values = []

    def update(self, data: list[float]) -> None:
        self.values.extend(data)

    def compute(self) -> list[float]:
        period = len(self.values) if self.period == "max" else self.period
        out: list[float] = []
        for i in range(len(self.values) // period):
            window: torch.Tensor = torch.tensor(self.values[i * period: (i + 1) * period])
            window_return = window[-1] - window[0]
            out.append((window_return / window.std().clamp(1e-5)).item())

        if len(self.values) > len(self.values) // period * period:
            window: torch.Tensor = torch.tensor(self.values[len(self.values) // period * period: -1])
            window_return = window[-1] - window[0]
            out.append((window_return / window.std().clamp(1e-5)).item())

        return out


class MovingSR(Metric):
    def __init__(
            self,
            period: Optional[int | Literal["max"]] = None
    ):
        super(MovingSR, self).__init__()
        """
        Args:
            period (int, optional): Defaults to 14. This represents the moving window to compute the sharpe ration in
        """
        if not period:
            period = "max"
        else:
            if isinstance(period, (int, float)):
                # the period must be an integer and positive
                period = abs(int(period))
            if isinstance(period, str) and period != "max":
                raise ValueError(f"The type of period, {type(period)}, is not invalid.")
        self.period = "max" if not period or period == "max" else period

        self.values = []

    def reset(self) -> None:
        self.values = []

    def update(self, data: list[float]) -> None:
        self.values.extend(data)

    def compute(self) -> list[float]:
        period = len(self.values) if self.period == "max" else self.period
        out: list[float] = []
        for i in range(len(self.values) // period):
            window: torch.Tensor = torch.tensor(self.values[i * period: (i + 1) * period])
            window_return = window[-1] - window[0]
            out.append((window_return / window.std().clamp(1e-5)).item())

        if len(self.values) > len(self.values) // period * period:
            window: torch.Tensor = torch.tensor(self.values[len(self.values) // period * period: -1])
            window_return = window[-1] - window[0]
            out.append((window_return / window.std().clamp(1e-5)).item())

        return out
