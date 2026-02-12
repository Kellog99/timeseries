from collections import deque
from datetime import datetime

import torch
from torchmetrics import Metric


class MovingSR(Metric):
    def __init__(self, period: int = 14):
        super(MovingSR, self).__init__()
        """
        Args:
            period (int, optional): Defaults to 14. This represents the moving window to compute the sharpe ration in
        """
        self.period = period
        self.windows = {}

    def reset(self) -> None:
        self.windows = {}

    def update(self, data: dict[datetime, float]):
        self.windows.update(data)

    def compute(self) -> torch.Tensor:
        all_dates = list(self.windows.keys())
        if len(all_dates) < self.period:
            return torch.Tensor(0)

        all_dates.sort()
        out = []
        queue = deque(maxlen=self.period)
        queue.append(self.windows[all_dates[0]])
        for i in range(1, len(all_dates)):
            queue.append(self.windows[all_dates[i]])
            if len(queue) == self.period:
                out.append(list(queue))
                queue.clear()
        values = torch.Tensor(out)
        window_returns = (values[:, -1] - values[:, 0]) / values[:, 0]
        window_std = torch.std(window_returns, dim=-1)

        return window_returns / window_std.pow(1 / self.period)
