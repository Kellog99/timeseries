"""Typing bridges for callable attributes in PyTorch and TorchMetrics.

At runtime these names are aliases to the original library classes. PyTorch
annotates ``forward`` as a writable Callable, although subclasses implement a
method. TorchMetrics annotates its wrapped instance methods with unbound
function types. The checking-only declarations preserve their actual binding
without changing model inheritance, hooks, serialization, or metric wrappers.
"""

from typing import TYPE_CHECKING, Any, Callable

from torch import nn
from torchmetrics import Metric as TorchMetric

if TYPE_CHECKING:
    class Module(nn.Module):
        def forward(
                self,
                *args: Any,
                **kwargs: Any
                ) -> Any:  # ty: ignore[invalid-mutable-override]
            raise NotImplementedError


    class _BoundMetricCalls(TorchMetric):
        update: Callable[..., None]
        compute: Callable[[], Any]


    class Metric(_BoundMetricCalls):
        def update(
                self,
                *args: Any,
                **kwargs: Any
                ) -> None:  # ty: ignore[invalid-mutable-override]
            raise NotImplementedError

        def compute(
                self
                ) -> Any:  # ty: ignore[invalid-mutable-override]
            raise NotImplementedError
else:
    Module = nn.Module
    Metric = TorchMetric
