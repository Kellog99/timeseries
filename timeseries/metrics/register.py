"""Factory for metrics registered by decorators on their implementations."""
from importlib import import_module

from pydantic import Field, field_validator
from timeseries.utils.typing import Metric

from timeseries.registry import Factory, Info


class MetricInfo(Info[Metric]):
    actions: set[str] = Field(default_factory=lambda: {"performance"}, description="Supported uses, for example performance or loss.")

    @field_validator("actions")
    @classmethod
    def nonempty_actions(cls, value: set[str]) -> set[str]:
        if not value:
            raise ValueError("At least one action must be specified.")
        return value


class MetricFactory(Factory[Metric]):
    _info_type = MetricInfo

    @classmethod
    def _load_builtins(cls, class_id: str | None = None) -> None:
        for module in ("risk", "RSI", "ts_correlation"):
            import_module(f"timeseries.metrics.{module}")

    @classmethod
    def filter(cls, info: Info[Metric], domain: str | set[str] | None = None,
               actions: set[str] | None = None, **kwargs) -> bool:
        return super().filter(info, domain=domain) and (actions is None or isinstance(info, MetricInfo) and bool(actions & info.actions))


MetricRegister = MetricFactory
