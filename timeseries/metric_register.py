from importlib import import_module
from inspect import Parameter, signature
from typing import Any, Optional

from pydantic import Field, field_validator

from timeseries.registry import Domain, Info, Registry
from timeseries.utils.typing import Metric


class MetricInfo(Info[Metric]):
    actions: set[str] = Field(
        default_factory = lambda: {"performance"}, description = "Supported uses, for example performance or loss."
    )

    @field_validator("actions")
    @classmethod
    def nonempty_actions(
            cls,
            value: set[str]
    ) -> set[str]:
        if not value:
            raise ValueError("At least one action must be specified.")
        return value


class MetricFactory(Registry[Metric]):
    _info_type = MetricInfo

    @classmethod
    def create(cls, class_id: str, **kwargs: Any) -> Metric:
        return cls.get_info(class_id).class_type(**kwargs)

    @classmethod
    def get_config_param(cls, id: str) -> dict[str, Any]:
        """Return constructor defaults; Parameter.empty marks required arguments."""
        return {
            name: parameter.default
            for name, parameter in signature(cls.get_info(id).class_type).parameters.items()
            if parameter.kind in (Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY)
        }

    @classmethod
    def _load_builtins(
            cls,
            class_id: Optional[str] = None
    ) -> None:
        for module in ("risk", "RSI", "ts_correlation"):
            import_module(f"timeseries.metrics.{module}")

    @classmethod
    def filter(
            cls,
            info: Info[Metric],
            domain: Optional[Domain | set[Domain]] = None,
            actions: Optional[set[str]] = None,
            **kwargs
    ) -> bool:
        return super().filter(info, domain = domain) and (
                actions is None or isinstance(info, MetricInfo) and bool(actions & info.actions))


MetricRegister = MetricFactory
