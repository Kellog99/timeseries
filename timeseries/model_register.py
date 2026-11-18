"""Model factory; implementations register themselves through decorators."""
from importlib import import_module
from typing import Any

from timeseries.registry import ConfigInfo, Factory


class ModelInfo(
    ConfigInfo[Any]
):
    """Implementation class, readable description, and application domain."""


class ModelFactory(
    Factory[Any]
):
    _info_type = ModelInfo

    @classmethod
    def _load_builtins(
        cls,
        class_id: str | None = None
    ) -> None:
        import_module(
            "timeseries.models"
        )


ModelRegister = ModelFactory
