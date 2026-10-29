"""Model factory; implementations register themselves through decorators."""
from importlib import import_module
from typing import Optional

from timeseries.registry import ConfigInfo, Factory


class ModelInfo(ConfigInfo[object]):
    """Implementation class, readable description, and application domain."""


class ModelFactory(Factory[object]):
    _info_type = ModelInfo

    @classmethod
    def _load_builtins(
            cls,
            class_id: Optional[str] = None
    ) -> None:
        import_module("timeseries.models")


ModelRegister = ModelFactory
