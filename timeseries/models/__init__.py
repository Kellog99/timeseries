"""One model package for forecasting, graph, and statistical implementations."""
from importlib import import_module
from typing import Any

from timeseries.register import ModelFactory, ModelInfo, ModelRegister

_FORECASTERS = ("RNN", "LinearTS", "Persistent", "D3VAE", "MyModel", "TFT", "Informer", "TFT2", "VVA", "VQVAEA",
                "CrossFormer")
_EXPORTS = {name: (f".{name}", name) for name in _FORECASTERS}
_EXPORTS.update({f"Config{name}": (f".{name}", f"Config{name}") for name in _FORECASTERS})
_EXPORTS.update({"GNN": (".gnn.model", "GNN"), "ConfigGNN": (".gnn.model", "ConfigGNN")})
__all__ = ["ModelFactory", "ModelInfo", "ModelRegister", *_EXPORTS]


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, attribute = _EXPORTS[name]
    value = getattr(import_module(module, __name__), attribute)
    globals()[name] = value
    return value
