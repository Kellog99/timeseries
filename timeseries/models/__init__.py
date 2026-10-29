"""Import model modules and expose each registered model with its configuration."""
from importlib import import_module
from pkgutil import walk_packages
from typing import TYPE_CHECKING

from timeseries.model_register import ModelFactory, ModelInfo, ModelRegister
from timeseries.registry import Domain

if TYPE_CHECKING:
    from .ARIMA import ARIMAPredictor, ConfigARIMAPredictor
    from .BlackScholes import BlackScholes, ConfigBlackScholes
    from .BollingerBands import BollingerBands, ConfigBollingerBands
    from .CrossFormer import CrossFormer, ConfigCrossFormer
    from .D3VAE import D3VAE, ConfigD3VAE
    from .GARCH import SimpleGARCH, ConfigSimpleGARCH
    from .Informer import Informer, ConfigInformer
    from .KalmanFilter import KalmanFilter, ConfigKalmanFilter
    from .LinearTS import LinearTS, ConfigLinearTS
    from .MovingAverage import MovingAverages, ConfigMovingAverages
    from .MyModel import MyModel, ConfigMyModel
    from .Persistent import Persistent, ConfigPersistent
    from .RegimeDetector import RegimeDetector, ConfigRegimeDetector
    from .RNN import RNN, ConfigRNN
    from .TFT import TFT, ConfigTFT
    from .TFT2 import TFT2, ConfigTFT2
    from .VVA import VVA, ConfigVVA
    from .VQVAEA import VQVAEA, ConfigVQVAEA
    from .ensamble import EnsemblePredictor, ConfigEnsemblePredictor
    from .gnn.model import GNN, ConfigGNN
    from .legacy_tft.model import Model, ConfigLegacyTFT


def _import_models() -> list[str]:
    # Import decorators recursively, including models in architecture packages.
    # Ignore backup files such as model-checkpoint.py (not Python identifiers).
    for module in walk_packages(__path__, prefix = f"{__name__}."):
        if all(part.isidentifier() and not part.startswith('_') for part in module.name.split('.')):
            import_module(module.name)

    exports = []
    for class_id in ModelFactory.get_list_classes():
        info = ModelFactory.get_info(class_id)
        for implementation in (info.class_type, info.config_type):
            globals()[implementation.__name__] = implementation
            exports.append(implementation.__name__)
    return exports


__all__ = ["Domain", "ModelFactory", "ModelInfo", "ModelRegister", *_import_models()]
