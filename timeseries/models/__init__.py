"""Import model modules and expose each registered model with its configuration."""
from importlib import import_module
from pkgutil import walk_packages
from typing import TYPE_CHECKING

from timeseries.model_register import ModelFactory, ModelInfo, ModelRegister
from timeseries.registry import Domain

if TYPE_CHECKING:
    from .ARIMA import ARIMAPredictor as ARIMAPredictor
    from .ARIMA import ConfigARIMAPredictor as ConfigARIMAPredictor
    from .Binomial import Binomial as Binomial
    from .Binomial import ConfigBinomial as ConfigBinomial
    from .BlackScholes import BlackScholes as BlackScholes
    from .BlackScholes import ConfigBlackScholes as ConfigBlackScholes
    from .BollingerBands import BollingerBands as BollingerBands
    from .BollingerBands import ConfigBollingerBands as ConfigBollingerBands
    from .CrossFormer import ConfigCrossFormer as ConfigCrossFormer
    from .CrossFormer import CrossFormer as CrossFormer
    from .D3VAE import D3VAE as D3VAE
    from .D3VAE import ConfigD3VAE as ConfigD3VAE
    from .ensamble import ConfigEnsemblePredictor as ConfigEnsemblePredictor
    from .ensamble import EnsemblePredictor as EnsemblePredictor
    from .GARCH import ConfigSimpleGARCH as ConfigSimpleGARCH
    from .GARCH import SimpleGARCH as SimpleGARCH
    from .gnn.model import GNN as GNN
    from .gnn.model import ConfigGNN as ConfigGNN
    from .Informer import ConfigInformer as ConfigInformer
    from .Informer import Informer as Informer
    from .KalmanFilter import ConfigKalmanFilter as ConfigKalmanFilter
    from .KalmanFilter import KalmanFilter as KalmanFilter
    from .legacy_tft.model import ConfigLegacyTFT as ConfigLegacyTFT
    from .legacy_tft.model import Model as Model
    from .LinearTS import ConfigLinearTS as ConfigLinearTS
    from .LinearTS import LinearTS as LinearTS
    from .MovingAverage import ConfigMovingAverages as ConfigMovingAverages
    from .MovingAverage import MovingAverages as MovingAverages
    from .MyModel import ConfigMyModel as ConfigMyModel
    from .MyModel import MyModel as MyModel
    from .Persistent import ConfigPersistent as ConfigPersistent
    from .Persistent import Persistent as Persistent
    from .RegimeDetector import ConfigRegimeDetector as ConfigRegimeDetector
    from .RegimeDetector import RegimeDetector as RegimeDetector
    from .RNN import RNN as RNN
    from .RNN import ConfigRNN as ConfigRNN
    from .TFT import TFT as TFT
    from .TFT import ConfigTFT as ConfigTFT
    from .TFT2 import TFT2 as TFT2
    from .TFT2 import ConfigTFT2 as ConfigTFT2
    from .VQVAEA import VQVAEA as VQVAEA
    from .VQVAEA import ConfigVQVAEA as ConfigVQVAEA
    from .VVA import VVA as VVA
    from .VVA import ConfigVVA as ConfigVVA


def _import_models() -> list[str]:
    # Import decorators recursively, including models in architecture packages.
    # Ignore backup files such as model-checkpoint.py (not Python identifiers).
    for module in walk_packages(
            __path__,
            prefix = f"{__name__}."
    ):
        if all(
                part.isidentifier() and not part.startswith(
                    '_'
                ) for part in module.name.split(
                    '.'
                )
        ):
            import_module(
                module.name
            )

    exports = []
    for class_id in ModelFactory.get_list_classes():
        info = ModelFactory.get_info(
            class_id
        )
        for implementation in (info.class_type, info.config_type):
            globals()[implementation.__name__] = implementation
            exports.append(
                implementation.__name__
            )
    return exports


__all__ = ["Domain", "ModelFactory", "ModelInfo", "ModelRegister"]
__all__.extend(
    _import_models()
)
