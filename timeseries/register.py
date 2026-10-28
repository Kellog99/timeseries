"""Model factory; implementations register themselves through decorators."""
from importlib import import_module
from typing import TYPE_CHECKING, Any, Literal, overload

from pydantic import BaseModel

if TYPE_CHECKING:
    from timeseries.models.VVA import VVA, ConfigVVA
    from timeseries.models.Persistent import Persistent, ConfigPersistent
    from timeseries.models.TFT import TFT, ConfigTFT
    from timeseries.models.Informer import Informer, ConfigInformer
    from timeseries.models.TFT2 import TFT2, ConfigTFT2
    from timeseries.models.D3VAE import D3VAE, ConfigD3VAE
    from timeseries.models.VQVAEA import VQVAEA, ConfigVQVAEA
    from timeseries.models.RNN import RNN, ConfigRNN
    from timeseries.models.LinearTS import LinearTS, ConfigLinearTS
    from timeseries.models.MyModel import MyModel, ConfigMyModel
    from timeseries.models.CrossFormer import CrossFormer, ConfigCrossFormer
    from timeseries.models.gnn.model import GNN, ConfigGNN
    from timeseries.models.legacy_tft.model import Model, ConfigLegacyTFT
    from timeseries.models.ARIMA import ARIMAPredictor, ConfigARIMAPredictor
    from timeseries.models.GARCH import SimpleGARCH, ConfigSimpleGARCH
    from timeseries.models.KalmanFilter import KalmanFilter, ConfigKalmanFilter
    from timeseries.models.MovingAverage import MovingAverages, ConfigMovingAverages
    from timeseries.models.BollingerBands import BollingerBands, ConfigBollingerBands
    from timeseries.models.RegimeDetector import RegimeDetector, ConfigRegimeDetector
    from timeseries.models.ensamble import EnsemblePredictor, ConfigEnsemblePredictor

from timeseries.registry import Factory, Info


class ModelInfo(Info[object]):
    """Implementation class, readable description, and application domain."""


class ModelFactory(Factory[object]):
    _info_type = ModelInfo
    # Import routing only: metadata is declared on each implementation's decorator.
    _modules = {
        'vva': 'timeseries.models.VVA',
        'persistent': 'timeseries.models.Persistent',
        'tft': 'timeseries.models.TFT',
        'informer': 'timeseries.models.Informer',
        'tft2': 'timeseries.models.TFT2',
        'd3vae': 'timeseries.models.D3VAE',
        'vqvaea': 'timeseries.models.VQVAEA',
        'rnn': 'timeseries.models.RNN',
        'linearts': 'timeseries.models.LinearTS',
        'my': 'timeseries.models.MyModel',
        'crossformer': 'timeseries.models.CrossFormer',
        'gnn': 'timeseries.models.gnn.model',
        'legacytft': 'timeseries.models.legacy_tft.model',
        'arima': 'timeseries.models.ARIMA',
        'garch': 'timeseries.models.GARCH',
        'kalman': 'timeseries.models.KalmanFilter',
        'movingaverages': 'timeseries.models.MovingAverage',
        'bollingerbands': 'timeseries.models.BollingerBands',
        'regimedetector': 'timeseries.models.RegimeDetector',
        'ensemble': 'timeseries.models.ensamble',
    }

    @overload
    @classmethod
    def create(
            cls,
            class_id: Literal["vva"],
            config: BaseModel | None = None,
            **kwargs: Any
            ) -> "VVA":
        ...

    @overload
    @classmethod
    def create(
            cls,
            class_id: Literal["persistent"],
            config: BaseModel | None = None,
            **kwargs: Any
            ) -> "Persistent":
        ...

    @overload
    @classmethod
    def create(
            cls,
            class_id: Literal["tft"],
            config: BaseModel | None = None,
            **kwargs: Any
            ) -> "TFT":
        ...

    @overload
    @classmethod
    def create(
            cls,
            class_id: Literal["informer"],
            config: BaseModel | None = None,
            **kwargs: Any
            ) -> "Informer":
        ...

    @overload
    @classmethod
    def create(
            cls,
            class_id: Literal["tft2"],
            config: BaseModel | None = None,
            **kwargs: Any
            ) -> "TFT2":
        ...

    @overload
    @classmethod
    def create(
            cls,
            class_id: Literal["d3vae"],
            config: BaseModel | None = None,
            **kwargs: Any
            ) -> "D3VAE":
        ...

    @overload
    @classmethod
    def create(
            cls,
            class_id: Literal["vqvaea"],
            config: BaseModel | None = None,
            **kwargs: Any
            ) -> "VQVAEA":
        ...

    @overload
    @classmethod
    def create(
            cls,
            class_id: Literal["rnn"],
            config: BaseModel | None = None,
            **kwargs: Any
            ) -> "RNN":
        ...

    @overload
    @classmethod
    def create(
            cls,
            class_id: Literal["linearts"],
            config: BaseModel | None = None,
            **kwargs: Any
            ) -> "LinearTS":
        ...

    @overload
    @classmethod
    def create(
            cls,
            class_id: Literal["my"],
            config: BaseModel | None = None,
            **kwargs: Any
            ) -> "MyModel":
        ...

    @overload
    @classmethod
    def create(
            cls,
            class_id: Literal["crossformer"],
            config: BaseModel | None = None,
            **kwargs: Any
            ) -> "CrossFormer":
        ...

    @overload
    @classmethod
    def create(
            cls,
            class_id: Literal["gnn"],
            config: BaseModel | None = None,
            **kwargs: Any
            ) -> "GNN":
        ...

    @overload
    @classmethod
    def create(
            cls,
            class_id: Literal["legacytft"],
            config: BaseModel | None = None,
            **kwargs: Any
            ) -> "Model":
        ...

    @overload
    @classmethod
    def create(
            cls,
            class_id: Literal["arima"],
            config: BaseModel | None = None,
            **kwargs: Any
            ) -> "ARIMAPredictor":
        ...

    @overload
    @classmethod
    def create(
            cls,
            class_id: Literal["garch"],
            config: BaseModel | None = None,
            **kwargs: Any
            ) -> "SimpleGARCH":
        ...

    @overload
    @classmethod
    def create(
            cls,
            class_id: Literal["kalman"],
            config: BaseModel | None = None,
            **kwargs: Any
            ) -> "KalmanFilter":
        ...

    @overload
    @classmethod
    def create(
            cls,
            class_id: Literal["movingaverages"],
            config: BaseModel | None = None,
            **kwargs: Any
            ) -> "MovingAverages":
        ...

    @overload
    @classmethod
    def create(
            cls,
            class_id: Literal["bollingerbands"],
            config: BaseModel | None = None,
            **kwargs: Any
            ) -> "BollingerBands":
        ...

    @overload
    @classmethod
    def create(
            cls,
            class_id: Literal["regimedetector"],
            config: BaseModel | None = None,
            **kwargs: Any
            ) -> "RegimeDetector":
        ...

    @overload
    @classmethod
    def create(
            cls,
            class_id: Literal["ensemble"],
            config: BaseModel | None = None,
            **kwargs: Any
            ) -> "EnsemblePredictor":
        ...

    @overload
    @classmethod
    def create(
            cls,
            class_id: None = None,
            *,
            config: "ConfigKalmanFilter",
            **kwargs: Any
            ) -> "KalmanFilter":
        ...

    @overload
    @classmethod
    def create(
            cls,
            class_id: str | None = None,
            config: BaseModel | None = None,
            **kwargs: Any
            ) -> object:
        ...

    @classmethod
    def create(
            cls,
            class_id: str | None = None,
            config: BaseModel | None = None,
            **kwargs: Any
            ) -> object:
        return super().create(class_id, config, **kwargs)

    @overload
    @classmethod
    def get_config(
            cls,
            class_id: Literal["vva"],
            **kwargs: Any
            ) -> "ConfigVVA":
        ...

    @overload
    @classmethod
    def get_config(
            cls,
            class_id: Literal["persistent"],
            **kwargs: Any
            ) -> "ConfigPersistent":
        ...

    @overload
    @classmethod
    def get_config(
            cls,
            class_id: Literal["tft"],
            **kwargs: Any
            ) -> "ConfigTFT":
        ...

    @overload
    @classmethod
    def get_config(
            cls,
            class_id: Literal["informer"],
            **kwargs: Any
            ) -> "ConfigInformer":
        ...

    @overload
    @classmethod
    def get_config(
            cls,
            class_id: Literal["tft2"],
            **kwargs: Any
            ) -> "ConfigTFT2":
        ...

    @overload
    @classmethod
    def get_config(
            cls,
            class_id: Literal["d3vae"],
            **kwargs: Any
            ) -> "ConfigD3VAE":
        ...

    @overload
    @classmethod
    def get_config(
            cls,
            class_id: Literal["vqvaea"],
            **kwargs: Any
            ) -> "ConfigVQVAEA":
        ...

    @overload
    @classmethod
    def get_config(
            cls,
            class_id: Literal["rnn"],
            **kwargs: Any
            ) -> "ConfigRNN":
        ...

    @overload
    @classmethod
    def get_config(
            cls,
            class_id: Literal["linearts"],
            **kwargs: Any
            ) -> "ConfigLinearTS":
        ...

    @overload
    @classmethod
    def get_config(
            cls,
            class_id: Literal["my"],
            **kwargs: Any
            ) -> "ConfigMyModel":
        ...

    @overload
    @classmethod
    def get_config(
            cls,
            class_id: Literal["crossformer"],
            **kwargs: Any
            ) -> "ConfigCrossFormer":
        ...

    @overload
    @classmethod
    def get_config(
            cls,
            class_id: Literal["gnn"],
            **kwargs: Any
            ) -> "ConfigGNN":
        ...

    @overload
    @classmethod
    def get_config(
            cls,
            class_id: Literal["legacytft"],
            **kwargs: Any
            ) -> "ConfigLegacyTFT":
        ...

    @overload
    @classmethod
    def get_config(
            cls,
            class_id: Literal["arima"],
            **kwargs: Any
            ) -> "ConfigARIMAPredictor":
        ...

    @overload
    @classmethod
    def get_config(
            cls,
            class_id: Literal["garch"],
            **kwargs: Any
            ) -> "ConfigSimpleGARCH":
        ...

    @overload
    @classmethod
    def get_config(
            cls,
            class_id: Literal["kalman"],
            **kwargs: Any
            ) -> "ConfigKalmanFilter":
        ...

    @overload
    @classmethod
    def get_config(
            cls,
            class_id: Literal["movingaverages"],
            **kwargs: Any
            ) -> "ConfigMovingAverages":
        ...

    @overload
    @classmethod
    def get_config(
            cls,
            class_id: Literal["bollingerbands"],
            **kwargs: Any
            ) -> "ConfigBollingerBands":
        ...

    @overload
    @classmethod
    def get_config(
            cls,
            class_id: Literal["regimedetector"],
            **kwargs: Any
            ) -> "ConfigRegimeDetector":
        ...

    @overload
    @classmethod
    def get_config(
            cls,
            class_id: Literal["ensemble"],
            **kwargs: Any
            ) -> "ConfigEnsemblePredictor":
        ...

    @overload
    @classmethod
    def get_config(
            cls,
            class_id: str,
            **kwargs: Any
            ) -> BaseModel:
        ...

    @classmethod
    def get_config(
            cls,
            class_id: str,
            **kwargs: Any
            ) -> BaseModel:
        return super().get_config(class_id, **kwargs)

    @classmethod
    def _load_builtins(
            cls,
            class_id: str | None = None
            ) -> None:
        modules = cls._modules.values() if class_id is None else [cls._modules.get(class_id)]
        for module in modules:
            if module is not None:
                import_module(module)


ModelRegister = ModelFactory
