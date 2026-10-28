from pydantic import Field, model_validator
from timeseries.registry import Configuration
from timeseries.register import ModelFactory
import numpy as np
from numpy.typing import ArrayLike
from timeseries.register import ModelRegister


class ConfigEnsemblePredictor(Configuration):
    """Ensemble weights are supplied to predict()."""



"""The historical module spelling is preserved for existing imports."""



@ModelFactory.register(
    name='EnsemblePredictor',
    description='Weighted combination of Kalman, AR and EMA forecasts.',
    id='ensemble', domain='forecasting',
)
class EnsemblePredictor:
    """Weighted Kalman, autoregressive and EMA prediction."""
    CONFIG_T = ConfigEnsemblePredictor

    def __init__(self) -> None:
        self.kalman = ModelRegister.create("kalman")
        self.arima = ModelRegister.create("arima")

    def predict(self, data: ArrayLike, weights: list[float] | None = None) -> float:
        weights_array = np.asarray([0.3, 0.3, 0.4] if weights is None else weights, dtype=float)
        if weights_array.shape != (3,) or not np.isfinite(weights_array).all() or (weights_array < 0).any() or not np.isclose(weights_array.sum(), 1):
            raise ValueError("Provide three nonnegative finite weights summing to one.")
        # Each call forecasts a complete history; avoid feeding it twice to the filter.
        self.kalman = ModelRegister.create("kalman")
        kalman_prediction = self.kalman.filter_series(data)[-1]
        self.arima.fit(data)
        arima_prediction = self.arima.predict(data, steps=1)[0]
        averages = ModelRegister.create("movingaverages")
        ema_prediction = averages.ema(data, span=20)[-1]
        return float(weights_array @ [kalman_prediction, arima_prediction, ema_prediction])
