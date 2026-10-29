import numpy as np
from numpy.typing import ArrayLike, NDArray
from pydantic import Field

from timeseries.model_register import ModelFactory
from timeseries.registry import Configuration


class ConfigARIMAPredictor(Configuration):
    p: int = Field(5, gt = 0, description = "Autoregressive lag order.")
    d: int = Field(1, ge = 0, le = 1, description = "Differencing order; this implementation supports zero or one.")
    q: int = Field(
        2, ge = 0,
        description = "Reserved moving-average order; the current predictor fits only AR coefficients."
        )


@ModelFactory.register(
    name = 'ARIMAPredictor',
    description = 'Autoregressive least-squares predictor with optional first differencing.',
    id = 'arima', domain = 'forecasting',
)
class ARIMAPredictor:
    """Least-squares AR predictor with optional first differencing."""
    CONFIG_T = ConfigARIMAPredictor

    def __init__(
            self,
            p: int = 5,
            d: int = 1,
            q: int = 2
            ) -> None:
        config = self.CONFIG_T(p = p, d = d, q = q)
        self.p, self.d, self.q = config.p, config.d, config.q
        self.ar_coef: NDArray[np.float64] | None = None
        self.ma_coef: NDArray[np.float64] | None = None

    def difference(
            self,
            data: ArrayLike,
            order: int = 1
            ) -> NDArray[np.float64]:
        return np.diff(np.asarray(data, dtype = float), n = order)

    def fit(
            self,
            data: ArrayLike
            ) -> None:
        values = self.difference(data, self.d)
        if values.ndim != 1 or len(values) <= self.p or not np.isfinite(values).all():
            raise ValueError("Expected finite univariate data with more observations than p + d.")
        x = np.array([values[i - self.p:i] for i in range(self.p, len(values))])
        self.ar_coef = np.linalg.lstsq(x, values[self.p:], rcond = None)[0]

    def predict(
            self,
            data: ArrayLike,
            steps: int = 1
            ) -> NDArray[np.float64]:
        if self.ar_coef is None:
            raise ValueError("Fit the predictor before predicting.")
        if steps <= 0:
            raise ValueError("steps must be positive.")
        values = np.asarray(data, dtype = float)
        current = self.difference(values, self.d).tolist()
        if len(current) < self.p:
            raise ValueError("Not enough observations for the configured lag order.")
        predictions: list[float] = []
        for _ in range(steps):
            prediction = float(np.dot(current[-self.p:], self.ar_coef))
            predictions.append(prediction)
            current.append(prediction)
        result = np.asarray(predictions)
        return values[-1] + result.cumsum() if self.d == 1 else result
