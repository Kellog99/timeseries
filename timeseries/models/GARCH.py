from pydantic import Field, model_validator
from timeseries.registry import Configuration
from timeseries.register import ModelFactory
import numpy as np
from numpy.typing import ArrayLike, NDArray


class ConfigSimpleGARCH(Configuration):
    omega: float = Field(0.01, gt=0, description="Constant variance contribution.")
    alpha: float = Field(0.05, ge=0, lt=1, description="Weight of the previous squared innovation.")
    beta: float = Field(0.9, ge=0, lt=1, description="Weight of the previous conditional variance.")

    @model_validator(mode="after")
    def check_stationarity(self):
        if self.alpha + self.beta >= 1:
            raise ValueError("Stationary GARCH requires alpha + beta < 1.")
        return self






@ModelFactory.register(
    name='SimpleGARCH',
    description='Stationary GARCH(1,1) volatility forecast.',
    id='garch', domain='volatility',
)
class SimpleGARCH:
    """Simplified stationary GARCH(1,1) volatility predictor."""
    CONFIG_T = ConfigSimpleGARCH

    def __init__(self, omega: float = 0.01, alpha: float = 0.05, beta: float = 0.9) -> None:
        config = self.CONFIG_T(omega=omega, alpha=alpha, beta=beta)
        self.omega, self.alpha, self.beta = config.omega, config.alpha, config.beta

    def fit_predict(self, returns: ArrayLike, steps: int = 1) -> NDArray[np.float64]:
        values = np.asarray(returns, dtype=float)
        if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
            raise ValueError("Expected a nonempty finite return sequence.")
        if steps <= 0:
            raise ValueError("steps must be positive.")
        variance = float(values.var())
        for value in values[-100:]:
            variance = self.omega + self.alpha * value ** 2 + self.beta * variance
        forecasts: list[float] = []
        for _ in range(steps):
            variance = self.omega + (self.alpha + self.beta) * variance
            forecasts.append(float(np.sqrt(variance)))
        return np.asarray(forecasts)
