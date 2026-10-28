import numpy as np
from numpy.typing import ArrayLike, NDArray
from pydantic import Field

from timeseries.register import ModelFactory
from timeseries.registry import Configuration


class ConfigKalmanFilter(Configuration):
    process_variance: float = Field(1e-5, ge = 0, description = "Process noise variance.")
    measurement_variance: float = Field(1e-1, gt = 0, description = "Measurement noise variance.")


@ModelFactory.register(
    name = 'KalmanFilter',
    description = 'One-dimensional Kalman filter.',
    id = 'kalman', domain = 'filtering',
)
class KalmanFilter:
    """One-dimensional Kalman filter with fixed noise variances."""
    CONFIG_T = ConfigKalmanFilter

    def __init__(
            self,
            process_variance: float = 1e-5,
            measurement_variance: float = 1e-1
            ) -> None:
        config = self.CONFIG_T(process_variance = process_variance, measurement_variance = measurement_variance)
        self.process_variance = config.process_variance
        self.measurement_variance = config.measurement_variance
        self.estimate = 0.0
        self.error_estimate = 1.0

    def update(
            self,
            measurement: float
            ) -> float:
        error_prediction = self.error_estimate + self.process_variance
        gain = error_prediction / (error_prediction + self.measurement_variance)
        self.estimate += gain * (measurement - self.estimate)
        self.error_estimate = (1 - gain) * error_prediction
        return self.estimate

    def filter_series(
            self,
            data: ArrayLike
            ) -> NDArray[np.float64]:
        return np.asarray([self.update(float(value)) for value in np.asarray(data)])
