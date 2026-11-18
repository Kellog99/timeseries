# ruff: noqa: N999  # Preserve public import paths.
import numpy as np
from numpy.typing import ArrayLike, NDArray
from pydantic import Field, model_validator

from timeseries.model_register import ModelFactory
from timeseries.registry import Configuration


class ConfigSimpleGARCH(
    Configuration
):
    omega: float = Field(
        1e-6,
        gt = 0,
        description = "Variance contribution in squared per-observation return units."
    )
    alpha: float = Field(
        0.05,
        ge = 0,
        lt = 1,
        description = "Weight of the previous squared innovation."
    )
    beta: float = Field(
        0.9,
        ge = 0,
        lt = 1,
        description = "Weight of the previous conditional variance."
    )

    @model_validator(
        mode = "after"
    )
    def check_stationarity(
        self
    ):
        if self.alpha + self.beta >= 1:
            raise ValueError(
                "Stationary GARCH requires alpha + beta < 1."
            )
        return self


@ModelFactory.register(
    name = 'SimpleGARCH',
    description = 'Stationary GARCH(1,1) volatility forecast.',
    id = 'garch',
    domain = 'volatility',
)
class SimpleGARCH:
    """Simplified stationary GARCH(1,1) volatility predictor."""
    CONFIG_T = ConfigSimpleGARCH

    def __init__(
        self,
        omega: float = 1e-6,
        alpha: float = 0.05,
        beta: float = 0.9
    ) -> None:
        config = self.CONFIG_T(
            omega = omega,
            alpha = alpha,
            beta = beta
        )
        self.omega, self.alpha, self.beta = config.omega, config.alpha, config.beta
        self.mean_ = 0.0

    def fit(
        self,
        returns: ArrayLike
    ) -> "SimpleGARCH":
        """Estimate stationary Gaussian GARCH parameters by maximum likelihood."""
        from scipy.optimize import minimize

        values = np.asarray(
            returns,
            dtype = float
        )
        if values.ndim != 1 or values.size < 20 or not np.isfinite(
                values
        ).all() or values.std() <= 0:
            raise ValueError(
                'Fit requires at least 20 finite, nonconstant returns.'
            )
        mean, scale = float(
            values.mean()
        ), float(
            values.std()
        )
        residuals = (values - mean) / scale

        def objective(
            params
        ):
            omega, alpha, beta = params
            variance, loss = 1., 0.
            for residual in residuals:
                loss += np.log(
                    variance
                ) + residual ** 2 / variance
                variance = omega + alpha * residual ** 2 + beta * variance
            return loss / len(
                residuals
            )

        fit = minimize(
            objective,
            [0.05, 0.05, 0.9],
            method = 'SLSQP',
            bounds = [(1e-8, 10), (0, .998), (0, .998)],
            constraints = [
                {
                    'type': 'ineq', 'fun': lambda
                    x: .999 - x[1] - x[2]
                }
            ],
            options = {'maxiter': 300, 'ftol': 1e-10}
        )
        if not fit.success:
            raise ValueError(
                f'GARCH calibration failed: {fit.message}'
            )
        config = self.CONFIG_T(
            omega = float(
                fit.x[0] * scale ** 2
            ),
            alpha = float(
                fit.x[1]
            ),
            beta = float(
                fit.x[2]
            )
        )
        self.omega, self.alpha, self.beta = config.omega, config.alpha, config.beta
        self.mean_ = mean
        return self

    def fit_predict(
        self,
        returns: ArrayLike,
        steps: int = 1
    ) -> NDArray[np.float64]:
        values = np.asarray(
            returns,
            dtype = float
        )
        if values.ndim != 1 or values.size == 0 or not np.isfinite(
                values
        ).all():
            raise ValueError(
                "Expected a nonempty finite return sequence."
            )
        if isinstance(
                steps,
                bool
        ) or not isinstance(
            steps,
            (int, np.integer)
        ) or steps <= 0:
            raise ValueError(
                "steps must be positive."
            )
        variance = float(
            values.var()
        )
        for value in values - self.mean_:
            variance = self.omega + self.alpha * value ** 2 + self.beta * variance
        forecasts: list[float] = []
        for _ in range(
                steps
        ):
            forecasts.append(
                float(
                    np.sqrt(
                        variance
                    )
                )
            )
            variance = self.omega + (self.alpha + self.beta) * variance
        return np.asarray(
            forecasts
        )
