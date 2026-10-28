"""Constructor configurations for registered metrics."""
from typing import Annotated, Literal

from pydantic import Field

from timeseries.registry import Configuration

Period = Annotated[int, Field(gt = 0)] | Literal["max"] | None


class WindowConfig(Configuration):
    period: Period = Field(
        None,
        description = "Non-overlapping window length; None or max uses all observations, including the final partial "
                      "window."
        )


class MovingAverageConfig(WindowConfig):
    """Arithmetic mean of each window."""


class MovingVarianceConfig(WindowConfig):
    """Population variance of each window."""


class MovingSRConfig(WindowConfig):
    """Unannualized Sharpe ratio of each window of decimal returns."""


class SharpeRatioConfig(Configuration):
    risk_free_rate: float = Field(
        0.0, description = "Risk-free decimal return per observation, in the same frequency as input returns."
        )
    periods_per_year: int = Field(
        252, gt = 0, description = "Observations per year used for square-root annualization."
        )


class OmegaRatioConfig(Configuration):
    threshold: float = Field(0.0, description = "Minimum acceptable decimal return per observation.")


class SortinoRatioConfig(Configuration):
    target_return: float = Field(0.0, description = "Minimum acceptable decimal return per observation.")
    periods_per_year: int = Field(
        252, gt = 0, description = "Observations per year used for square-root annualization."
        )


class VolatilityConfig(Configuration):
    periods_per_year: int = Field(
        252, gt = 0, description = "Observations per year used for square-root annualization."
        )


class MaxDrawdownConfig(Configuration):
    """Maximum peak-to-trough loss of compounded decimal returns."""


class CalmarRatioConfig(Configuration):
    periods_per_year: int = Field(
        252, gt = 0, description = "Observations per year used to annualize compounded returns."
        )


class TotalReturnConfig(Configuration):
    """Compounded return over all observations."""


class CanonicalCorrelationConfig(Configuration):
    """First canonical correlation between paired feature matrices."""


class RVCoefficientConfig(Configuration):
    """Multivariate RV coefficient between paired feature matrices."""


class DistanceCorrelationConfig(Configuration):
    """Distance correlation between paired feature matrices."""


class PearsonCorrelationConfig(Configuration):
    """Pearson correlation for paired univariate observations."""
