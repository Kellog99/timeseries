from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from timeseries.registry import Configuration

PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeInt = Annotated[int, Field(ge=0)]
Probability = Annotated[float, Field(ge=0, lt=1)]
Quantile = Annotated[float, Field(gt=0, lt=1)]
ForecastLoss = Literal[
    "l1", "mse", "linear_penalization", "exponential_penalization",
    "sinkhorn", "high_order", "smape",
]
BaseLoss = ForecastLoss | Literal["kl"]


class ForecastConfiguration(Configuration):
    @model_validator(mode="after")
    def validate_quantiles(self) -> Self:
        quantiles = getattr(self, "quantiles", None)
        if quantiles and quantiles != sorted(set(quantiles)):
            raise ValueError("Forecasting quantiles must be distinct increasing probabilities.")
        return self


class ThreeQuantileConfiguration(ForecastConfiguration):
    @model_validator(mode="after")
    def validate_quantile_count(self) -> Self:
        quantiles = getattr(self, "quantiles", None)
        if quantiles and len(quantiles) != 3:
            raise ValueError("This architecture requires three quantiles or an empty sequence.")
        return self
