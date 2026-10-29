from math import erfc
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pydantic import Field

from timeseries.model_register import ModelFactory
from timeseries.registry import Configuration


class ConfigBlackScholes(Configuration):
    risk_free_rate: float = Field(0.0, description = "Annual continuously compounded risk-free rate.")
    volatility: float = Field(0.2, ge = 0, description = "Annualized volatility as a decimal.")


@ModelFactory.register(
    name = 'BlackScholes',
    description = 'Analytical European option prices for a non-dividend-paying underlying.',
    id = 'blackscholes', domain = 'option_pricing',
)
class BlackScholes:
    """Solve the Black–Scholes equation for European call and put payoffs.

    Rates and volatility are annualized decimals; time to maturity is in years.
    Pricing inputs broadcast using NumPy rules. No fitting is required.
    """
    CONFIG_T = ConfigBlackScholes

    def __init__(
            self,
            risk_free_rate: float = 0.0,
            volatility: float = 0.2
    ) -> None:
        config = self.CONFIG_T(risk_free_rate = risk_free_rate, volatility = volatility)
        self.risk_free_rate = config.risk_free_rate
        self.volatility = config.volatility

    def price(
            self,
            spot: ArrayLike,
            strike: ArrayLike,
            time_to_maturity: ArrayLike,
            option_type: Literal['call', 'put'] = 'call'
    ) -> float | NDArray[np.float64]:
        """Return option prices, including expiry and deterministic limits.

        Spot and strike must be positive; maturity must be nonnegative.
        All inputs must be finite. A scalar input combination returns a float.
        """
        if option_type not in ('call', 'put'):
            raise ValueError("option_type must be 'call' or 'put'.")
        spot, strike, maturity = np.broadcast_arrays(
            np.asarray(spot, dtype = float),
            np.asarray(strike, dtype = float),
            np.asarray(time_to_maturity, dtype = float),
        )
        if not all(np.isfinite(value).all() for value in (spot, strike, maturity)):
            raise ValueError("Pricing inputs must be finite.")
        if np.any(spot <= 0) or np.any(strike <= 0) or np.any(maturity < 0):
            raise ValueError("Spot and strike must be positive; time to maturity must be nonnegative.")

        discounted_strike = strike * np.exp(-self.risk_free_rate * maturity)
        sign = 1 if option_type == 'call' else -1
        prices = np.maximum(sign * (spot - discounted_strike), 0.0)
        active = maturity > 0
        if self.volatility > 0 and np.any(active):
            total_volatility = self.volatility * np.sqrt(maturity[active])
            d1 = (
                    (np.log(spot[active]) - np.log(strike[active])
                     + self.risk_free_rate * maturity[active]) / total_volatility
                    + 0.5 * total_volatility
            )
            d2 = d1 - total_volatility
            # erfc avoids subtracting nearly equal numbers in the normal tails.
            normal_cdf = np.vectorize(
                lambda
                    x: 0.5 * erfc(-x / np.sqrt(2.0)), otypes = [float]
                )
            prices = np.array(prices, copy = True)
            prices[active] = np.maximum(
                sign * (spot[active] * normal_cdf(sign * d1)
                        - discounted_strike[active] * normal_cdf(sign * d2)),
                0.0,
            )
        return float(prices) if prices.ndim == 0 else prices
