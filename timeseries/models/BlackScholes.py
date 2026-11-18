# ruff: noqa: N999  # Preserve public import paths.
from math import erfc
from typing import Literal, overload

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pydantic import Field

from timeseries.model_register import ModelFactory
from timeseries.registry import Configuration


class ConfigBlackScholes(
    Configuration
):
    risk_free_rate: float = Field(
        0.0,
        description = "Annual continuously compounded risk-free rate."
    )
    volatility: float = Field(
        0.2,
        ge = 0,
        description = "Annualized volatility as a decimal."
    )
    dividend_yield: float = Field(
        0.0,
        ge = 0,
        description = "Annual continuous dividend yield."
    )


@ModelFactory.register(
    name = 'BlackScholes',
    description = 'European option prices with a continuous dividend yield.',
    id = 'blackscholes',
    domain = 'option_pricing',
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
        volatility: float = 0.2,
        dividend_yield: float = 0.0,
    ) -> None:
        config = self.CONFIG_T(
            risk_free_rate = risk_free_rate,
            volatility = volatility,
            dividend_yield = dividend_yield
        )
        self.risk_free_rate = config.risk_free_rate
        self.volatility = config.volatility
        self.dividend_yield = config.dividend_yield

    @overload
    def price(
        self, spot: float, strike: float, time_to_maturity: float,
        option_type: Literal['call', 'put'] = 'call',
    ) -> float:
        ...

    @overload
    def price(
        self, spot: ArrayLike, strike: ArrayLike, time_to_maturity: ArrayLike,
        option_type: Literal['call', 'put'] = 'call',
    ) -> float | NDArray[np.float64]:
        ...

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
            raise ValueError(
                "option_type must be 'call' or 'put'."
            )
        spot, strike, maturity = np.broadcast_arrays(
            np.asarray(
                spot,
                dtype = float
            ),
            np.asarray(
                strike,
                dtype = float
            ),
            np.asarray(
                time_to_maturity,
                dtype = float
            ),
        )
        if not all(
                np.isfinite(
                    value
                ).all() for value in (spot, strike, maturity)
        ):
            raise ValueError(
                "Pricing inputs must be finite."
            )
        if np.any(
                spot <= 0
        ) or np.any(
            strike <= 0
        ) or np.any(
            maturity < 0
        ):
            raise ValueError(
                "Spot and strike must be positive; time to maturity must be nonnegative."
            )

        discounted_strike = strike * np.exp(
            -self.risk_free_rate * maturity
        )
        discounted_spot = spot * np.exp(
            -self.dividend_yield * maturity
        )
        sign = 1 if option_type == 'call' else -1
        prices = np.maximum(
            sign * (discounted_spot - discounted_strike),
            0.0
        )
        active = maturity > 0
        if self.volatility > 0 and np.any(
                active
        ):
            total_volatility = self.volatility * np.sqrt(
                maturity[active]
            )
            d1 = (
                    (np.log(
                        spot[active]
                    ) - np.log(
                        strike[active]
                    )
                     + (self.risk_free_rate - self.dividend_yield) * maturity[active]) / total_volatility
                    + 0.5 * total_volatility
            )
            d2 = d1 - total_volatility
            # erfc avoids subtracting nearly equal numbers in the normal tails.
            normal_cdf = np.vectorize(
                lambda
                    x: 0.5 * erfc(
                    -x / np.sqrt(
                        2.0
                    )
                ),
                otypes = [float]
            )
            prices = np.array(
                prices,
                copy = True
            )
            prices[active] = np.maximum(
                sign * (discounted_spot[active] * normal_cdf(
                    sign * d1
                )
                        - discounted_strike[active] * normal_cdf(
                            sign * d2
                        )),
                0.0,
            )
        return float(
            prices
        ) if prices.ndim == 0 else prices

    def greeks(
        self,
        spot: float,
        strike: float,
        time_to_maturity: float,
        option_type: Literal['call', 'put'] = 'call'
    ) -> "Greeks":
        """Per underlying unit: theta/year, vega and rho per 1.0 change, not 1%."""
        from scipy.special import ndtr

        self.price(
            spot,
            strike,
            time_to_maturity,
            option_type
        )  # Shared input validation.
        if time_to_maturity <= 0 or self.volatility <= 0:
            raise ValueError(
                "Greeks require positive maturity and volatility."
            )
        s, k, t, v = spot, strike, time_to_maturity, self.volatility
        r, q = self.risk_free_rate, self.dividend_yield
        root = np.sqrt(
            t
        )
        d1 = (np.log(
            s / k
        ) + (r - q + v * v / 2) * t) / (v * root)
        d2 = d1 - v * root
        sign = 1 if option_type == 'call' else -1
        pdf = np.exp(
            -d1 * d1 / 2
        ) / np.sqrt(
            2 * np.pi
        )
        dq, dr = np.exp(
            -q * t
        ), np.exp(
            -r * t
        )
        return Greeks(
            delta = sign * dq * ndtr(
                sign * d1
            ),
            gamma = dq * pdf / (s * v * root),
            vega = s * dq * pdf * root,
            theta = -s * dq * pdf * v / (2 * root)
                    - sign * r * k * dr * ndtr(
                sign * d2
            ) + sign * q * s * dq * ndtr(
                sign * d1
            ),
            rho = sign * k * t * dr * ndtr(
                sign * d2
            ),
        )

    def implied_volatility(
        self,
        premium: float,
        spot: float,
        strike: float,
        time_to_maturity: float,
        option_type: Literal['call', 'put'] = 'call'
    ) -> float:
        """Invert European prices. Reject prices outside finite-volatility bounds."""
        from scipy.optimize import brentq

        self.price(
            spot,
            strike,
            time_to_maturity,
            option_type
        )
        if time_to_maturity <= 0 or not np.isfinite(
                premium
        ):
            raise ValueError(
                "Implied volatility requires finite premium and positive maturity."
            )
        lower = BlackScholes(
            self.risk_free_rate,
            0,
            self.dividend_yield
        ).price(
            spot,
            strike,
            time_to_maturity,
            option_type
        )
        upper = (spot * np.exp(
            -self.dividend_yield * time_to_maturity
        ) if option_type == 'call'
                 else strike * np.exp(
            -self.risk_free_rate * time_to_maturity
        ))
        if premium < lower or premium >= upper:
            raise ValueError(
                "Premium violates European no-arbitrage bounds."
            )
        if premium == lower:
            return 0.0

        def residual(
            vol
        ):
            return BlackScholes(
                self.risk_free_rate,
                vol,
                self.dividend_yield
            ).price(
                spot,
                strike,
                time_to_maturity,
                option_type
            ) - premium

        high = 1.0
        while residual(
                high
        ) < 0 and high < 1024:
            high *= 2
        return float(
            brentq(
                residual,
                0,
                high,
                xtol = 1e-12
            )
        )


class Greeks(
    Configuration
):
    delta: float = Field(
        0.0,
        description = 'Price sensitivity to a one-unit increase in the underlying price.'
    )
    gamma: float = Field(
        0.0,
        description = 'Delta sensitivity to a one-unit increase in the underlying price.'
    )
    vega: float = Field(
        0.0,
        description = 'Price sensitivity to a 1.0 change in annualized volatility, not 1%.'
    )
    theta: float = Field(
        0.0,
        description = 'Price sensitivity to the passage of time, expressed per year.'
    )
    rho: float = Field(
        0.0,
        description = 'Price sensitivity to a 1.0 change in the risk-free rate, not 1%.'
    )
