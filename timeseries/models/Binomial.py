"""Cox–Ross–Rubinstein tree with continuous dividends and early exercise."""
from typing import Literal

import numpy as np
from pydantic import Field

from timeseries.model_register import ModelFactory
from timeseries.models.BlackScholes import ConfigBlackScholes


class ConfigBinomial(ConfigBlackScholes):
    steps: int = Field(
        300, ge = 1, le = 5000, strict = True, description = 'Number of time steps in the Cox-Ross-Rubinstein tree.'
    )
    american: bool = Field(True, description = 'Allow early exercise; False selects European exercise at expiry.')


@ModelFactory.register(
    'Binomial', 'CRR European and American option pricing.', id = 'binomial', domain = 'option_pricing'
    )
class Binomial:
    CONFIG_T = ConfigBinomial

    def __init__(
        self,
        risk_free_rate: float = 0.0,
        volatility: float = 0.2,
        dividend_yield: float = 0.0,
        steps: int = 300,
        american: bool = True
    ):
        self.config = ConfigBinomial(
            risk_free_rate = risk_free_rate, volatility = volatility,
            dividend_yield = dividend_yield, steps = steps, american = american
            )

    def price(
        self,
        spot: float,
        strike: float,
        time_to_maturity: float,
        option_type: Literal['call', 'put'] = 'call'
    ) -> float:
        from timeseries.models.BlackScholes import BlackScholes

        c = self.config
        baseline = BlackScholes(c.risk_free_rate, c.volatility, c.dividend_yield)
        baseline.price(spot, strike, time_to_maturity, option_type)
        sign = 1 if option_type == 'call' else -1
        if time_to_maturity == 0:
            return float(max(sign * (spot - strike), 0))
        dt = time_to_maturity / c.steps
        discount = np.exp(-c.risk_free_rate * dt)
        if c.volatility == 0:
            times = np.arange(c.steps + 1) * dt if c.american else np.array([time_to_maturity])
            path = spot * np.exp((c.risk_free_rate - c.dividend_yield) * times)
            return float(np.max(np.exp(-c.risk_free_rate * times) * np.maximum(sign * (path - strike), 0)))
        up = np.exp(c.volatility * np.sqrt(dt))
        probability = (np.exp((c.risk_free_rate - c.dividend_yield) * dt) - 1 / up) / (up - 1 / up)
        if not 0 <= probability <= 1:
            raise ValueError("Invalid CRR probability; increase steps or check rate/volatility.")
        stock = spot * up ** (2 * np.arange(c.steps + 1) - c.steps)
        values = np.maximum(sign * (stock - strike), 0)
        for level in range(c.steps - 1, -1, -1):
            values = discount * ((1 - probability) * values[:-1] + probability * values[1:])
            if c.american:
                stock = spot * up ** (2 * np.arange(level + 1) - level)
                values = np.maximum(values, sign * (stock - strike))
        return float(values[0])
