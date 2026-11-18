"""Validated cash instruments, option contracts and timestamped two-sided quotes."""
from datetime import UTC
from typing import Literal

from pydantic import AwareDatetime, ConfigDict, Field, model_validator

from timeseries.registry import Configuration


class Instrument(
    Configuration
):
    model_config = ConfigDict(
        frozen = True
    )
    symbol: str = Field(
        min_length = 1,
        description = 'Unique trading symbol of the instrument.'
    )
    currency: str = Field(
        'USD',
        min_length = 3,
        max_length = 3,
        description = 'Three-letter currency code for prices and cash flows.'
    )
    multiplier: float = Field(
        1,
        gt = 0,
        description = 'Factor converting quantity times quoted price into cash value.'
    )


class OptionContract(
    Instrument
):
    underlying: str = Field(
        min_length = 1,
        description = 'Trading symbol of the underlying instrument.'
    )
    strike: float = Field(
        gt = 0,
        description = 'Exercise price per underlying unit.'
    )
    expiry: AwareDatetime = Field(
        description = 'Timezone-aware contract expiration instant.'
    )
    option_type: Literal['call', 'put'] = Field(
        description = 'Option payoff type: call or put.'
    )
    exercise: Literal['european', 'american'] = Field(
        'european',
        description = 'Exercise style: at expiry only or before expiry as well.'
    )
    settlement: Literal['cash', 'physical'] = Field(
        'cash',
        description = 'Settlement through a cash payment or delivery of the underlying.'
    )
    multiplier: float = Field(
        100,
        gt = 0,
        description = 'Number of underlying units represented by one option contract.'
    )

    def maturity(
        self,
        as_of
    ) -> float:
        """ACT/365 using timezone-aware instants."""
        return max(
            (self.expiry.astimezone(
                UTC
            ) - as_of.astimezone(
                UTC
            )).total_seconds() / (365 * 86400),
            0
        )


class Quote(
    Configuration
):
    model_config = ConfigDict(
        frozen = True
    )
    symbol: str = Field(
        min_length = 1,
        description = 'Trading symbol associated with the quote.'
    )
    timestamp: AwareDatetime = Field(
        description = 'Timezone-aware quote event time.'
    )
    bid: float = Field(
        ge = 0,
        description = 'Best bid price per quoted unit.'
    )
    ask: float = Field(
        gt = 0,
        description = 'Best ask price per quoted unit.'
    )
    bid_size: float | None = Field(
        None,
        ge = 0,
        description = 'Available bid quantity in units or contracts; None means no size cap.'
    )
    ask_size: float | None = Field(
        None,
        ge = 0,
        description = 'Available ask quantity in units or contracts; None means no size cap.'
    )

    @model_validator(
        mode = 'after'
    )
    def check_spread(
        self
    ):
        if self.bid > self.ask:
            raise ValueError(
                'bid must not exceed ask.'
            )
        return self

    @property
    def mid(
        self
    ) -> float:
        return (self.bid + self.ask) / 2
