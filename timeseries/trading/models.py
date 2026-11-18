"""Small, serializable trading records; quantities are signed underlying units/contracts."""
from math import copysign, isfinite
from typing import Literal
from uuid import uuid4

from pydantic import AwareDatetime, ConfigDict, Field, model_validator

from timeseries.instruments import Instrument, OptionContract
from timeseries.registry import Configuration


class Order(Configuration):
    model_config = ConfigDict(frozen = True)
    id: str = Field(
        default_factory = lambda: uuid4().hex, min_length = 1,
        description = 'Unique order identifier, generated when omitted.'
    )
    symbol: str = Field(min_length = 1, description = 'Trading symbol of the instrument to trade.')
    quantity: float = Field(description = 'Signed units or contracts: positive to buy, negative to sell.')
    timestamp: AwareDatetime = Field(description = 'Timezone-aware order submission time.')
    limit_price: float | None = Field(
        None, gt = 0, description = 'Maximum buy price or minimum sell price; None selects a market order.'
    )

    @model_validator(mode = 'after')
    def nonzero(
        self
    ):
        if self.quantity == 0:
            raise ValueError('Order quantity must be nonzero.')
        return self


class Fill(Configuration):
    model_config = ConfigDict(frozen = True)
    order_id: str = Field(description = 'Identifier of the order associated with this execution.')
    symbol: str = Field(description = 'Trading symbol of the executed instrument.')
    quantity: float = Field(description = 'Executed units or contracts: positive for a buy, negative for a sell.')
    price: float = Field(ge = 0, description = 'Execution price before applying the instrument multiplier.')
    commission: float = Field(
        0, ge = 0, description = 'Total commission charged for this fill in the instrument currency.'
    )
    timestamp: AwareDatetime = Field(description = 'Timezone-aware execution time.')


class OrderState(Configuration):
    order: Order = Field(description = 'Original order being tracked.')
    filled_quantity: float = Field(0, description = 'Cumulative executed quantity with the same sign as the order.')
    status: Literal['open', 'partial', 'filled', 'cancelled', 'rejected'] = Field(
        'open', description = 'Current execution state of the order.'
    )
    reason: str | None = Field(None, description = 'Explanation of the order state, such as a rejection reason.')

    @property
    def remaining(
        self
    ) -> float:
        return self.order.quantity - self.filled_quantity


class Position(Configuration):
    instrument: OptionContract | Instrument = Field(description = 'Instrument or option contract held in the position.')
    quantity: float = Field(0, description = 'Signed units or contracts held: positive for long, negative for short.')
    average_price: float = Field(
        0, ge = 0, description = 'Average entry price before applying the instrument multiplier.'
    )
    realized_pnl: float = Field(0, description = 'Cumulative realized profit or loss before commissions and financing.')


class Portfolio(Configuration):
    currency: str = Field('USD', description = 'Account currency shared by all instruments and cash balances.')
    cash: float = Field(10_000, description = 'Cash balance; negative values represent borrowing.')
    positions: dict[str, Position] = Field(
        default_factory = dict, description = 'Positions indexed by instrument trading symbol.'
    )
    fills: list[Fill] = Field(default_factory = list, description = 'Executed trades recorded in the portfolio ledger.')
    financing_cost: float = Field(0, ge = 0, description = 'Cumulative cash borrowing interest and short borrow fees.')

    @model_validator(mode = 'after')
    def check_symbols(
        self
    ):
        if any(
                symbol != p.instrument.symbol or p.instrument.currency != self.currency
                for symbol, p in self.positions.items()
                ):
            raise ValueError('Portfolio position keys/currencies must match instruments.')
        return self

    def apply(
        self,
        fill: Fill,
        instrument: Instrument
    ) -> None:
        if fill.symbol != instrument.symbol or instrument.currency != self.currency:
            raise ValueError('Fill/instrument mismatch or unsupported currency conversion.')
        if fill.quantity == 0:
            raise ValueError('Fill quantity must be nonzero.')
        p = self.positions.get(fill.symbol, Position(instrument = instrument))
        if p.instrument != instrument:
            raise ValueError('Instrument definition changed for an existing position.')
        old, change = p.quantity, fill.quantity
        new = old + change
        if old * change >= 0:
            average = (abs(old) * p.average_price + abs(change) * fill.price) / abs(new)
        else:
            p.realized_pnl += min(abs(old), abs(change)) * (fill.price - p.average_price) * copysign(
                1, old
                ) * instrument.multiplier
            average = (fill.price if old * new < 0 else p.average_price) if new else 0
        p.quantity, p.average_price = new, average
        self.positions[fill.symbol] = p
        self.cash -= change * fill.price * instrument.multiplier + fill.commission
        self.fills.append(fill)

    def market_values(
        self,
        marks: dict[str, float]
    ) -> dict[str, float]:
        values = {}
        for symbol, p in self.positions.items():
            if p.quantity:
                price = marks[symbol]  # Missing marks must not silently use cost basis.
                if not isfinite(price) or price < 0:
                    raise ValueError('Marks must be finite and nonnegative.')
                values[symbol] = p.quantity * price * p.instrument.multiplier
        return values

    def equity(
        self,
        marks: dict[str, float]
    ) -> float:
        return self.cash + sum(self.market_values(marks).values())

    @property
    def realized_pnl(
        self
    ) -> float:
        return sum(p.realized_pnl for p in self.positions.values()) - sum(
            f.commission for f in self.fills
            ) - self.financing_cost

    def unrealized_pnl(
        self,
        marks: dict[str, float]
    ) -> float:
        return sum(self.market_values(marks).values()) - sum(
            p.quantity * p.average_price * p.instrument.multiplier for p in self.positions.values()
        )
