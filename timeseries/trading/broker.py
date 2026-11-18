"""In-memory paper execution. Broker adapters can implement the same small protocol."""
from collections.abc import Callable
from datetime import datetime
from math import copysign
from typing import Protocol

from pydantic import Field

from timeseries.instruments import Instrument, OptionContract, Quote
from timeseries.registry import Configuration
from .models import Fill, Order, OrderState, Portfolio
from .risk import RiskLimits


class ExecutionConfig(
    Configuration
):
    commission_per_unit: float = Field(
        0,
        ge = 0,
        description = 'Commission per executed unit or contract in the account currency.'
    )
    minimum_commission: float = Field(
        0,
        ge = 0,
        description = 'Minimum commission charged per fill in the account currency.'
    )
    slippage_bps: float = Field(
        0,
        ge = 0,
        lt = 10_000,
        description = 'Adverse execution price adjustment in basis points.'
    )
    max_quote_age_seconds: float = Field(
        60,
        ge = 0,
        description = 'Maximum accepted quote age in seconds at execution time.'
    )


class Broker(
    Protocol
):
    def submit(
        self,
        order: Order
    ) -> OrderState: ...

    def cancel(
        self,
        order_id: str
    ) -> OrderState: ...

    def on_quote(
        self,
        quote: Quote,
        now: datetime
    ) -> list[Fill]: ...


class PaperBroker:
    def __init__(
        self,
        instruments: list[Instrument],
        portfolio: Portfolio | None = None,
        execution: ExecutionConfig | None = None,
        limits: RiskLimits | None = None,
        pre_trade_check: Callable[[Portfolio], None] | None = None
    ):
        self.instruments = {i.symbol: i for i in instruments}
        if len(
                self.instruments
        ) != len(
            instruments
        ):
            raise ValueError(
                'Duplicate instrument symbols.'
            )
        self.portfolio = portfolio if portfolio is not None else Portfolio()
        self.execution = execution or ExecutionConfig()
        self.limits = limits or RiskLimits()
        self.pre_trade_check = pre_trade_check
        for i in instruments:
            if i.currency != self.portfolio.currency:
                raise ValueError(
                    'A paper account supports one currency.'
                )
            if isinstance(
                    i,
                    OptionContract
            ) and (i.exercise != 'european' or i.settlement != 'cash'):
                raise ValueError(
                    'Paper options require European exercise and cash settlement.'
                )
        for symbol, p in self.portfolio.positions.items():
            if symbol not in self.instruments or p.instrument != self.instruments[symbol]:
                raise ValueError(
                    'Portfolio and instrument catalog disagree.'
                )
        self.orders: dict[str, OrderState] = {}
        self.quotes: dict[str, Quote] = {}
        self.peak_equity = (-float(
            'inf'
        ) if any(
            p.quantity for p in self.portfolio.positions.values()
        )
                            else self.portfolio.cash)
        self.enabled = True
        self._last_event: datetime | None = None

    def submit(
        self,
        order: Order
    ) -> OrderState:
        if order.id in self.orders:
            if self.orders[order.id].order != order:
                raise ValueError(
                    'Order id already used with different parameters.'
                )
            return self.orders[order.id]
        if order.symbol not in self.instruments:
            raise ValueError(
                'Unknown instrument.'
            )
        state = OrderState(
            order = order
        )
        instrument = self.instruments[order.symbol]
        if not self.enabled or (isinstance(
                instrument,
                OptionContract
        ) and order.timestamp >= instrument.expiry):
            state.status, state.reason = 'rejected', 'Trading disabled or contract expired.'
        elif isinstance(
                instrument,
                OptionContract
        ) and order.quantity != int(
            order.quantity
        ):
            state.status, state.reason = 'rejected', 'Option quantities must be whole contracts.'
        self.orders[order.id] = state
        return state

    def cancel(
        self,
        order_id: str
    ) -> OrderState:
        state = self.orders[order_id]
        if state.status in ('open', 'partial'):
            state.status = 'cancelled'
        return state

    def halt(
        self
    ) -> None:
        self.enabled = False
        for order_id in self.orders:
            self.cancel(
                order_id
            )

    def on_quote(
        self,
        quote: Quote,
        now: datetime
    ) -> list[Fill]:
        if quote.symbol not in self.instruments:
            raise ValueError(
                'Unknown quote symbol.'
            )
        if (not now.tzinfo or not 0 <= (now - quote.timestamp).total_seconds() <= self.execution.max_quote_age_seconds
                or (self._last_event is not None and now < self._last_event)):
            raise ValueError(
                'Stale, future or out-of-order quote event.'
            )
        old = self.quotes.get(
            quote.symbol
        )
        if old and quote.timestamp <= old.timestamp:
            raise ValueError(
                'Quote timestamps must increase; replay would reuse liquidity.'
            )
        self._last_event = now
        self.quotes[quote.symbol] = quote
        if any(
                p.quantity and s not in self.quotes for s, p in self.portfolio.positions.items()
        ):
            return []  # Wait for marks on all restored positions before evaluating risk.
        marks = {s: q.mid for s, q in self.quotes.items()}
        valid_marks = all(
            (now - self.quotes[s].timestamp).total_seconds() <= self.execution.max_quote_age_seconds
            and not (isinstance(
                p.instrument,
                OptionContract
            ) and now >= p.instrument.expiry)
            for s, p in self.portfolio.positions.items() if p.quantity
        )
        if valid_marks:
            self.peak_equity = max(
                self.peak_equity,
                self.portfolio.equity(
                    marks
                )
            )
        available = {1: quote.ask_size, -1: quote.bid_size}
        fills = []
        instrument = self.instruments[quote.symbol]
        for state in self.orders.values():
            order = state.order
            if order.symbol != quote.symbol or state.status not in (
                    'open',
                    'partial'
            ) or order.timestamp > quote.timestamp:
                continue
            if not self.enabled or isinstance(
                    instrument,
                    OptionContract
            ) and now >= instrument.expiry:
                state.status, state.reason = 'rejected', 'Trading disabled or contract expired.'
                continue
            side = 1 if state.remaining > 0 else -1
            if side == -1 and quote.bid == 0:
                continue
            price = (quote.ask if side == 1 else quote.bid) * (1 + side * self.execution.slippage_bps / 10_000)
            if order.limit_price is not None and side * (price - order.limit_price) > 0:
                continue
            capacity = available[side]
            size = abs(
                state.remaining
            ) if capacity is None else min(
                abs(
                    state.remaining
                ),
                capacity
            )
            if not size:
                continue
            if isinstance(
                    instrument,
                    OptionContract
            ) and size != int(
                size
            ):
                state.status, state.reason = 'rejected', 'Option quantities must be whole contracts.'
                continue
            fee = max(
                size * self.execution.commission_per_unit,
                self.execution.minimum_commission
            )
            fill = Fill(
                order_id = order.id,
                symbol = order.symbol,
                quantity = copysign(
                    size,
                    side
                ),
                price = price,
                commission = fee,
                timestamp = now
            )
            candidate = self.portfolio.model_copy(
                deep = True
            )
            candidate.apply(
                fill,
                instrument
            )
            try:
                for symbol, p in candidate.positions.items():
                    if p.quantity:
                        q = self.quotes[symbol]
                        if (now - q.timestamp).total_seconds() > self.execution.max_quote_age_seconds:
                            raise ValueError(
                                'An existing position has a stale quote.'
                            )
                        if isinstance(
                                p.instrument,
                                OptionContract
                        ) and p.quantity < 0:
                            raise ValueError(
                                'Uncovered short options are unsupported.'
                            )
                        if isinstance(
                                p.instrument,
                                OptionContract
                        ) and now >= p.instrument.expiry:
                            raise ValueError(
                                'Expired option positions must be settled before execution.'
                            )
                self.limits.check(
                    candidate,
                    marks,
                    self.peak_equity,
                    previous = self.portfolio
                )
                if self.pre_trade_check is not None:
                    self.pre_trade_check(
                        candidate
                    )
            except ValueError as exc:
                state.status, state.reason = 'rejected', str(
                    exc
                )
                continue
            self.portfolio.apply(
                fill,
                instrument
            )
            state.filled_quantity += fill.quantity
            state.status = 'filled' if abs(
                state.remaining
            ) < 1e-10 else 'partial'
            if capacity is not None:
                available[side] = capacity - size
            fills.append(
                fill
            )
        return fills

    def settle_option(
        self,
        symbol: str,
        underlying: Quote,
        now: datetime
    ) -> Fill | None:
        """Explicit expiry settlement; underlying must be quoted at the expiry instant."""
        contract = self.instruments[symbol]
        if not isinstance(
                contract,
                OptionContract
        ) or now < contract.expiry:
            raise ValueError(
                'An expired option is required.'
            )
        if (underlying.symbol != contract.underlying or underlying.timestamp != contract.expiry
                or underlying.bid != underlying.ask):
            raise ValueError(
                'Provide the official settlement price as equal bid/ask at expiry.'
            )
        position = self.portfolio.positions.get(
            symbol
        )
        if position is None or not position.quantity:
            return None
        sign = 1 if contract.option_type == 'call' else -1
        payoff = max(
            sign * (underlying.mid - contract.strike),
            0
        )
        fill = Fill(
            order_id = f'expiry:{symbol}',
            symbol = symbol,
            quantity = -position.quantity,
            price = payoff,
            timestamp = now
        )
        self.portfolio.apply(
            fill,
            contract
        )
        for state in self.orders.values():
            if state.order.symbol == symbol:
                self.cancel(
                    state.order.id
                )
        return fill

    def accrue_financing(
        self,
        days: float,
        annual_rate: float = 0,
        short_borrow_rate: float = 0
    ) -> float:
        """Simple ACT/365 debit interest and stock borrow fees; no interest on cash credits."""
        import math

        if any(
                not math.isfinite(
                    v
                ) or v < 0 for v in (days, annual_rate, short_borrow_rate)
        ):
            raise ValueError(
                'Financing inputs must be finite and nonnegative.'
            )
        marks = {s: q.mid for s, q in self.quotes.items()}
        shorts = -sum(
            min(
                v,
                0
            ) for v in self.portfolio.market_values(
                marks
            ).values()
        )
        charge = (max(
            -self.portfolio.cash,
            0
        ) * annual_rate + shorts * short_borrow_rate) * days / 365
        self.portfolio.cash -= charge
        self.portfolio.financing_cost += charge
        return charge

    def reconcile(
        self,
        expected_positions: dict[str, float],
        expected_cash: float,
        tolerance: float = 1e-8
    ) -> None:
        """Raise on broker/account differences; never silently overwrite the ledger."""
        from math import isfinite

        if not all(
                isfinite(
                    v
                ) for v in [expected_cash, tolerance, *expected_positions.values()]
        ) or tolerance < 0:
            raise ValueError(
                'Reconciliation inputs must be finite, with nonnegative tolerance.'
            )
        actual = {s: p.quantity for s, p in self.portfolio.positions.items()}
        if abs(
                self.portfolio.cash - expected_cash
        ) > tolerance or any(
            abs(
                actual.get(
                    s,
                    0
                ) - expected_positions.get(
                    s,
                    0
                )
            ) > tolerance
            for s in actual.keys() | expected_positions.keys()
        ):
            raise ValueError(
                'Cash or position reconciliation failed.'
            )
