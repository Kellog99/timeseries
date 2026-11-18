"""Single cash-instrument backtests: close signals execute on the next bar's open."""
from collections.abc import Callable

import numpy as np
import pandas as pd
from pydantic import ConfigDict, Field, model_validator

from timeseries.instruments import Instrument, OptionContract, Quote
from timeseries.registry import Configuration
from .broker import ExecutionConfig, PaperBroker
from .models import Order, OrderState, Portfolio
from .risk import RiskLimits


def validate_prices(
    prices: pd.Series
) -> None:
    if not len(prices) or not np.isfinite(prices.to_numpy(dtype = float)).all() or (prices <= 0).any():
        raise ValueError('Prices must be nonempty, positive and finite.')


class MovingAverageStrategy(Configuration):
    fast: int = Field(
        10, ge = 1, strict = True, description = 'Number of observations in the fast moving-average window.'
    )
    slow: int = Field(
        30, ge = 2, strict = True, description = 'Number of observations in the slow window; must exceed fast.'
    )
    short: bool = Field(False, description = 'Generate short exposure when the fast average is below the slow average.')

    @model_validator(mode = 'after')
    def check_windows(
        self
    ):
        if self.fast >= self.slow:
            raise ValueError('fast must be smaller than slow.')
        return self

    def signals(
        self,
        close: pd.Series
    ) -> pd.Series:
        """Target exposure in [-1, 1], observable at each close; zero during warm-up."""
        validate_prices(close)
        fast, slow = close.rolling(self.fast).mean(), close.rolling(self.slow).mean()
        result = pd.Series(np.where(fast > slow, 1., -1. if self.short else 0.), index = close.index)
        return result.where(slow.notna(), 0.)


class BacktestConfig(Configuration):
    initial_cash: float = Field(10_000, gt = 0, description = 'Starting cash balance in the instrument currency.')
    allocation: float = Field(
        0.95, gt = 0, le = 1, description = 'Fraction of equity allocated at a full-strength target signal.'
    )
    spread_bps: float = Field(
        5, ge = 0, lt = 10_000, description = 'Full bid-ask spread around the opening price in basis points.'
    )
    participation: float = Field(
        0.01, gt = 0, le = 1, description = 'Maximum fraction of the previous bar volume available per quote side.'
    )
    annual_financing_rate: float = Field(
        0, ge = 0, description = 'Annual borrowing rate on negative cash balances as a decimal.'
    )
    annual_short_borrow_rate: float = Field(
        0, ge = 0, description = 'Annual stock borrow fee on short market value as a decimal.'
    )
    periods_per_year: int = Field(
        252, ge = 1, strict = True, description = 'Number of observation periods per year for annualized metrics.'
    )
    execution: ExecutionConfig = Field(
        default_factory = ExecutionConfig, description = 'Commission, slippage and quote freshness settings.'
    )
    limits: RiskLimits = Field(
        default_factory = RiskLimits, description = 'Portfolio risk limits applied before execution.'
    )


class BacktestResult(Configuration):
    model_config = ConfigDict(arbitrary_types_allowed = True)
    history: pd.DataFrame = Field(description = 'Per-bar equity, cash, quantity, commission and financing history.')
    orders: list[OrderState] = Field(description = 'Submitted orders and their final execution states.')
    initial_cash: float = Field(description = 'Starting cash balance used as the baseline for returns.')
    periods_per_year: int = Field(description = 'Number of observation periods per year for annualized metrics.')

    @property
    def returns(
        self
    ) -> pd.Series:
        equity = self.history.equity
        previous = equity.shift(1, fill_value = self.initial_cash)
        return equity / previous - 1

    def summary(
        self
    ) -> dict[str, float]:
        from timeseries.metrics import MetricFactory

        result = {}
        for name in ('total_return', 'sharpe', 'sortino', 'max_drawdown', 'volatility'):
            params = {'periods_per_year': self.periods_per_year} if name in ('sharpe', 'sortino', 'volatility') else {}
            metric = MetricFactory.create(name, **params)
            metric.update(self.returns.to_list())
            result[name] = float(metric.compute())
        result['commissions'] = float(self.history.commission.sum())
        result['financing'] = float(self.history.financing.sum())
        return result


def backtest(
    bars: pd.DataFrame,
    signals: pd.Series | None = None,
    *,
    strategy: MovingAverageStrategy | None = None,
    config: BacktestConfig | None = None,
    instrument: Instrument | None = None
) -> BacktestResult:
    """Bars indexed by timezone-aware opening instants, with open/close and optional volume.

    Volume capacity uses the PREVIOUS bar's volume, known before execution. Orders
    expire after one opening quote. End holdings remain marked to the last close.
    Input prices must be tradable unadjusted prices; corporate actions need preprocessing.
    """
    c = config or BacktestConfig()
    instrument = instrument or Instrument(symbol = 'ASSET')
    if isinstance(instrument, OptionContract):
        raise ValueError('Use PaperBroker for options; this bar backtester handles cash underlyings.')
    if (not isinstance(bars.index, pd.DatetimeIndex) or bars.index.tz is None
            or not bars.index.is_monotonic_increasing or bars.index.has_duplicates or len(bars) < 2):
        raise ValueError('Provide at least two unique, increasing, timezone-aware bars.')
    for column in ('open', 'close'):
        validate_prices(bars[column])
    if 'volume' in bars and (not np.isfinite(bars.volume).all() or (bars.volume < 0).any()):
        raise ValueError('Volume must be finite and nonnegative.')
    targets = signals if signals is not None else (strategy or MovingAverageStrategy()).signals(bars.close)
    if (not targets.index.equals(bars.index) or not np.isfinite(targets).all() or (targets.abs() > 1).any()):
        raise ValueError('Signals must align with bars and contain finite target weights in [-1, 1].')
    if (targets < 0).any() and not c.limits.allow_short:
        raise ValueError('Short signals require allow_short=True.')
    broker = PaperBroker(
        [instrument], Portfolio(cash = c.initial_cash, currency = instrument.currency),
        c.execution, c.limits
        )
    history = []
    for i, timestamp in enumerate(bars.index):
        row = bars.iloc[i]
        financing = 0.
        if i:
            days = (timestamp - bars.index[i - 1]).total_seconds() / 86400
            financing = broker.accrue_financing(days, c.annual_financing_rate, c.annual_short_borrow_rate)
        equity = broker.portfolio.equity({instrument.symbol: row.open})
        weight = float(targets.iloc[i - 1]) if i else 0.
        # Include spread, slippage and a conservative fee allowance when sizing buys.
        ask = row.open * (1 + c.spread_bps / 20_000)
        bid = row.open * (1 - c.spread_bps / 20_000)
        unit_cost = ask * (1 + c.execution.slippage_bps / 10_000) * instrument.multiplier
        budget = max(abs(weight) * c.allocation * equity - c.execution.minimum_commission, 0)
        desired = np.sign(weight) * np.floor(budget / (unit_cost + c.execution.commission_per_unit))
        position = broker.portfolio.positions.get(instrument.symbol)
        quantity = desired - (position.quantity if position else 0)
        state = None
        if quantity:
            state = broker.submit(
                Order(symbol = instrument.symbol, quantity = quantity, timestamp = timestamp.to_pydatetime())
                )
        capacity = float(np.floor(bars.volume.iloc[i - 1] * c.participation)) if 'volume' in bars and i else None
        quote = Quote(
            symbol = instrument.symbol, timestamp = timestamp.to_pydatetime(), bid = bid, ask = ask,
            bid_size = capacity, ask_size = capacity
            )
        fills = broker.on_quote(quote, timestamp.to_pydatetime())
        if state:
            broker.cancel(state.order.id)
        position = broker.portfolio.positions.get(instrument.symbol)
        closing_equity = broker.portfolio.equity({instrument.symbol: row.close})
        broker.peak_equity = max(broker.peak_equity, closing_equity)
        history.append(
            dict(
                equity = closing_equity, cash = broker.portfolio.cash,
                quantity = position.quantity if position else 0,
                commission = sum(f.commission for f in fills), financing = financing
                )
            )
        if closing_equity <= 0:
            raise ValueError('Account insolvent; simulation stopped.')
    return BacktestResult(
        history = pd.DataFrame(history, index = bars.index), orders = list(broker.orders.values()),
        initial_cash = c.initial_cash, periods_per_year = c.periods_per_year
        )


class WalkForwardConfig(Configuration):
    train_size: int = Field(
        100, ge = 2, strict = True, description = 'Number of observations in each rolling training window.'
    )
    test_size: int = Field(
        20, ge = 1, strict = True, description = 'Number of out-of-sample observations predicted before refitting.'
    )
    gap: int = Field(
        0, ge = 0, strict = True,
        description = 'Number of observations excluded between training and prediction windows.'
    )


def walk_forward_signals(
    close: pd.Series,
    fit: Callable[[pd.Series], Callable[[pd.Series], float]],
    config: WalkForwardConfig | None = None
) -> pd.Series:
    """fit(train) returns predict(history)->weight; history ends at the current close.

    The gap excludes the last training rows (e.g. for overlapping forward labels).
    Each fold creates a fresh fitted predictor; tuning must happen inside fit(train).
    """
    c = config or WalkForwardConfig()
    validate_prices(close)
    if not close.index.is_monotonic_increasing or close.index.has_duplicates:
        raise ValueError('Walk-forward input must be ordered and unique.')
    if len(close) <= c.train_size + c.gap:
        raise ValueError('Insufficient observations for an out-of-sample fold.')
    signals = pd.Series(0., index = close.index)
    for start in range(c.train_size + c.gap, len(close), c.test_size):
        end = start - c.gap
        predict = fit(close.iloc[end - c.train_size:end].copy())
        for i in range(start, min(start + c.test_size, len(close))):
            value = float(predict(close.iloc[:i + 1].copy()))
            if not np.isfinite(value) or abs(value) > 1:
                raise ValueError('Predictor must return a finite weight in [-1, 1].')
            signals.iloc[i] = value
    return signals
