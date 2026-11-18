from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest
from pydantic import ValidationError

from timeseries.instruments import Instrument, OptionContract, Quote
from timeseries.trading import (
    BacktestConfig,
    ExecutionConfig,
    Fill,
    MovingAverageStrategy,
    Order,
    PaperBroker,
    Portfolio,
    RiskLimits,
    TailRisk,
    WalkForwardConfig,
    backtest,
    stress_test,
    walk_forward_signals,
)

NOW = datetime(
    2026,
    1,
    1,
    tzinfo = UTC
)
STOCK = Instrument(
    symbol = 'TEST'
)


def quote(
    seconds = 0,
    **kwargs
):
    return Quote(
        symbol = 'TEST',
        timestamp = NOW + timedelta(
            seconds = seconds
        ),
        **({'bid': 99, 'ask': 101} | kwargs)
    )


def order(
    quantity,
    **kwargs
):
    return Order(
        symbol = 'TEST',
        quantity = quantity,
        timestamp = NOW,
        **kwargs
    )


def apply(
    portfolio,
    quantity,
    price,
    fee = 0
):
    portfolio.apply(
        Fill(
            order_id = 'test',
            symbol = 'TEST',
            quantity = quantity,
            price = price,
            commission = fee,
            timestamp = NOW
        ),
        STOCK
    )


def test_ledger_average_cost_closes_flips_and_roundtrip():
    p = Portfolio(
        cash = 10000
    )
    apply(
        p,
        10,
        100,
        1
    )
    apply(
        p,
        10,
        120,
        1
    )
    assert p.positions['TEST'].average_price == 110
    apply(
        p,
        -25,
        130,
        2
    )
    assert p.positions['TEST'].quantity == -5
    assert p.positions['TEST'].average_price == 130
    assert p.realized_pnl == 396
    assert p.unrealized_pnl(
        {'TEST': 120}
    ) == 50
    assert p.equity(
        {'TEST': 120}
    ) - 10000 == p.realized_pnl + p.unrealized_pnl(
        {'TEST': 120}
    )
    apply(
        p,
        5,
        120,
        1
    )
    assert p.realized_pnl == 445
    assert p.equity(
        {}
    ) == 10445
    assert Portfolio.model_validate_json(
        p.model_dump_json()
    ) == p
    with pytest.raises(
            KeyError
    ):
        apply(
            p,
            1,
            100
        )
        p.equity(
            {}
        )


def test_partial_fills_share_liquidity_are_idempotent_and_cancellable():
    b = PaperBroker(
        [STOCK],
        execution = ExecutionConfig(
            commission_per_unit = .1
        )
    )
    first = order(
        5,
        id = 'first'
    )
    state = b.submit(
        first
    )
    assert b.submit(
        first
    ) is state
    b.submit(
        order(
            5,
            id = 'second'
        )
    )
    fills = b.on_quote(
        quote(
            ask_size = 7
        ),
        NOW
    )
    assert [f.quantity for f in fills] == [5, 2]
    assert [s.status for s in b.orders.values()] == ['filled', 'partial']
    assert b.portfolio.cash == pytest.approx(
        10000 - 7 * 101 - .7
    )
    b.cancel(
        'second'
    )
    assert b.on_quote(
        quote(
            1
        ),
        NOW + timedelta(
            seconds = 1
        )
    ) == []
    b.reconcile(
        {'TEST': 7},
        b.portfolio.cash
    )
    with pytest.raises(
            ValueError,
            match = 'reconciliation'
    ):
        b.reconcile(
            {'TEST': 8},
            b.portfolio.cash
        )
    with pytest.raises(
            ValueError,
            match = 'different'
    ):
        b.submit(
            order(
                6,
                id = 'first'
            )
        )
    with pytest.raises(
            ValueError,
            match = 'increase'
    ):
        b.on_quote(
            quote(
                1
            ),
            NOW + timedelta(
                seconds = 1
            )
        )


def test_limits_use_execution_costs_reject_atomically_and_allow_exit():
    b = PaperBroker(
        [STOCK],
        Portfolio(
            cash = 1000
        ),
        limits = RiskLimits(
            max_position_fraction = .5
        )
    )
    state = b.submit(
        order(
            6
        )
    )
    assert b.on_quote(
        quote(),
        NOW
    ) == []
    assert state.status == 'rejected'
    assert b.portfolio.cash == 1000 and not b.portfolio.fills
    state = b.submit(
        order(
            4
        )
    )
    assert len(
        b.on_quote(
            quote(
                1
            ),
            NOW + timedelta(
                seconds = 1
            )
        )
    ) == 1
    b.peak_equity = 2000
    b.submit(
        order(
            -4
        )
    )  # Closing must remain possible after a drawdown.
    assert len(
        b.on_quote(
            quote(
                2
            ),
            NOW + timedelta(
                seconds = 2
            )
        )
    ) == 1
    assert b.portfolio.positions['TEST'].quantity == 0


def test_limit_prices_include_slippage_and_halt_cancels_pending():
    b = PaperBroker(
        [STOCK],
        execution = ExecutionConfig(
            slippage_bps = 100
        )
    )
    state = b.submit(
        order(
            1,
            limit_price = 101
        )
    )
    assert b.on_quote(
        quote(),
        NOW
    ) == []
    assert state.status == 'open'
    b.halt()
    assert state.status == 'cancelled'
    assert b.submit(
        order(
            1
        )
    ).status == 'rejected'


def test_short_cash_interest_and_borrow_fees():
    b = PaperBroker(
        [STOCK],
        limits = RiskLimits(
            allow_short = True
        )
    )
    b.submit(
        order(
            -10
        )
    )
    b.on_quote(
        quote(),
        NOW
    )
    assert b.portfolio.positions['TEST'].quantity == -10
    assert b.accrue_financing(
        365,
        short_borrow_rate = .03
    ) == pytest.approx(
        30
    )
    b.portfolio.cash = -100
    assert b.accrue_financing(
        365,
        annual_rate = .1
    ) == pytest.approx(
        10
    )
    assert b.portfolio.financing_cost == 40


@pytest.mark.parametrize(
    'event,now',
    [
        (
                quote(), NOW + timedelta(
                seconds = 61
            )
        ), (
            quote(
                1
            ), NOW
    )
    ]
)
def test_stale_and_future_quotes_cannot_fill(
    event,
    now
):
    b = PaperBroker(
        [STOCK]
    )
    b.submit(
        order(
            1
        )
    )
    with pytest.raises(
            ValueError
    ):
        b.on_quote(
            event,
            now
        )
    assert not b.portfolio.fills


def test_option_settlement_preserves_contract_after_serialization():
    expiry = NOW + timedelta(
        days = 1
    )
    contract = OptionContract(
        symbol = 'CALL',
        underlying = 'TEST',
        strike = 100,
        expiry = expiry,
        option_type = 'call'
    )
    b = PaperBroker(
        [contract]
    )
    b.submit(
        Order(
            symbol = 'CALL',
            quantity = 1,
            timestamp = NOW
        )
    )
    b.on_quote(
        Quote(
            symbol = 'CALL',
            timestamp = NOW,
            bid = 4,
            ask = 5
        ),
        NOW
    )
    restored = Portfolio.model_validate_json(
        b.portfolio.model_dump_json()
    )
    assert isinstance(
        restored.positions['CALL'].instrument,
        OptionContract
    )
    fill = b.settle_option(
        'CALL',
        Quote(
            symbol = 'TEST',
            timestamp = expiry,
            bid = 110,
            ask = 110
        ),
        expiry
    )
    assert fill is not None
    assert fill.price == 10
    assert b.portfolio.cash == 10500
    assert b.settle_option(
        'CALL',
        Quote(
            symbol = 'TEST',
            timestamp = expiry,
            bid = 110,
            ask = 110
        ),
        expiry
    ) is None
    with pytest.raises(
            ValueError
    ):
        PaperBroker(
            [
                contract.model_copy(
                    update = {'exercise': 'american'}
                )
            ]
        )


def bars(
    prices
):
    return pd.DataFrame(
        {'open': prices, 'close': prices},
        index = pd.date_range(
            '2026-01-01',
            periods = len(
                prices
            ),
            tz = 'UTC'
        )
    )


def test_next_open_execution_and_accounting_have_hand_computed_pnl():
    data = bars(
        [100, 110, 120, 130]
    )
    targets = pd.Series(
        [1., 0., 0., 0.],
        index = data.index
    )
    result = backtest(
        data,
        targets,
        config = BacktestConfig(
            initial_cash = 1000,
            allocation = 1,
            spread_bps = 0
        )
    )
    assert result.history.quantity.to_list() == [0, 9, 0, 0]
    assert result.history.equity.to_list() == [1000, 1000, 1090, 1090]
    assert result.summary()['total_return'] == pytest.approx(
        .09
    )
    expensive = backtest(
        data,
        targets,
        config = BacktestConfig(
            initial_cash = 1000,
            allocation = 1,
            spread_bps = 10,
            execution = ExecutionConfig(
                commission_per_unit = .2,
                slippage_bps = 10
            )
        )
    )
    assert expensive.history.equity.iloc[-1] < 1090
    assert expensive.summary()['commissions'] > 0


def test_current_volume_and_future_prices_do_not_change_earlier_results():
    data = bars(
        [100, 101, 102, 103, 104, 105]
    )
    data['volume'] = [100, 1000, 1000, 1000, 1000, 1000]
    targets = pd.Series(
        1.,
        index = data.index
    )
    result = backtest(
        data,
        targets
    )
    assert result.history.quantity.iloc[1] == 1  # Previous volume 100 * 1%.
    altered = data.copy()
    altered.iloc[4:, :2] *= 10
    changed = backtest(
        altered,
        targets
    )
    pd.testing.assert_frame_equal(
        result.history.iloc[:4],
        changed.history.iloc[:4]
    )
    strategy = MovingAverageStrategy(
        fast = 2,
        slow = 3
    )
    pd.testing.assert_series_equal(
        strategy.signals(
            data.close
        ).iloc[:4],
        strategy.signals(
            altered.close
        ).iloc[:4]
    )


def test_walk_forward_training_windows_gap_and_predictor_history():
    close = bars(
        range(
            100,
            112
        )
    ).close
    training, seen = [], []

    def fit(
        train
    ):
        training.append(
            train.index
        )

        def predict(
            history
        ):
            seen.append(
                history.index[-1]
            )
            return float(
                history.iloc[-1] > train.mean()
            )

        return predict

    signals = walk_forward_signals(
        close,
        fit,
        WalkForwardConfig(
            train_size = 4,
            test_size = 3,
            gap = 1
        )
    )
    assert training[0].equals(
        close.index[:4]
    )
    assert training[1].equals(
        close.index[3:7]
    )
    assert seen == list(
        close.index[5:]
    )
    assert signals.iloc[:5].eq(
        0
    ).all()
    assert signals.iloc[5:].eq(
        1
    ).all()


def test_tail_risk_fractional_mass_and_stress_pnl():
    result = TailRisk(
        confidence = .625
    ).compute(
        [-.1, -.2, .1, 0],
        capital = 100
    )
    assert result['expected_shortfall'] == pytest.approx(
        (20 + .5 * 10) / 1.5
    )
    assert result['expected_shortfall'] >= result['var']
    p = Portfolio()
    apply(
        p,
        10,
        100
    )
    assert stress_test(
        p,
        {'TEST': 100},
        {'crash': {'TEST': -.2}}
    ) == {'crash': -200}


def test_validation_rejects_invalid_configs_data_and_short_signals():
    with pytest.raises(
            ValidationError
    ):
        MovingAverageStrategy(
            fast = 5,
            slow = 3
        )
    with pytest.raises(
            ValidationError
    ):
        quote(
            bid = 102,
            ask = 100
        )
    with pytest.raises(
            ValidationError
    ):
        order(
            0
        )
    data = bars(
        [100, 101]
    )
    with pytest.raises(
            ValueError,
            match = 'Short'
    ):
        backtest(
            data,
            pd.Series(
                -1.,
                index = data.index
            )
        )
    with pytest.raises(
            ValueError
    ):
        backtest(
            data.set_axis(
                data.index.tz_localize(
                    None
                )
            )
        )


def test_pre_trade_hook_can_block_a_fill_without_mutating_account():
    def check(
        candidate
    ):
        if candidate.positions['TEST'].quantity > 2:
            raise ValueError(
                'Custom exposure limit.'
            )

    broker = PaperBroker(
        [STOCK],
        pre_trade_check = check
    )
    state = broker.submit(
        order(
            3
        )
    )
    broker.on_quote(
        quote(),
        NOW
    )
    assert state.status == 'rejected' and state.reason == 'Custom exposure limit.'
    assert broker.portfolio.cash == 10000 and not broker.portfolio.fills


def test_restored_short_portfolio_starts_peak_at_marked_equity():
    p = Portfolio(
        cash = 1000
    )
    apply(
        p,
        -5,
        100
    )
    broker = PaperBroker(
        [STOCK],
        portfolio = p,
        limits = RiskLimits(
            allow_short = True
        )
    )
    state = broker.submit(
        order(
            -1
        )
    )
    broker.on_quote(
        quote(),
        NOW
    )
    assert broker.peak_equity == 1000
    assert state.status == 'filled'


def test_options_reject_fractional_short_and_expired_orders():
    contract = OptionContract(
        symbol = 'C',
        underlying = 'TEST',
        strike = 100,
        expiry = NOW + timedelta(
            days = 1
        ),
        option_type = 'call'
    )
    broker = PaperBroker(
        [contract],
        limits = RiskLimits(
            allow_short = True
        )
    )
    assert broker.submit(
        Order(
            symbol = 'C',
            quantity = 1.5,
            timestamp = NOW
        )
    ).status == 'rejected'
    short = broker.submit(
        Order(
            symbol = 'C',
            quantity = -1,
            timestamp = NOW
        )
    )
    broker.on_quote(
        Quote(
            symbol = 'C',
            timestamp = NOW,
            bid = 4,
            ask = 5
        ),
        NOW
    )
    assert short.status == 'rejected'
    assert broker.submit(
        Order(
            symbol = 'C',
            quantity = 1,
            timestamp = contract.expiry
        )
    ).status == 'rejected'


def test_positions_with_stale_marks_and_invalid_reconciliation_fail_closed():
    other = Instrument(
        symbol = 'OTHER'
    )
    broker = PaperBroker(
        [STOCK, other]
    )
    broker.submit(
        order(
            1
        )
    )
    broker.on_quote(
        quote(),
        NOW
    )
    state = broker.submit(
        Order(
            symbol = 'OTHER',
            quantity = 1,
            timestamp = NOW + timedelta(
                seconds = 61
            )
        )
    )
    broker.on_quote(
        Quote(
            symbol = 'OTHER',
            timestamp = NOW + timedelta(
                seconds = 61
            ),
            bid = 10,
            ask = 10
        ),
        NOW + timedelta(
            seconds = 61
        )
    )
    assert state.status == 'rejected' and state.reason is not None and 'stale' in state.reason
    with pytest.raises(
            ValueError
    ):
        broker.reconcile(
            {},
            float(
                'nan'
            )
        )


def test_stale_marks_do_not_raise_the_equity_peak():
    other = Instrument(
        symbol = 'OTHER'
    )
    portfolio = Portfolio(
        cash = 10000
    )
    apply(
        portfolio,
        1,
        100
    )
    portfolio.apply(
        Fill(
            order_id = 'other',
            symbol = 'OTHER',
            quantity = 1,
            price = 100,
            timestamp = NOW
        ),
        other
    )
    broker = PaperBroker(
        [STOCK, other],
        portfolio
    )
    broker.on_quote(
        quote(
            bid = 100,
            ask = 100
        ),
        NOW
    )
    broker.on_quote(
        Quote(
            symbol = 'OTHER',
            timestamp = NOW,
            bid = 100,
            ask = 100
        ),
        NOW
    )
    assert broker.peak_equity == 10000
    later = NOW + timedelta(
        seconds = 61
    )
    broker.on_quote(
        Quote(
            symbol = 'OTHER',
            timestamp = later,
            bid = 1000,
            ask = 1000
        ),
        later
    )
    assert broker.peak_equity == 10000  # TEST still has a stale mark.
    broker.on_quote(
        quote(
            62,
            bid = 50,
            ask = 50
        ),
        NOW + timedelta(
            seconds = 62
        )
    )
    assert broker.peak_equity == 10850


def test_expired_positions_block_execution_until_settlement():
    expiry = NOW + timedelta(
        seconds = 1
    )
    contract = OptionContract(
        symbol = 'C',
        underlying = 'TEST',
        strike = 100,
        expiry = expiry,
        option_type = 'call'
    )
    broker = PaperBroker(
        [contract, STOCK]
    )
    broker.submit(
        Order(
            symbol = 'C',
            quantity = 1,
            timestamp = NOW
        )
    )
    broker.on_quote(
        Quote(
            symbol = 'C',
            timestamp = NOW,
            bid = 5,
            ask = 5
        ),
        NOW
    )
    state = broker.submit(
        Order(
            symbol = 'TEST',
            quantity = 1,
            timestamp = NOW + timedelta(
                seconds = 2
            )
        )
    )
    broker.on_quote(
        quote(
            2
        ),
        NOW + timedelta(
            seconds = 2
        )
    )
    assert state.status == 'rejected' and state.reason is not None and 'settled' in state.reason
    assert broker.portfolio.positions.get(
        'TEST'
    ) is None
    broker.settle_option(
        'C',
        Quote(
            symbol = 'TEST',
            timestamp = expiry,
            bid = 100,
            ask = 100
        ),
        NOW + timedelta(
            seconds = 2
        )
    )
    state = broker.submit(
        Order(
            symbol = 'TEST',
            quantity = 1,
            timestamp = NOW + timedelta(
                seconds = 3
            )
        )
    )
    broker.on_quote(
        quote(
            3
        ),
        NOW + timedelta(
            seconds = 3
        )
    )
    assert state.status == 'filled'
