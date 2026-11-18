from datetime import UTC, date, datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fastapi import HTTPException

from timeseries.metrics.returns import get_return
from timeseries.models.GARCH import SimpleGARCH
from timeseries.models.RegimeDetector import RegimeDetector


def test_returns_include_holidays_and_missing_sessions():
    result = get_return(
        {'2026-05-22': 100, '2026-05-26': 102, '2026-05-27': 103}
    )
    assert result[date(
        2026,
        5,
        26
    )] == pytest.approx(
        .02
    )
    assert np.prod(
        1 + np.array(
            list(
                result.values()
            )
        )
        ) == pytest.approx(
        1.03
    )
    with pytest.raises(
            ValueError
    ):
        get_return(
            {'2026-01-01': 0}
        )


def test_garch_forecast_uses_latest_observation_once_and_calibration_scales():
    model = SimpleGARCH(
        omega = .01,
        alpha = .1,
        beta = .8
    )
    values = np.array(
        [.1, -.1, .2]
    )
    variance = values.var()
    for value in values:
        variance = .01 + .1 * value ** 2 + .8 * variance
    np.testing.assert_allclose(
        model.fit_predict(
            values,
            2
        ),
        [
            np.sqrt(
                variance
            ), np.sqrt(
            .01 + .9 * variance
            )
        ]
    )
    returns = np.random.default_rng(
        12
    ).normal(
        0,
        .01,
        300
    )
    first = SimpleGARCH().fit(
        returns
    )
    scaled = SimpleGARCH().fit(
        returns * 10
        )
    assert first.alpha + first.beta < 1
    np.testing.assert_allclose(
        scaled.fit_predict(
            returns * 10
            ),
        first.fit_predict(
            returns
        ) * 10,
        rtol = 1e-3
        )
    with pytest.raises(
            ValueError
    ):
        SimpleGARCH().fit(
            np.zeros(
                30
            )
        )


def test_regime_labels_are_causal():
    values = np.random.default_rng(
        1
    ).normal(
        size = 100
    )
    prefix = RegimeDetector.detect_regimes(
        values[:60]
    )
    full = RegimeDetector.detect_regimes(
        np.r_[values[:60], values[60:] * 100]
    )
    np.testing.assert_array_equal(
        prefix,
        full[:60]
    )
    assert (prefix[:20] == -1).all()


def test_fundamentals_use_provider_data_and_null_for_missing(
    monkeypatch
):
    from services.routers.get_indicators import get_indicators
    monkeypatch.setattr(
        'services.routers.get_indicators.yf.Ticker',
        lambda
            ticker: SimpleNamespace(
            info = {'marketCap': 123 if ticker == 'ONE' else 456, 'debtToEquity': 150, 'profitMargins': .2}
        )
    )
    assert get_indicators(
        'ONE'
    ).market_cap == 123
    assert get_indicators(
        'TWO'
    ).market_cap == 456
    assert get_indicators(
        'ONE'
    ).debt_to_equity == 1.5
    assert get_indicators(
        'ONE'
    ).pe_ratio is None


def test_portfolio_valuation_endpoint_and_saved_snapshot(
    tmp_path
):
    from services.models.portfolio import PortfolioSnapshot
    from services.routers.load_data import evaluate_portfolio, get_portfolio_analysis
    from timeseries.instruments import Instrument
    from timeseries.trading import Fill, Portfolio
    now = datetime.now(
        UTC
    )
    p = Portfolio(
        cash = 1000
    )
    p.apply(
        Fill(
            order_id = 'x',
            symbol = 'TEST',
            quantity = 2,
            price = 100,
            commission = 1,
            timestamp = now
        ),
        Instrument(
            symbol = 'TEST'
        )
    )
    snapshot = PortfolioSnapshot(
        as_of = now,
        portfolio = p,
        marks = {'TEST': 110}
    )
    valuation = evaluate_portfolio(
        snapshot
    )
    assert valuation.equity == 1019
    assert valuation.realized_pnl + valuation.unrealized_pnl == 19
    (tmp_path / 'portfolio.json').write_text(
        snapshot.model_dump_json()
    )
    assert get_portfolio_analysis(
        tmp_path
    ) == valuation
    with pytest.raises(
            HTTPException
    ) as exc:
        evaluate_portfolio(
            PortfolioSnapshot(
                as_of = now,
                portfolio = p,
                marks = {}
            )
        )
    assert exc.value.status_code == 422


def test_option_loader_filters_invalid_quotes_without_inventing_exchange_time(
    monkeypatch
):
    from services.routers.options import load_option_chain
    now = datetime.now(
        UTC
    )
    frame = pd.DataFrame(
        [
            {
                'contractSymbol': 'OK', 'strike': 100, 'bid': 3, 'ask': 4, 'volume': 2, 'openInterest': 100,
                'lastTradeDate': now
            },
            {
                'contractSymbol': 'BAD', 'strike': 100, 'bid': 5, 'ask': 4, 'volume': 0, 'openInterest': 0,
                'lastTradeDate': now
            },
        ]
    )
    monkeypatch.setattr(
        'services.routers.options.yf.Ticker',
        lambda
            _: SimpleNamespace(
            option_chain = lambda
                expiry: SimpleNamespace(
                calls = frame,
                puts = frame.iloc[:0]
            )
        )
    )
    quotes = load_option_chain(
        'TEST',
        now + timedelta(
            days = 30
        )
        )
    assert len(
        quotes
    ) == 1 and quotes[0].contract.symbol == 'OK'
    assert quotes[0].contract.exercise == 'american'
    assert quotes[0].last_trade_at == now


def test_api_schema_includes_new_endpoints():
    from services.app import app
    paths = app.openapi()['paths']
    assert '/load/options' in paths
    assert '/load/portfolio/evaluate' in paths


def test_invalid_price_cache_is_rebuilt_with_explicit_adjustment_policy(
    monkeypatch,
    tmp_path
):
    from services.routers.load_data import get_data

    calls = []

    def history(
        **kwargs
    ):
        calls.append(
            kwargs
        )
        return pd.DataFrame(
            {
                'Date': [
                    pd.Timestamp(
                        '2026-01-01'
                    )
                ], 'Open': [100], 'Close': [101],
                'High': [102], 'Low': [99], 'Volume': [1000]
            }
        )

    monkeypatch.setattr(
        'services.routers.load_data.yf.Ticker',
        lambda
            _: SimpleNamespace(
            info = {},
            history = history
        )
    )
    (tmp_path / 'TEST.json').write_text(
        '{broken'
    )
    result = get_data(
        'TEST',
        path_data = tmp_path,
        step_size = 1,
        save = False
    )
    assert result.history['2026-01-01'].Close == 101
    assert calls == [{'period': 'max', 'auto_adjust': False}]
