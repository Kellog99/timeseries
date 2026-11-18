from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from timeseries.instruments import Instrument, OptionContract
from timeseries.models import ModelFactory
from timeseries.models.BlackScholes import BlackScholes
from timeseries.options import (
    GreekLimits, OptionMarket, OptionMarketQuote, OptionScenario, VolatilityPoint, VolatilitySurface, portfolio_greeks,
)
from timeseries.trading import Fill, Portfolio

NOW = datetime(2026, 1, 1, tzinfo = timezone.utc)


@pytest.mark.parametrize('kind', ['call', 'put'])
def test_dividend_parity_iv_roundtrip_and_finite_difference_greeks(
    kind
):
    r, q, v, s, k, t = .04, .02, .3, 105, 100, .7
    model = BlackScholes(r, v, q)
    price = model.price(s, k, t, kind)
    assert model.implied_volatility(price, s, k, t, kind) == pytest.approx(v)
    assert model.price(s, k, t) - model.price(s, k, t, 'put') == pytest.approx(s * np.exp(-q * t) - k * np.exp(-r * t))
    g = model.greeks(s, k, t, kind)
    h = 1e-3
    assert g.delta == pytest.approx(
        (model.price(s + h, k, t, kind) - model.price(s - h, k, t, kind)) / (2 * h), rel = 1e-6
        )
    assert g.gamma == pytest.approx(
        (model.price(s + h, k, t, kind) - 2 * price + model.price(s - h, k, t, kind)) / h ** 2, rel = 1e-4
        )
    assert g.vega == pytest.approx(
        (BlackScholes(r, v + h, q).price(s, k, t, kind) - BlackScholes(r, v - h, q).price(s, k, t, kind)) / (2 * h),
        rel = 1e-5
        )
    assert g.rho == pytest.approx(
        (BlackScholes(r + h, v, q).price(s, k, t, kind) - BlackScholes(r - h, v, q).price(s, k, t, kind)) / (2 * h),
        rel = 1e-5
        )
    assert g.theta == pytest.approx(
        (model.price(s, k, t - h, kind) - model.price(s, k, t + h, kind)) / (2 * h), rel = 1e-5
        )


def test_iv_bounds_and_undefined_greeks():
    model = BlackScholes()
    assert model.implied_volatility(0, 100, 100, 1) == 0
    for premium in (-1, 100, float('nan')):
        with pytest.raises(ValueError):
            model.implied_volatility(premium, 100, 100, 1)
    with pytest.raises(ValueError):
        model.greeks(100, 100, 0)


def test_crr_european_convergence_and_american_early_exercise():
    euro = ModelFactory.create('binomial', risk_free_rate = .05, volatility = .2, steps = 1000, american = False)
    american = ModelFactory.create('binomial', risk_free_rate = .05, volatility = .2, steps = 1000)
    bs = BlackScholes(.05, .2)
    assert euro.price(100, 100, 1) == pytest.approx(bs.price(100, 100, 1), abs = .003)
    assert american.price(100, 100, 1) == pytest.approx(euro.price(100, 100, 1))
    assert american.price(80, 100, 1, 'put') > euro.price(80, 100, 1, 'put')
    assert american.price(80, 100, 1, 'put') >= 20
    assert american.price(80, 100, 0, 'put') == 20
    deterministic = ModelFactory.create('binomial', risk_free_rate = .05, volatility = 0)
    assert deterministic.price(80, 100, 1, 'put') == 20


def contract(
    strike = 100,
    maturity = 1,
    symbol = 'C'
):
    return OptionContract(
        symbol = symbol, underlying = 'TEST', strike = strike,
        expiry = NOW + timedelta(days = 365 * maturity), option_type = 'call'
        )


def test_surface_inverts_quotes_interpolates_variance_and_rejects_bad_data():
    market = OptionMarket(as_of = NOW, spot = 100, volatility = .25)
    quotes = []
    for t in (.5, 1):
        for k in (90, 110):
            price = market.model().price(100, k, t)
            quotes.append(
                OptionMarketQuote(contract = contract(k, t), observed_at = NOW, bid = price - .01, ask = price + .01)
                )
    surface = VolatilitySurface.from_quotes(quotes, market)
    assert surface.volatility(100, .75) == pytest.approx(.25)
    assert surface.volatility(90, .5) == pytest.approx(.25)
    with pytest.raises(ValueError):
        surface.volatility(80, .75)
    with pytest.raises(ValueError):
        surface.volatility(100, 2)
    with pytest.raises(ValueError, match = 'Stale'):
        VolatilitySurface.from_quotes(quotes, market.model_copy(update = {'as_of': NOW + timedelta(minutes = 5)}))
    with pytest.raises(ValueError, match = 'European'):
        VolatilitySurface.from_quotes(
            [quotes[0].model_copy(
                update = {
                    'contract': quotes[0].contract.model_copy(update = {'exercise': 'american'})
                }
            )], market
        )
    nonflat = VolatilitySurface(
        points = [VolatilityPoint(strike = 100, maturity = .5, volatility = .2),
                  VolatilityPoint(strike = 100, maturity = 1, volatility = .3)]
        )
    assert nonflat.volatility(100, .75) == pytest.approx(np.sqrt((.5 * .2 ** 2 + .3 ** 2) / 2 / .75))


def test_portfolio_greeks_contract_multipliers_limits_and_full_repricing():
    p = Portfolio()
    c = contract()
    p.apply(Fill(order_id = 'o', symbol = 'C', quantity = 2, price = 8, timestamp = NOW), c)
    p.apply(
        Fill(order_id = 's', symbol = 'TEST', quantity = -50, price = 100, timestamp = NOW), Instrument(symbol = 'TEST')
        )
    market = OptionMarket(as_of = NOW, spot = 100)
    exposures = portfolio_greeks(p, {'TEST': market})
    expected = market.model().greeks(100, 100, 1)
    assert exposures['TEST'].delta == pytest.approx(expected.delta * 200 - 50)
    assert exposures['TEST'].vega == pytest.approx(expected.vega * 200)
    with pytest.raises(ValueError, match = 'delta'):
        GreekLimits(max_abs_delta = 1).check(exposures)
    scenario = OptionScenario(spot_shocks = {'TEST': -.1}, volatility_shifts = {'TEST': .05}, days = 1)
    expected_pnl = 200 * (
                BlackScholes(volatility = .25).price(90, 100, 364 / 365) - market.model().price(100, 100, 1)) + 500
    assert scenario.pnl(p, {'TEST': market}) == pytest.approx(expected_pnl)
