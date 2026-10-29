import numpy as np
import pytest
from pydantic import ValidationError

from timeseries.models import ModelFactory


def test_reference_prices_through_factory():
    config = ModelFactory.get_config('blackscholes', risk_free_rate = 0.05, volatility = 0.2)
    model = ModelFactory.create(config = config)
    assert model.price(100, 100, 1) == pytest.approx(10.450583572185565)
    assert model.price(100, 100, 1, 'put') == pytest.approx(5.573526022256971)
    assert ModelFactory.get_info('blackscholes').domain == 'option_pricing'


def test_broadcast_prices_and_put_call_parity():
    model = ModelFactory.create('blackscholes', risk_free_rate = -0.01, volatility = 0.3)
    spot = np.array([[80.0], [100.0], [120.0]])
    strike = np.array([90.0, 100.0])
    maturity = np.array([0.0, 2.0])
    call = model.price(spot, strike, maturity)
    put = model.price(spot, strike, maturity, 'put')
    assert call.shape == (3, 2)
    np.testing.assert_allclose(call - put, spot - strike * np.exp(0.01 * maturity), atol = 1e-12)
    np.testing.assert_allclose(call[:, 0], [0, 10, 30])
    assert np.all(call >= 0) and np.all(put >= 0)
    for i in range(3):
        for j in range(2):
            assert call[i, j] == pytest.approx(model.price(spot[i, 0], strike[j], maturity[j]))


def test_expiry_and_zero_volatility():
    model = ModelFactory.create('blackscholes', risk_free_rate = 0.05)
    np.testing.assert_array_equal(model.price([80, 100, 120], 100, 0), [0, 0, 20])
    np.testing.assert_array_equal(model.price([80, 100, 120], 100, 0, 'put'), [20, 0, 0])
    deterministic = ModelFactory.create('blackscholes', risk_free_rate = 0.05, volatility = 0)
    assert deterministic.price(100, 100, 1) == pytest.approx(100 - 100 * np.exp(-0.05))
    assert deterministic.price(100, 100, 1, 'put') == 0
    assert deterministic.price(80, 100, 1, 'put') == pytest.approx(100 * np.exp(-0.05) - 80)


def test_deep_out_of_the_money_prices_are_finite_and_nonnegative():
    model = ModelFactory.create('blackscholes')
    for option_type in ('call', 'put'):
        prices = model.price([1e-10, 100, 1e10], 100, 1e-10, option_type)
        assert np.isfinite(prices).all()
        assert np.all(prices >= 0)


@pytest.mark.parametrize('kwargs', [
    {'volatility': -0.1}, {'volatility': float('inf')},
    {'risk_free_rate': float('nan')}, {'risk_free_rate': float('inf')},
])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValidationError):
        ModelFactory.create('blackscholes', **kwargs)


@pytest.mark.parametrize('spot,strike,maturity', [
    (0, 100, 1), (-1, 100, 1), (100, 0, 1), (100, -1, 1), (100, 100, -1),
    ([100, float('nan')], 100, 1), (100, float('inf'), 1), (100, 100, float('inf')),
])
def test_invalid_pricing_inputs(spot, strike, maturity):
    with pytest.raises(ValueError):
        ModelFactory.create('blackscholes').price(spot, strike, maturity)


def test_invalid_option_type_and_incompatible_shapes():
    model = ModelFactory.create('blackscholes')
    with pytest.raises(ValueError, match = 'option_type'):
        model.price(100, 100, 1, 'invalid')
    with pytest.raises(ValueError):
        model.price([90, 100], [90, 100, 110], 1)
