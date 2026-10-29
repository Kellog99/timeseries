import math
from typing import Any

import numpy as np
import pytest
import torch
from torchmetrics import Metric

from timeseries.metrics import MetricRegister


def test_metric_parameters_are_independent_dictionaries_without_configs():
    assert not hasattr(MetricRegister, 'get_config')
    assert MetricRegister.get_config_param('sharpe') == {'risk_free_rate': 0.0, 'periods_per_year': 252}
    assert MetricRegister.get_config_param('moving_average') == {'period': None}
    assert MetricRegister.get_config_param('pearson') == {}
    params = MetricRegister.get_config_param('sharpe')
    params['periods_per_year'] = 12
    metric = MetricRegister.create('sharpe', **params)
    assert metric.periods_per_year == 12
    assert MetricRegister.get_config_param('sharpe')['periods_per_year'] == 252
    for class_id in MetricRegister.get_list_classes():
        info = MetricRegister.get_info(class_id)
        assert not hasattr(info, 'config_type')
        assert not hasattr(info.class_type, 'CONFIG_T')
        MetricRegister.create(class_id, **MetricRegister.get_config_param(class_id))
    with pytest.raises(TypeError):
        MetricRegister.create('sharpe', typo = 1)
    with pytest.raises(TypeError):
        MetricRegister.create('sharpe', config = {})


def test_registration_does_not_require_a_metric_config():
    class CustomFactory(MetricRegister):
        pass

    @CustomFactory.register('Custom', 'Metric without a configuration.', domain = 'statistics')
    class Custom(Metric):
        def __init__(self, scale: float = 2.0):
            super().__init__()
            self.scale = scale
            self.add_state('total', default = torch.tensor(0.0), dist_reduce_fx = 'sum')

        def update(self, value):
            self.total += value * self.scale

        def compute(self):
            return self.total

    assert CustomFactory.get_config_param('custom') == {'scale': 2.0}
    metric = CustomFactory.create('custom', scale = 3)
    metric.update(torch.tensor(2.0))
    assert metric.compute().item() == 6
    assert 'custom' not in MetricRegister.get_list_classes()


@pytest.mark.parametrize('metric_id,kwargs', [
    ('sharpe', {'risk_free_rate': float('nan')}),
    ('sortino', {'target_return': float('inf')}),
    ('omega', {'threshold': float('nan')}),
    ('sharpe', {'periods_per_year': 0}),
    ('sortino', {'periods_per_year': -1}),
    ('volatility', {'periods_per_year': 1.5}),
    ('calmar', {'periods_per_year': True}),
    ('moving_average', {'period': 0}),
    ('moving_variance', {'period': -1}),
    ('moving_sr', {'period': 2.5}),
])
def test_metric_constructor_validation(metric_id, kwargs):
    implementation = MetricRegister.get_info(metric_id).class_type
    with pytest.raises(ValueError):
        implementation(**kwargs)
    with pytest.raises(ValueError):
        MetricRegister.create(metric_id, **kwargs)


@pytest.mark.parametrize(
    "metric_id", ["sharpe", "omega", "sortino", "volatility", "max_drawdown", "calmar", "total_return"]
    )
def test_financial_formulas_streaming_and_reset(
        metric_id
        ):
    returns = np.array([.10, -.20, .05, .03])
    annualization = 12
    wealth = np.r_[1., (1 + returns).cumprod()]
    drawdown = (1 - wealth / np.maximum.accumulate(wealth)).max()
    expected = {
        "sharpe": returns.mean() / returns.std(ddof = 1) * math.sqrt(annualization),
        "omega": returns.clip(0).sum() / (-returns).clip(0).sum(),
        "sortino": returns.mean() / np.sqrt(np.minimum(returns, 0).__pow__(2).mean()) * math.sqrt(annualization),
        "volatility": returns.std(ddof = 1) * math.sqrt(annualization),
        "max_drawdown": drawdown,
        "calmar": ((1 + returns).prod() ** (annualization / len(returns)) - 1) / drawdown,
        "total_return": (1 + returns).prod() - 1,
    }
    kwargs: dict[str, Any] = {"periods_per_year": annualization} if metric_id in ["sharpe", "sortino", "volatility",
                                                                                  "calmar"] else {}
    metric = MetricRegister.create(metric_id, **kwargs)
    metric.update(returns[:2].tolist())
    metric.update(torch.tensor(returns[2:]))
    assert metric.compute().item() == pytest.approx(expected[metric_id])
    metric.reset()
    metric.update(returns.tolist())
    assert metric.compute().item() == pytest.approx(expected[metric_id])


def test_thresholds_and_initial_loss():
    metric = MetricRegister.create("omega", threshold = .01)
    metric.update([.03, -.01])
    assert metric.compute() == pytest.approx(1.)
    metric = MetricRegister.create("sharpe", risk_free_rate = .01, periods_per_year = 1)
    metric.update([.03, -.01])
    assert metric.compute() == pytest.approx(0.)
    metric = MetricRegister.create("max_drawdown")
    metric.update([-.25])
    assert metric.compute() == pytest.approx(.25)
    metric.reset()
    metric.update([-1., .5])
    assert metric.compute() == pytest.approx(1.)


@pytest.mark.parametrize("values", [[float("nan")], [float("inf")], [-1.01], [[.1, .2]]])
def test_invalid_returns(
        values
        ):
    with pytest.raises(ValueError):
        MetricRegister.create("sharpe").update(values)


def test_empty_and_degenerate_returns():
    with pytest.warns(UserWarning):
        with pytest.raises(ValueError):
            MetricRegister.create("omega").compute()
    positive = MetricRegister.create("omega")
    positive.update([.1, .2])
    assert math.isinf(positive.compute())
    zero = MetricRegister.create("omega")
    zero.update([0., 0.])
    assert math.isnan(zero.compute())
    single = MetricRegister.create("sharpe")
    single.update([.1])
    assert math.isnan(single.compute())


@pytest.mark.parametrize("metric_id, expected", [("moving_average", [2., 5.]), ("moving_variance", [2 / 3, 1.])])
def test_windows_include_entire_partial_tail(
        metric_id,
        expected
        ):
    metric = MetricRegister.create(metric_id, period = 3)
    metric.update([1., 2.])
    metric.update([3., 4., 6.])
    assert metric.compute() == pytest.approx(expected)
    metric.reset()
    metric.update([7.])
    assert metric.compute() == pytest.approx([7.] if metric_id == "moving_average" else [0.])


@pytest.mark.parametrize("metric_id", ["canonical", "rv", "distance", "pearson"])
def test_dependence_metrics(
        metric_id
        ):
    metric = MetricRegister.create(metric_id)
    values = torch.arange(6.)
    metric.update(values[:3], values[:3] * 2)
    metric.update(values[3:], values[3:] * 2)
    assert metric.compute() == pytest.approx(1.)
    metric.reset()
    metric.update(values, -values)
    assert metric.compute() == pytest.approx(-1. if metric_id == "pearson" else 1.)
