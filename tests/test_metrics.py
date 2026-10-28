from typing import Any
import math

import numpy as np
import pytest
import torch

from timeseries.metrics import MetricRegister


@pytest.mark.parametrize("metric_id", ["sharpe", "omega", "sortino", "volatility", "max_drawdown", "calmar", "total_return"])
def test_financial_formulas_streaming_and_reset(metric_id):
    returns = np.array([.10, -.20, .05, .03])
    annualization = 12
    wealth = np.r_[1., (1 + returns).cumprod()]
    drawdown = (1 - wealth / np.maximum.accumulate(wealth)).max()
    expected = {
        "sharpe": returns.mean() / returns.std(ddof=1) * math.sqrt(annualization),
        "omega": returns.clip(0).sum() / (-returns).clip(0).sum(),
        "sortino": returns.mean() / np.sqrt(np.minimum(returns, 0).__pow__(2).mean()) * math.sqrt(annualization),
        "volatility": returns.std(ddof=1) * math.sqrt(annualization),
        "max_drawdown": drawdown,
        "calmar": ((1 + returns).prod() ** (annualization / len(returns)) - 1) / drawdown,
        "total_return": (1 + returns).prod() - 1,
    }
    kwargs: dict[str, Any] = {"periods_per_year": annualization} if metric_id in ["sharpe", "sortino", "volatility", "calmar"] else {}
    metric = MetricRegister.create(metric_id, **kwargs)
    metric.update(returns[:2].tolist())
    metric.update(torch.tensor(returns[2:]))
    assert metric.compute().item() == pytest.approx(expected[metric_id])
    metric.reset()
    metric.update(returns.tolist())
    assert metric.compute().item() == pytest.approx(expected[metric_id])


def test_thresholds_and_initial_loss():
    metric = MetricRegister.create("omega", threshold=.01)
    metric.update([.03, -.01])
    assert metric.compute() == pytest.approx(1.)
    metric = MetricRegister.create("sharpe", risk_free_rate=.01, periods_per_year=1)
    metric.update([.03, -.01])
    assert metric.compute() == pytest.approx(0.)
    metric = MetricRegister.create("max_drawdown")
    metric.update([-.25])
    assert metric.compute() == pytest.approx(.25)
    metric.reset()
    metric.update([-1., .5])
    assert metric.compute() == pytest.approx(1.)


@pytest.mark.parametrize("values", [[float("nan")], [float("inf")], [-1.01], [[.1, .2]]])
def test_invalid_returns(values):
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


@pytest.mark.parametrize("metric_id, expected", [("moving_average", [2., 5.]), ("moving_variance", [2/3, 1.])])
def test_windows_include_entire_partial_tail(metric_id, expected):
    metric = MetricRegister.create(metric_id, period=3)
    metric.update([1., 2.])
    metric.update([3., 4., 6.])
    assert metric.compute() == pytest.approx(expected)
    metric.reset()
    metric.update([7.])
    assert metric.compute() == pytest.approx([7.] if metric_id == "moving_average" else [0.])


@pytest.mark.parametrize("metric_id", ["canonical", "rv", "distance", "pearson"])
def test_dependence_metrics(metric_id):
    metric = MetricRegister.create(metric_id)
    values = torch.arange(6.)
    metric.update(values[:3], values[:3] * 2)
    metric.update(values[3:], values[3:] * 2)
    assert metric.compute() == pytest.approx(1.)
    metric.reset()
    metric.update(values, -values)
    assert metric.compute() == pytest.approx(-1. if metric_id == "pearson" else 1.)
