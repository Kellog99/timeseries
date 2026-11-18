import numpy as np
import pandas as pd
import pytest

from examples.generate_readme_figures import forecast_example


def test_holdout_targets_and_future_observations_do_not_leak_into_forecasts():
    observed = pd.Series(
        100 + np.sin(np.arange(90) * .4), index = pd.date_range('2026-01-01', periods = 90, tz = 'UTC')
        )
    forecast = forecast_example(observed, train_size = 40)
    changed = observed.copy()
    changed.iloc[60:] += 10
    changed_forecast = forecast_example(changed, train_size = 40)
    # Even the prediction FOR observation 60 must be identical: it uses history through 59.
    pd.testing.assert_series_equal(forecast.iloc[:21], changed_forecast.iloc[:21])
    assert forecast.index.equals(observed.index[40:])
    assert np.isfinite(forecast).all()


def test_plot_rejects_forecasts_with_unobserved_timestamps():
    from timeseries.plotting import plot_forecast

    observed = pd.Series([100., 101., 102.], index = pd.date_range('2026-01-01', periods = 3, tz = 'UTC'))
    predicted = pd.Series([103.], index = pd.date_range('2026-01-04', periods = 1, tz = 'UTC'))
    with pytest.raises(ValueError, match = 'align'):
        plot_forecast(observed, predicted)
