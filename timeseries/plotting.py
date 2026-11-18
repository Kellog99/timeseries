"""Matplotlib figures for observed volatility surfaces and aligned forecasts."""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter

from timeseries.options import VolatilitySurface


def plot_volatility_surface(
    surface: VolatilitySurface,
    grid_size: int = 40
) -> Figure:
    """Plot the shared strike domain as a 3D surface and heatmap; no extrapolation."""
    times = sorted({p.maturity for p in surface.points})
    low = max(min(p.strike for p in surface.points if p.maturity == t) for t in times)
    high = min(max(p.strike for p in surface.points if p.maturity == t) for t in times)
    if len(times) < 2 or low >= high or isinstance(grid_size, bool) or not isinstance(grid_size, int) or grid_size < 2:
        raise ValueError('Plotting requires two maturities, overlapping strike intervals and grid_size >= 2.')
    strikes, maturities = np.meshgrid(np.linspace(low, high, grid_size), np.linspace(times[0], times[-1], grid_size))
    vol = np.array([surface.volatility(float(k), float(t)) for k, t in zip(strikes.flat, maturities.flat)]).reshape(
        strikes.shape
        )
    fig = plt.figure(figsize = (12, 4.8), layout = 'constrained')
    left = fig.add_subplot(121, projection = '3d')
    left.plot_surface(strikes, maturities, vol * 100, cmap = 'viridis', alpha = .9, linewidth = 0)
    left.set(
        xlabel = 'Strike', ylabel = 'Maturity (years)', zlabel = 'Implied volatility (%)',
        title = 'Smile and term structure'
        )
    left.view_init(elev = 26, azim = -125)
    right = fig.add_subplot(122)
    heat = right.pcolormesh(strikes, maturities, vol, cmap = 'viridis', shading = 'auto')
    right.scatter(
        [p.strike for p in surface.points], [p.maturity for p in surface.points],
        s = 12, facecolors = 'none', edgecolors = 'white', linewidths = .7, label = 'Input quotes'
        )
    right.set(xlabel = 'Strike', ylabel = 'Maturity (years)', title = 'Interpolated implied volatility')
    right.legend(loc = 'upper right', facecolor = '#233c50', labelcolor = 'white', fontsize = 8)
    fig.colorbar(heat, ax = right, format = PercentFormatter(1), label = 'Annualized volatility')
    return fig


def plot_forecast(
    observed: pd.Series,
    predicted: pd.Series,
    *,
    title: str = 'Rolling one-step forecast'
) -> Figure:
    """Compare forecasts against observations and previous-observation persistence."""
    for series in (observed, predicted):
        if (not isinstance(series.index, pd.DatetimeIndex) or series.empty
                or not series.index.is_monotonic_increasing or series.index.has_duplicates
                or not np.isfinite(series.to_numpy(dtype = float)).all()):
            raise ValueError('Provide finite, ordered, unique datetime-indexed series.')
    if not predicted.index.isin(observed.index).all() or predicted.index[0] <= observed.index[0]:
        raise ValueError('Forecasts must align to observed timestamps with earlier history for the baseline.')
    actual = observed.reindex(predicted.index)
    baseline = observed.shift(1).reindex(predicted.index)
    mae = float((actual - predicted).abs().mean())
    baseline_mae = float((actual - baseline).abs().mean())
    fig, ax = plt.subplots(figsize = (12, 4.8), layout = 'constrained')
    ax.axvspan(observed.index[0], predicted.index[0], color = '#edf1f5', label = 'Training window')
    ax.axvline(predicted.index[0], color = '#7d8b99', linestyle = ':', linewidth = 1)
    ax.plot(observed.index, observed, color = '#193b56', linewidth = 1.6, label = 'Observed')
    ax.plot(
        predicted.index, baseline, color = '#8c96a0', linestyle = '--', linewidth = 1,
        label = f'Persistence · MAE {baseline_mae:.3f}'
        )
    ax.plot(predicted.index, predicted, color = '#d56b27', linewidth = 1.5, label = f'Forecast · MAE {mae:.3f}')
    ax.set(title = title, xlabel = 'Date', ylabel = 'Series value')
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis = 'y', color = '#e2e7ec', linewidth = .6)
    ax.legend(loc = 'upper left', ncols = 2, fontsize = 9)
    return fig
