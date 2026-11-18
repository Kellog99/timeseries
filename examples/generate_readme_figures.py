"""Generate the README figures from deterministic synthetic data, without network access."""
import argparse
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import Field

from timeseries.instruments import OptionContract
from timeseries.models.ARIMA import ARIMAPredictor
from timeseries.models.BlackScholes import BlackScholes
from timeseries.options import OptionMarket, OptionMarketQuote, VolatilitySurface
from timeseries.registry import Configuration


class FigureConfig(
    Configuration
):
    output_dir: Path = Path(
        __file__
    ).resolve().parents[1] / 'docs' / 'assets'
    seed: int = Field(
        7,
        ge = 0,
        strict = True
    )
    train_size: int = Field(
        140,
        ge = 10,
        strict = True
    )
    test_size: int = Field(
        50,
        ge = 2,
        strict = True
    )


def volatility_example() -> VolatilitySurface:
    """Invert synthetic European calls across 7 strikes and 5 expiries."""
    now = datetime(
        2026,
        1,
        1,
        tzinfo = UTC
    )
    market = OptionMarket(
        as_of = now,
        spot = 100,
        risk_free_rate = .03,
        dividend_yield = .01
    )
    quotes = []
    for days in (60, 120, 180, 270, 365):
        for strike in np.linspace(
                80,
                120,
                7
        ):
            m = np.log(
                strike / market.spot
                )
            vol = .2 - .16 * m + .55 * m ** 2 + .025 * days / 365
            premium = float(
                BlackScholes(
                    .03,
                    vol,
                    .01
                ).price(
                    100,
                    strike,
                    days / 365
                    )
            )
            contract = OptionContract(
                symbol = f'C-{strike:.2f}-{days}',
                underlying = 'SYNTHETIC',
                strike = float(
                    strike
                ),
                expiry = now + timedelta(
                    days = days
                ),
                option_type = 'call'
            )
            # Proportional spread keeps even small OTM premiums positive.
            quotes.append(
                OptionMarketQuote(
                    contract = contract,
                    observed_at = now,
                    bid = premium * .99,
                    ask = premium * 1.01
                )
            )
    return VolatilitySurface.from_quotes(
        quotes,
        market
    )


def forecast_example(
    observed: pd.Series,
    train_size: int
) -> pd.Series:
    """Fit once on the prefix; predict each holdout point using observations strictly before it."""
    if not 10 <= train_size < len(
            observed
    ):
        raise ValueError(
            'Provide at least 10 training observations and a nonempty holdout.'
        )
    train = observed.iloc[:train_size]
    center = float(
        train.mean()
    )
    model = ARIMAPredictor(
        p = 2,
        d = 0,
        q = 0
    )
    model.fit(
        train.to_numpy() - center
        )
    predictions = [float(
        model.predict(
            observed.iloc[:i].to_numpy() - center
            )[0] + center
        )
                   for i in range(
            train_size,
            len(
                observed
            )
        )]
    return pd.Series(
        predictions,
        index = observed.index[train_size:],
        name = 'forecast'
    )


def generate(
    config: FigureConfig | None = None
) -> list[Path]:
    import matplotlib
    matplotlib.use(
        'Agg'
    )
    from matplotlib import pyplot as plt

    from timeseries.plotting import plot_forecast, plot_volatility_surface

    c = config or FigureConfig()
    rng = np.random.default_rng(
        c.seed
    )
    values = np.zeros(
        c.train_size + c.test_size
        )
    for i in range(
            2,
            len(
                    values
            )
    ):
        values[i] = 1.5 * values[i - 1] - .7 * values[i - 2] + rng.normal(
            0,
            .6
        )
    observed = pd.Series(
        100 + values,
        index = pd.date_range(
            '2026-01-01',
            periods = len(
                values
            ),
            tz = 'UTC'
        )
        )
    predicted = forecast_example(
        observed,
        c.train_size
    )
    figures = {
        'volatility_surface.png': plot_volatility_surface(
            volatility_example()
        ),
        'forecasting.png': plot_forecast(
            observed,
            predicted,
            title = 'AR(2) · one-step forecasts on a held-out window'
        ),
    }
    c.output_dir.mkdir(
        parents = True,
        exist_ok = True
    )
    paths = []
    for name, figure in figures.items():
        figure.suptitle(
            'Synthetic data · reproducible offline example',
            fontsize = 10,
            color = '#667788'
        )
        path = c.output_dir / name
        figure.savefig(
            path,
            dpi = 160,
            facecolor = 'white'
        )
        plt.close(
            figure
        )
        paths.append(
            path
        )
    return paths


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description = __doc__
    )
    parser.add_argument(
        '--output-dir',
        type = Path,
        default = FigureConfig.model_fields['output_dir'].default
    )
    parser.add_argument(
        '--seed',
        type = int,
        default = 7
    )
    args = parser.parse_args()
    for path in generate(
            FigureConfig(
                    output_dir = args.output_dir,
                    seed = args.seed
            )
    ):
        print(
            path
        )
