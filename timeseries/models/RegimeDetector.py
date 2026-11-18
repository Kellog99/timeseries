# ruff: noqa: N999  # Preserve public import paths.
import numpy as np
import pandas as pd

from timeseries.model_register import ModelFactory
from timeseries.registry import Configuration


class ConfigRegimeDetector(
    Configuration
):
    """Regime count is supplied to detect_regimes()."""


@ModelFactory.register(
    name = 'RegimeDetector',
    description = 'Volatility-based regime classification.',
    domain = 'technical_analysis',
)
class RegimeDetector:
    """Causal volatility quantiles; -1 denotes insufficient warm-up history."""
    CONFIG_T = ConfigRegimeDetector

    @staticmethod
    def detect_regimes(
        returns,
        n_regimes = 2
    ):
        if isinstance(
                n_regimes,
                bool
        ) or not isinstance(
            n_regimes,
            int
        ) or n_regimes < 2:
            raise ValueError(
                'n_regimes must be an integer >= 2.'
            )
        values = np.asarray(
            returns,
            dtype = float
        )
        if values.ndim != 1 or not np.isfinite(
                values
        ).all():
            raise ValueError(
                'Expected finite one-dimensional returns.'
            )
        vol = pd.Series(
            values
        ).rolling(
            window = 20
        ).std()
        thresholds = [vol.expanding(
            min_periods = 1
        ).quantile(
            q
        ).shift(
            1
        )
            for q in np.linspace(
                0,
                1,
                n_regimes + 1
            )[1:-1]]
        regimes = np.full(
            len(
                values
            ),
            -1,
            dtype = int
        )
        ready = vol.notna() & thresholds[0].notna()
        regimes[ready] = sum(
            (vol[ready] > threshold[ready]).astype(
                int
            ) for threshold in thresholds
        )
        return regimes
