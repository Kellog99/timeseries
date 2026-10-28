import numpy as np
import pandas as pd

from timeseries.register import ModelFactory
from timeseries.registry import Configuration


class ConfigRegimeDetector(Configuration):
    """Regime count is supplied to detect_regimes()."""


@ModelFactory.register(
    name='RegimeDetector',
    description='Volatility-based regime classification.',
    id='regimedetector',
    domain='technical_analysis',
)
class RegimeDetector:
    """Hidden Markov Model-style regime detection"""
    CONFIG_T = ConfigRegimeDetector

    @staticmethod
    def detect_regimes(returns, n_regimes=2):
        """Simple regime detection based on volatility clustering"""
        # Calculate rolling volatility
        vol = pd.Series(returns).rolling(window=20).std().to_numpy()

        # K-means style clustering
        thresholds = np.percentile(vol[~np.isnan(vol)],
                                   np.linspace(0, 100, n_regimes + 1)[1:-1])

        regimes = np.digitize(vol, thresholds)
        return regimes
