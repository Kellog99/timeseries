# ==================== REGIME DETECTION ====================
import numpy as np
import pandas as pd


class RegimeDetector:
    """Hidden Markov Model-style regime detection"""

    @staticmethod
    def detect_regimes(returns, n_regimes=2):
        """Simple regime detection based on volatility clustering"""
        # Calculate rolling volatility
        vol = pd.Series(returns).rolling(window=20).std().values

        # K-means style clustering
        thresholds = np.percentile(vol[~np.isnan(vol)],
                                   np.linspace(0, 100, n_regimes + 1)[1:-1])

        regimes = np.digitize(vol, thresholds)
        return regimes
