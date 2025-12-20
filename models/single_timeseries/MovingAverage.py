# ==================== MOVING AVERAGES ====================
import numpy as np
import pandas as pd


class MovingAverages:
    """Various moving average implementations"""

    @staticmethod
    def sma(data, window):
        """Simple Moving Average"""
        return pd.Series(data).rolling(window=window).mean().values

    @staticmethod
    def ema(data, span):
        """Exponential Moving Average"""
        return pd.Series(data).ewm(span=span, adjust=False).mean().values

    @staticmethod
    def wma(data, window):
        """Weighted Moving Average"""
        weights = np.arange(1, window + 1)
        wma = pd.Series(data).rolling(window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
        return wma.values

    @staticmethod
    def hull_ma(data, window):
        """Hull Moving Average - reduced lag"""
        half_length = window // 2
        sqrt_length = int(np.sqrt(window))

        wma_half = MovingAverages.wma(data, half_length)
        wma_full = MovingAverages.wma(data, window)

        raw_hma = 2 * wma_half - wma_full
        return MovingAverages.wma(raw_hma, sqrt_length)
