# ==================== BOLLINGER BANDS ====================
import pandas as pd

from timeseries.models.single_timeseries.MovingAverage import MovingAverages


class BollingerBands:
    """Bollinger Bands for volatility analysis"""

    @staticmethod
    def calculate(data, window=20, num_std=2):
        """Calculate Bollinger Bands"""
        sma = MovingAverages.sma(data, window)
        std = pd.Series(data).rolling(window=window).std().values

        upper = sma + (num_std * std)
        lower = sma - (num_std * std)

        return {'middle': sma, 'upper': upper, 'lower': lower}
