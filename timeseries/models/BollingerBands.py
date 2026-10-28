import pandas as pd

from timeseries.models.MovingAverage import MovingAverages
from timeseries.register import ModelFactory
from timeseries.registry import Configuration


class ConfigBollingerBands(Configuration):
    """Window and standard-deviation multiplier are supplied to calculate()."""


# ==================== BOLLINGER BANDS ====================


@ModelFactory.register(
    name = 'BollingerBands',
    description = 'Moving-average bands based on rolling standard deviation.',
    domain = 'technical_analysis',
)
class BollingerBands:
    """Bollinger Bands for volatility analysis"""
    CONFIG_T = ConfigBollingerBands

    @staticmethod
    def calculate(
            data,
            window = 20,
            num_std = 2
            ):
        """Calculate Bollinger Bands"""
        sma = MovingAverages.sma(data, window)
        std = pd.Series(data).rolling(window = window).std().to_numpy()

        upper = sma + (num_std * std)
        lower = sma - (num_std * std)

        return {'middle': sma, 'upper': upper, 'lower': lower}
