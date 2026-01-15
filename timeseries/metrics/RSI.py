import numpy as np
import pandas as pd


class RSI:
    """Relative Strength Index"""

    @staticmethod
    def calculate(data, period=14):
        """Calculate RSI"""
        deltas = np.diff(data)

        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = pd.Series(gains).rolling(window=period).mean().values
        avg_loss = pd.Series(losses).rolling(window=period).mean().values

        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))

        return np.concatenate([[np.nan], rsi])
