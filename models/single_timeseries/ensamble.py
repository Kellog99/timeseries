# ==================== ENSEMBLE PREDICTOR ====================
from models.single_timeseries.ARIMA import ARIMAPredictor
from models.single_timeseries.KalmanFilter import KalmanFilter
from models.single_timeseries.MovingAverage import MovingAverages


class EnsemblePredictor:
    """Combine multiple predictors"""

    def __init__(self):
        self.kalman = KalmanFilter()
        self.arima = ARIMAPredictor()

    def predict(self, data, weights=None):
        """Weighted ensemble prediction"""
        if weights is None:
            weights = [0.3, 0.3, 0.4]  # Kalman, ARIMA, EMA

        # Kalman prediction
        kalman_pred = self.kalman.filter_series(data)[-1]

        # ARIMA prediction
        self.arima.fit(data)
        arima_pred = self.arima.predict(data, steps=1)[0]

        # EMA prediction (use last value as prediction)
        ema_pred = MovingAverages.ema(data, span=20)[-1]

        # Weighted combination
        ensemble = (weights[0] * kalman_pred +
                    weights[1] * arima_pred +
                    weights[2] * ema_pred)

        return ensemble
