# ==================== KALMAN FILTER ====================
class KalmanFilter:
    """1D Kalman Filter for time series prediction"""

    def __init__(self, process_variance=1e-5, measurement_variance=1e-1):
        self.process_variance = process_variance  # Q
        self.measurement_variance = measurement_variance  # R
        self.estimate = 0.0
        self.error_estimate = 1.0

    def update(self, measurement):
        # Prediction
        prediction = self.estimate
        error_prediction = self.error_estimate + self.process_variance

        # Update
        kalman_gain = error_prediction / (error_prediction + self.measurement_variance)
        self.estimate = prediction + kalman_gain * (measurement - prediction)
        self.error_estimate = (1 - kalman_gain) * error_prediction

        return self.estimate

    def filter_series(self, data):
        """Apply Kalman filter to entire series"""
        filtered = []
        for val in data:
            filtered.append(self.update(val))
        return np.array(filtered)