# ==================== ARIMA-STYLE FORECASTING ====================
import numpy as np


class ARIMAPredictor:
    """Simple ARIMA-style predictor"""

    def __init__(self, p=5, d=1, q=2):
        self.p = p  # AR order
        self.d = d  # Differencing
        self.q = q  # MA order
        self.ar_coef = None
        self.ma_coef = None

    def difference(self, data, order=1):
        """Apply differencing"""
        diff = data.copy()
        for _ in range(order):
            diff = np.diff(diff)
        return diff

    def fit(self, data):
        """Fit AR coefficients using least squares"""
        diff_data = self.difference(data, self.d)

        # Create lagged features for AR
        X, y = [], []
        for i in range(self.p, len(diff_data)):
            X.append(diff_data[i - self.p:i])
            y.append(diff_data[i])

        X = np.array(X)
        y = np.array(y)

        # Least squares
        self.ar_coef = np.linalg.lstsq(X, y, rcond=None)[0]

    def predict(self, data, steps=1):
        """Predict next steps"""
        diff_data = self.difference(data, self.d)
        predictions = []

        current = diff_data[-self.p:].tolist()

        for _ in range(steps):
            pred = np.dot(current[-self.p:], self.ar_coef)
            predictions.append(pred)
            current.append(pred)

        # Inverse differencing
        last_val = data[-1]
        for i, pred in enumerate(predictions):
            predictions[i] = last_val + pred
            last_val = predictions[i]

        return np.array(predictions)
