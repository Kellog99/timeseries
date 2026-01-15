# ==================== GARCH VOLATILITY ====================
class SimpleGARCH:
    """Simplified GARCH(1,1) for volatility forecasting"""

    def __init__(self, omega=0.01, alpha=0.05, beta=0.9):
        self.omega = omega
        self.alpha = alpha
        self.beta = beta

    def fit_predict(self, returns, steps=1):
        """Forecast volatility"""
        # Calculate initial variance
        var = np.var(returns)

        # Update variance based on recent returns
        for ret in returns[-100:]:
            var = self.omega + self.alpha * ret ** 2 + self.beta * var

        # Forecast
        forecasts = []
        for _ in range(steps):
            var = self.omega + (self.alpha + self.beta) * var
            forecasts.append(np.sqrt(var))

        return np.array(forecasts)