"""Validated strategies, backtests and paper execution; no live orders are sent."""
from .backtest import (
    BacktestConfig,
    BacktestResult,
    MovingAverageStrategy,
    WalkForwardConfig,
    backtest,
    walk_forward_signals,
)
from .broker import Broker, ExecutionConfig, PaperBroker
from .models import Fill, Order, OrderState, Portfolio, Position
from .risk import RiskLimits, TailRisk, stress_test

__all__ = [
    'BacktestConfig',
    'BacktestResult',
    'Broker',
    'ExecutionConfig',
    'Fill',
    'MovingAverageStrategy',
    'Order',
    'OrderState',
    'PaperBroker',
    'Portfolio',
    'Position',
    'RiskLimits',
    'TailRisk',
    'WalkForwardConfig',
    'backtest',
    'stress_test',
    'walk_forward_signals',
]
