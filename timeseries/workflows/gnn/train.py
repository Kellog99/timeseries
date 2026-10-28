"""Compatibility entry point; the training loop lives in timeseries.training."""
from timeseries.training import train_gnn as training

__all__ = ["training"]
