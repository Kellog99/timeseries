"""Batch and loss contracts; optimization is owned by timeseries.training."""
from dataclasses import dataclass, field
from typing import Any, Protocol

import torch
from torch import Tensor, nn

from timeseries.base import Base


@dataclass
class StepResult:
    loss: Tensor
    observations: int
    metrics: dict[str, Tensor] = field(default_factory=dict)


class TrainingAdapter(Protocol):
    def step(self, model: nn.Module, batch: Any, index: int, *, training: bool, epoch: int) -> StepResult: ...


class ForecastAdapter:
    def step(self, model: Base, batch, index, *, training, epoch):
        loss = (model.training_step if training else model.validation_step)(batch, index)
        return StepResult(loss, batch['y'].shape[0])


class GNNAdapter:
    def __init__(self, loss_function=None, *, alpha=0.2, gamma=2e-3):
        self.loss_function = loss_function if loss_function is not None else nn.MSELoss()
        self.alpha = alpha
        self.gamma = gamma

    def step(self, model, batch, index, *, training, epoch):
        prediction, distances, adjacency = model(batch[0])
        reconstruction = (1 - self.alpha) * self.loss_function(prediction, batch[1].float())
        energy = self.alpha * 0.5 * torch.matmul(distances, adjacency).sum()
        frobenius = self.gamma * torch.norm(adjacency)
        return StepResult(reconstruction + energy + frobenius, batch[1].shape[0],
                          {'rec': reconstruction, 'energy': energy, 'frobenius': frobenius})


class LegacyTFTAdapter:
    def __init__(self, loss_function=None):
        self.loss_function = loss_function if loss_function is not None else nn.L1Loss()

    def step(self, model, batch, index, *, training, epoch):
        categorical, target = batch
        forward = model.iter_forward if not training or epoch % 5 == 3 else model
        output = forward(categorical[:, :, 1:], target)
        # Remove only the output channel: batch size and horizon may both be one.
        if output.ndim == 3 and output.shape[-1] == 1:
            output = output.squeeze(-1)
        loss = self.loss_function(output, target[:, -model.lag:].float())
        return StepResult(loss, target.shape[0])


def move_to_device(batch, device):
    """Move dictionary and tuple datasets without changing integer feature types."""
    if isinstance(batch, Tensor):
        return batch.to(device)
    if isinstance(batch, dict):
        return {key: move_to_device(value, device) for key, value in batch.items()}
    if isinstance(batch, tuple):
        return tuple(move_to_device(value, device) for value in batch)
    if isinstance(batch, list):
        return [move_to_device(value, device) for value in batch]
    return batch


def get_adapter(model: nn.Module) -> TrainingAdapter:
    if isinstance(model, Base):
        return ForecastAdapter()
    from timeseries.models.gnn.model import GNN
    if isinstance(model, GNN):
        return GNNAdapter()
    from timeseries.models.legacy_tft.model import Model
    if isinstance(model, Model) or type(model).__module__ == 'timeseries.models.legacy_tft.model-checkpoint':
        return LegacyTFTAdapter()
    raise TypeError('Unsupported model: provide a TrainingAdapter for its batch and loss contract.')
