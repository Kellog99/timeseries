"""Common state and loss computation for PyTorch forecasting models."""
from timeseries.utils.typing import Module
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import StepLR

from .configuration import BaseLoss
from .utils.torch import QuantileLossMO, SinkhornDistance

Batch = dict[str, Tensor]
OptimizerSetup = Optimizer | tuple[Optimizer, StepLR] | None


class Base(Module, ABC):
    _device_anchor: Tensor

    def __init__(
            self,
            *,
            past_steps: int,
            future_steps: int,
            past_channels: int = 0,
            future_channels: int = 0,
            out_channels: int = 1,
            quantiles: list[float] | None = None,
            loss_type: BaseLoss | None = "l1",
            persistence_weight: float = 0.0,
            n_classes: int = 0,
            optim: str | None = None,
            optim_config: dict[str, Any] | None = None,
            scheduler_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.past_steps = past_steps
        self.future_steps = future_steps
        self.past_channels = past_channels
        self.future_channels = future_channels
        self.out_channels = out_channels
        self.quantiles = list(quantiles or [])
        self.loss_type: BaseLoss = loss_type or "l1"
        self.persistence_weight = persistence_weight
        self.optim = optim
        self.optim_config = dict(optim_config or {})
        self.scheduler_config = dict(scheduler_config) if scheduler_config is not None else None
        self.epoch = 0
        self.n_classes = n_classes
        self.is_classification = n_classes > 0
        self.use_quantiles = bool(self.quantiles) and not self.is_classification
        self.mul = n_classes if self.is_classification else (len(self.quantiles) or 1)
        if self.is_classification:
            self.loss = nn.CrossEntropyLoss()
        elif self.use_quantiles:
            self.loss = QuantileLossMO(self.quantiles)
        else:
            match self.loss_type:
                case "mse":
                    self.loss = nn.MSELoss()
                case "l1" | "smape" | "sinkhorn" | "high_order" | "linear_penalization" | "exponential_penalization" | "kl":
                    # Architecture-specific training steps handle KL divergence.
                    self.loss = nn.L1Loss()
                case _:
                    raise ValueError(f"Unsupported loss type: {self.loss_type!r}")
        # Parameter-free baselines must also follow model.to(device).
        self.register_buffer("_device_anchor", torch.empty(0), persistent=False)

    @property
    def device(self) -> torch.device:
        return next(self.parameters(), self._device_anchor).device

    @abstractmethod
    def forward(self, batch: Batch) -> Tensor | tuple[Tensor, ...]:
        """Compute predictions or architecture-specific training outputs."""
        raise NotImplementedError

    def inference(self, batch: Batch) -> Tensor:
        return self(batch)

    def configure_optimizers(self) -> OptimizerSetup:
        return self._configure_optimizer(self.parameters())

    def _configure_optimizer(self, parameters: Iterable[nn.Parameter]) -> OptimizerSetup:
        trainable = [parameter for parameter in parameters if parameter.requires_grad]
        if not trainable:
            return None
        name = (self.optim or "Adam").removeprefix("torch.optim.").removeprefix("optim.")
        optimizer_type = getattr(torch.optim, name, None)
        if not isinstance(optimizer_type, type) or not issubclass(optimizer_type, Optimizer):
            raise ValueError(f"Unknown torch optimizer: {self.optim!r}")
        optimizer = optimizer_type(trainable, **({"lr": 5e-5} | self.optim_config))
        if self.scheduler_config is not None:
            return optimizer, StepLR(optimizer, **self.scheduler_config)
        return optimizer

    def training_step(self, batch: Batch, batch_idx: int = 0) -> Tensor:
        return self.compute_loss(batch, self(batch))

    def validation_step(self, batch: Batch, batch_idx: int = 0) -> Tensor:
        return self.compute_loss(batch, self(batch))

    def compute_loss(self, batch: Batch, prediction: Tensor) -> Tensor:
        target = batch["y"]
        if self.is_classification:
            return self.loss(prediction.reshape(-1, self.n_classes), target.reshape(-1).long())
        point = prediction[..., 1] if self.use_quantiles else prediction[..., 0]
        initial_loss = self.loss(prediction if self.use_quantiles else point, target)
        match self.loss_type:
            case "l1" | "mse":
                return initial_loss
            case "smape":
                denominator = (point.abs() + target.abs()).clamp_min(torch.finfo(point.dtype).eps)
                return (2 * (point - target).abs() / denominator).mean()
            case "sinkhorn":
                return SinkhornDistance(eps=0.1, max_iter=100, reduction="mean").compute(point, target)
            case "high_order":
                real_std = (target.var(dim=(0, 1), correction=0) + 1e-8).sqrt()
                predicted_std = (point.var(dim=(0, 1), correction=0) + 1e-8).sqrt()
                return initial_loss + self.persistence_weight * (real_std - predicted_std).abs().mean()
            case "linear_penalization" | "exponential_penalization":
                indices = batch["idx_target"][0]
                persistence = batch["x_num_past"][:, -1, indices].unsqueeze(1)
                match self.loss_type:
                    case "linear_penalization":
                        deviation = ((persistence - point) / (0.001 + persistence.abs())).abs().clamp(0, 0.1)
                        weights = self.persistence_weight * (2 - 10 * deviation)
                    case "exponential_penalization":
                        weights = 1 + self.persistence_weight * torch.exp(-(persistence - point).abs())
                return ((point - target).abs() * weights).mean()
            case _:
                raise ValueError(f"Loss {self.loss_type!r} requires an architecture-specific compute_loss().")
