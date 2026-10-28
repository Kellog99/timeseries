"""Configuration helpers for the GNN command line interface."""
import torch.nn as nn


def loss_from_config(
        expression: str
        ) -> nn.Module:
    """Resolve historical loss names without evaluating configuration as code."""
    name = expression.strip().removesuffix('()').removeprefix('torch.nn.').removeprefix('nn.')
    losses = {
        'MSELoss': nn.MSELoss, 'L1Loss': nn.L1Loss,
        'SmoothL1Loss': nn.SmoothL1Loss, 'HuberLoss': nn.HuberLoss
    }
    if name not in losses:
        raise ValueError(f'Unsupported GNN loss: {expression!r}')
    return losses[name]()
