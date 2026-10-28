"""Behavioral checks for the shared PyTorch forecasting path."""
from typing import Any
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from timeseries.models import ModelFactory
from timeseries.training import train_model


def make_linear_model(**overrides):
    params: dict[str, Any] = dict(past_steps=8, future_steps=2, past_channels=1, future_channels=0,
                  embs=[], cat_emb_dim=2, kernel_size=3, sum_emb=False,
                  out_channels=1, hidden_size=16, dropout_rate=0.)
    return ModelFactory.create("linearts", **(params | overrides))


class Samples(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def sample():
    return {"x_num_past": torch.randn(8, 1), "x_num_future": torch.empty(2, 0),
            "x_cat_past": torch.empty(8, 0), "x_cat_future": torch.empty(2, 0),
            "idx_target": torch.tensor([0]), "y": torch.randn(2, 1)}


def test_training_updates_weights_and_checkpoints_round_trip(tmp_path):
    model = make_linear_model(optim_config={"lr": 0.01}, scheduler_config={"step_size": 1, "gamma": 0.5})
    loader = DataLoader(Samples([sample() for _ in range(5)]), batch_size=2)
    initial = {name: value.clone() for name, value in model.state_dict().items()}
    history = train_model(model, loader, loader, epochs=2, checkpoint_dir=tmp_path,
                          gradient_clip_val=1.0)
    assert len(history) == 2
    assert all(torch.isfinite(torch.tensor(list(epoch.values()))).all() for epoch in history)
    assert any(not torch.equal(value, initial[name]) for name, value in model.state_dict().items())
    checkpoint = torch.load(tmp_path / "last.pt", weights_only=True)
    assert checkpoint["epoch"] == 2
    assert checkpoint["optimizer_state_dict"]["param_groups"][0]["lr"] == pytest.approx(0.0025)
    restored = make_linear_model()
    restored.load_state_dict(checkpoint["state_dict"])
    restored.eval()
    batch = next(iter(loader))
    torch.testing.assert_close(model.inference(batch), restored.inference(batch))
    assert (tmp_path / "best.pt").is_file()


def test_persistence_needs_no_artificial_trainable_parameters():
    model = ModelFactory.create("persistent", past_steps=8, future_steps=2)
    assert not list(model.parameters())
    assert model.configure_optimizers() is None
    batch = next(iter(DataLoader(Samples([sample(), sample()]), batch_size=2)))
    expected = batch["x_num_past"][:, -1:, :].repeat(1, 2, 1).unsqueeze(-1)
    torch.testing.assert_close(model(batch), expected)


def test_smape_zero_targets_have_finite_loss_and_gradients():
    model = make_linear_model(loss_type="smape")
    prediction = torch.zeros(2, 2, 1, 1, requires_grad=True)
    loss = model.compute_loss({"y": torch.zeros(2, 2, 1)}, prediction)
    loss.backward()
    assert loss.item() == 0
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
