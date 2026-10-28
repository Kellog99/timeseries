"""Regression tests for the unified training paths and data/workflow boundary."""
import pickle
from configparser import ConfigParser

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from timeseries.data_management import TimeSeries as TimeSeriesData
from timeseries.models import ModelFactory
from timeseries.models.gnn.model import GNN
from timeseries.models.legacy_tft.model import Model
from timeseries.training import train_gnn, train_legacy_tft, train_model
from timeseries.workflows import TimeSeries
from timeseries.workflows.gnn.support import loss_from_config


def test_data_splits_and_workflow_checkpoint_round_trip(tmp_path):
    data = TimeSeriesData('data')
    assert not hasattr(data, 'train_model')
    data.generate_signal(length=150, noise_mean=0)
    assert all(len(split) for split in data.split_for_train(past_steps=8, future_steps=2))

    series = TimeSeries('workflow')
    series.generate_signal(length=150, noise_mean=0)
    config = dict(past_steps=8, future_steps=2)
    model = ModelFactory.create('persistent', **config)
    series.set_model(model, {'model_configs': config, 'optim_config': {}, 'scheduler_config': None})
    loss = series.train_model(str(tmp_path), {'past_steps': 8, 'future_steps': 2}, max_epochs=1, batch_size=7)
    assert np.isfinite(loss)
    assert (tmp_path / 'loss.csv').is_file()
    fitted_mean = series.scaler_num['signal'].mean_.copy()
    predictions = series.inference_on_set(batch_size=3, num_workers=0)
    np.testing.assert_array_equal(fitted_mean, series.scaler_num['signal'].mean_)
    series.save(str(tmp_path / 'workflow'))
    restored = TimeSeries('restored')
    restored.load(type(model), str(tmp_path / 'workflow'))
    assert restored.inference_on_set(batch_size=3, num_workers=0).equals(predictions)


def test_gnn_training_saves_first_epoch_and_component_metrics(tmp_path):
    model = GNN(past_steps=200, future_steps=65, future_channel=0, embs=[4],
                cat_emb_dim=2, hidden_GNN=4, in_feat=4, past=200, future=65,
                emb={'category': (4, 2)}, device='cpu', nfeat_out_gnn=4,
                hid_out_features1=4, hid_out_features2=4)
    inputs = torch.randn(2, 265, 4)
    inputs[:, :, 0] = torch.randint(0, 4, (2, 265))
    loader = DataLoader(TensorDataset(inputs, torch.randn(2, 65)), batch_size=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)
    config = ConfigParser()
    config.read_dict({'paths': {'net_weights_train': str(tmp_path), 'net_weights': str(tmp_path)},
                      'model': {'id_model': 'test'}, 'optimizer': {'epochs': '1', 'loss': 'nn.MSELoss()'}})
    initial = model.gnn1.lin.weight.detach().clone()
    history = train_gnn(model, loader, loader, optimizer, scheduler, config)
    assert not torch.equal(initial, model.gnn1.lin.weight)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert scheduler.last_epoch == 0  # Historical GNN cadence is 100 epochs.
    assert history[0]['val_loss'] == pytest.approx(sum(history[0][f'val_{key}'] for key in ('rec', 'energy', 'frobenius')))
    restored = torch.load(tmp_path / 'GNN_test.pt', weights_only=True)
    model.load_state_dict(restored)
    with (tmp_path / 'loss_test.pkl').open('rb') as file:
        assert len(pickle.load(file)['validation']['loss']) == 1


def test_legacy_tft_training_and_single_observation_validation(tmp_path):
    model = Model(use_target_past=True, use_yprec=True, n_cat_var=5, n_target_var=1,
                  seq_len=8, lag=2, d_model=4, n_enc=1, n_dec=1, head_size=2,
                  num_heads=2, fw_exp=1, dropout=0., num_lstm_layers=1, device='cpu',
                  path_model_save=str(tmp_path / 'legacy'))
    categorical = torch.ones(3, 8, 6, dtype=torch.long)
    target = torch.randn(3, 8)
    loader = DataLoader(TensorDataset(categorical, target), batch_size=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    initial = model.outLinear.weight.detach().clone()
    history = train_legacy_tft(model, 4, loader, loader, nn.L1Loss(), optimizer, scheduler)
    assert len(history) == 4
    assert not torch.equal(initial, model.outLinear.weight)
    assert optimizer.param_groups[0]['lr'] == pytest.approx(0.001 * 0.5 ** 4)
    for kind in ('best', 'last'):
        model.load_state_dict(torch.load(tmp_path / f'legacy_{kind}.pt', weights_only=True))
    with (tmp_path / 'legacy.pkl').open('rb') as file:
        assert len(pickle.load(file)[1]['val']) == 4


def test_unknown_module_requires_explicit_batch_contract():
    with pytest.raises(TypeError, match='TrainingAdapter'):
        train_model(nn.Linear(1, 1), [], [], epochs=1)


def test_gnn_loss_config_does_not_execute_code():
    with pytest.raises(ValueError, match='Unsupported GNN loss'):
        loss_from_config("__import__('os').system('false')")


def test_validation_weights_partial_batches_by_sample_count():
    model = ModelFactory.create('persistent', past_steps=2, future_steps=1, loss_type='mse')
    samples = [{'x_num_past': torch.zeros(2, 1), 'idx_target': torch.tensor([0]),
                'y': torch.tensor([[value]])} for value in (1., 2., 3.)]
    loader = DataLoader(samples, batch_size=2)
    history = train_model(model, loader, loader, epochs=1)
    assert history[0]['val_loss'] == pytest.approx(14 / 3)
    assert not model.training


def test_training_rejects_empty_validation_loader():
    model = ModelFactory.create('persistent', past_steps=2, future_steps=1)
    loader = DataLoader([{'x_num_past': torch.zeros(2, 1), 'idx_target': torch.tensor([0]),
                          'y': torch.zeros(1, 1)}])
    with pytest.raises(ValueError, match='contain observations'):
        train_model(model, loader, DataLoader([]), epochs=1)
