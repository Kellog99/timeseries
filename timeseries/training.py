"""Shared training orchestration for forecasting, GNN and legacy TFT models."""
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.utils.data import DataLoader

from .workflows.adapters import TrainingAdapter, get_adapter, move_to_device

if TYPE_CHECKING:
    from .workflows.forecasting import TimeSeries


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    *,
    epochs: int,
    device: str | torch.device = 'cpu',
    checkpoint_dir: str | Path | None = None,
    gradient_clip_val: float | None = None,
    gradient_clip_algorithm: str = 'norm',
    adapter: TrainingAdapter | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler=None,
    scheduler_interval: int = 1,
) -> list[dict[str, float]]:
    """Run one optimization/validation loop, with sample-weighted epoch metrics.

    Base models configure their optimizer automatically. Other modules can use an
    explicit optimizer and adapter; registered GNN and legacy TFT adapters are
    selected automatically. Parameter-free models need no optimizer.
    """
    if epochs <= 0:
        raise ValueError('epochs must be positive.')
    if scheduler_interval <= 0:
        raise ValueError('scheduler_interval must be positive.')
    if gradient_clip_algorithm not in {'norm', 'value'}:
        raise ValueError('gradient_clip_algorithm must be norm or value.')
    if gradient_clip_val is not None and gradient_clip_val <= 0:
        raise ValueError('gradient_clip_val must be positive.')
    adapter = adapter if adapter is not None else get_adapter(model)
    model.to(device)
    # Legacy architectures also use a device attribute inside their forward pass.
    for module in model.modules():
        if not isinstance(getattr(type(module), 'device', None), property) and hasattr(module, 'device'):
            module.device = str(device)
    if optimizer is None:
        configure = getattr(model, 'configure_optimizers', None)
        configured = configure() if configure is not None else None
        optimizer, configured_scheduler = configured if isinstance(configured, tuple) else (configured, None)
        if scheduler is None:
            scheduler = configured_scheduler
    if optimizer is None and any(p.requires_grad for p in model.parameters()):
        raise ValueError('Trainable models require an optimizer or configure_optimizers().')
    directory = Path(checkpoint_dir) if checkpoint_dir is not None else None
    if directory is not None:
        directory.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float]] = []
    best_loss = float('inf')

    def run_epoch(loader: DataLoader, training: bool, epoch: int) -> dict[str, float]:
        model.train(training)
        totals: dict[str, float] = {}
        observations = 0
        with torch.set_grad_enabled(training and optimizer is not None):
            for index, data in enumerate(loader):
                batch = move_to_device(data, device)
                if training and optimizer is not None:
                    optimizer.zero_grad(set_to_none=True)
                result = adapter.step(model, batch, index, training=training, epoch=epoch)
                loss = result.loss
                metrics = {'loss': loss, **result.metrics}
                if any(value.ndim != 0 or not torch.isfinite(value) for value in metrics.values()):
                    raise ValueError('Training and validation steps must return finite scalar metrics.')
                if result.observations <= 0:
                    raise ValueError('A training step must contain observations.')
                if training and optimizer is not None:
                    loss.backward()
                    if gradient_clip_val is not None:
                        clip = torch.nn.utils.clip_grad_norm_ if gradient_clip_algorithm == 'norm' else torch.nn.utils.clip_grad_value_
                        clip(model.parameters(), gradient_clip_val)
                    optimizer.step()
                for name, value in metrics.items():
                    totals[name] = totals.get(name, 0.) + value.detach().item() * result.observations
                observations += result.observations
        if not observations:
            raise ValueError('Training and validation loaders must contain observations.')
        return {name: value / observations for name, value in totals.items()}

    for epoch in range(epochs):
        model.epoch = epoch
        losses = {f'train_{key}': value for key, value in run_epoch(train_loader, True, epoch).items()}
        losses.update({f'val_{key}': value for key, value in run_epoch(validation_loader, False, epoch).items()})
        history.append(losses)
        if scheduler is not None and (epoch + 1) % scheduler_interval == 0:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(losses['val_loss'])
            else:
                scheduler.step()
        model.epoch = epoch + 1
        if directory is not None:
            checkpoint = {'state_dict': model.state_dict(), 'epoch': model.epoch,
                          'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
                          'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None}
            torch.save(checkpoint, directory / 'last.pt')
            if losses['val_loss'] < best_loss:
                best_loss = losses['val_loss']
                torch.save(checkpoint, directory / 'best.pt')
    model.eval()
    return history


def train_timeseries(
    series: 'TimeSeries', dirpath: str, split_params: dict, *,
    batch_size: int = 100, num_workers: int = 0, max_epochs: int = 500,
    device: str | torch.device | None = None, gradient_clip_val: float | None = None,
    gradient_clip_algorithm: str = 'norm', modifier: str | None = None,
    modifier_params: dict | None = None,
) -> float:
    """Prepare splits/modifiers and train a forecasting workflow."""
    import os
    import pandas as pd
    from .data_management.modifiers import ModifierVVA

    MODIFIERS = {'ModifierVVA': ModifierVVA}
    series.split_params = split_params
    series.check_custom = False
    series.is_trained = False
    series.scalers_fitted = False
    train, validation, _ = series.split_for_train(**split_params)
    if modifier is not None:
        modifier_type = MODIFIERS.get(modifier)
        if modifier_type is None:
            raise ValueError(f"Unknown data modifier: {modifier!r}")
        series.modifier = modifier_type(**(modifier_params or {}))
        train, validation = series.modifier.fit_transform(train=train, val=validation)
    else:
        series.modifier = None
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    validation_loader = DataLoader(validation, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    selected_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    history = train_model(
        series.model, train_loader, validation_loader, epochs=max_epochs,
        device=selected_device, checkpoint_dir=dirpath,
        gradient_clip_val=gradient_clip_val, gradient_clip_algorithm=gradient_clip_algorithm,
    )
    series.dirpath = dirpath
    series.checkpoint_file_best = os.path.join(dirpath, "best.pt")
    series.checkpoint_file_last = os.path.join(dirpath, "last.pt")
    series.losses = pd.DataFrame(history)
    series.losses.to_csv(os.path.join(dirpath, "loss.csv"), index=False)
    series.is_trained = True
    return history[-1]["val_loss"]
    

def train_gnn(model, dataloader_train, dataloader_validation, optimizer, scheduler, config):
    """Train a GNN and export artifacts consumed by the existing plotting tools."""
    import pickle
    from .workflows.adapters import GNNAdapter
    from .workflows.gnn.support import loss_from_config

    directory = Path(config['paths']['net_weights_train'])
    model_id = config['model']['id_model']
    history = train_model(
        model, dataloader_train, dataloader_validation,
        epochs=config.getint('optimizer', 'epochs'), device=model.device,
        checkpoint_dir=directory / f'GNN_{model_id}', optimizer=optimizer,
        scheduler=scheduler, scheduler_interval=100,
        adapter=GNNAdapter(loss_from_config(config['optimizer']['loss'])),
    )
    checkpoint = torch.load(directory / f'GNN_{model_id}' / 'best.pt', map_location='cpu', weights_only=True)
    torch.save(checkpoint['state_dict'], directory / f'GNN_{model_id}.pt')
    losses = {
        split: {metric: [row[f'{prefix}_{metric}'] for row in history]
                for metric in ('loss', 'rec', 'energy', 'frobenius')}
        for split, prefix in [('train', 'train'), ('validation', 'val')]
    }
    loss_dir = Path(config['paths']['net_weights'])
    loss_dir.mkdir(parents=True, exist_ok=True)
    with (loss_dir / f'loss_{model_id}.pkl').open('wb') as file:
        pickle.dump(losses, file)
    return history


def train_legacy_tft(model, epochs, dl_training, dl_validation, cost_func, optimizer, scheduler):
    """Train legacy TFT using the common loop and retain its report file format."""
    import pickle
    from .workflows.adapters import LegacyTFTAdapter

    if model.path_model_save is None:
        raise ValueError('A checkpoint path is required for training.')
    path = Path(model.path_model_save)
    history = train_model(
        model, dl_training, dl_validation, epochs=epochs, device=model.device,
        checkpoint_dir=path.parent / f'{path.name}_checkpoints', optimizer=optimizer,
        scheduler=scheduler, adapter=LegacyTFTAdapter(cost_func),
    )
    for kind in ('best', 'last'):
        checkpoint = torch.load(path.parent / f'{path.name}_checkpoints' / f'{kind}.pt',
                                map_location='cpu', weights_only=True)
        torch.save(checkpoint['state_dict'], f'{path}_{kind}.pt')
    config = None
    if Path(f'{path}.pkl').exists():
        with open(f'{path}.pkl', 'rb') as file:
            config, _ = pickle.load(file)
    losses = {'train': [row['train_loss'] for row in history], 'val': [row['val_loss'] for row in history]}
    with open(f'{path}.pkl', 'wb') as file:
        pickle.dump([config, losses], file)
    return history


def run_gnn(config) -> None:
    
    import os
    import pickle
    from .models.gnn.model import GNN
    from .workflows.gnn.inference import get_plot

    print(f'{" Upload the dataframe ":=^60s}')
    batch_size = config.getint('dataset', 'batch_size')
    path_dataset = os.path.join(config['paths']['dataset'], f"dataframes.pkl")    
    path_emb = os.path.join(config['paths']['dataset'], f"embedding_setting.pkl")    
    
    
    with open(path_dataset, 'rb') as f :
        ds = pickle.load(f)

    with open(path_emb, 'rb') as f :
        emb = pickle.load(f)
        
    dl = {}  
    for key in ds.keys():
        dl[key] = DataLoader(ds[key], 
                             batch_size = batch_size, 
                             shuffle = key == 'train')
        
        config['dataset'][f'n_obs_{key}']=f"{len(ds[key])}"
    
    print(f'{" Upload complete ":=^60s}')


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    params_model = {key: config.getint('model', key) for key in (
        "in_feat", "nfeat_out_gnn", "num_layer1", "num_layer2",
        "hid_out_features1", "hid_out_features2",
    )}

    id_model = config['model']['id_model']
    
    print(f'{f" Configuration (hid_out_features1, hid_out_features2, num_layer1, num_layer2, nfeat_out_gnn, lr, weight_decay) = ({id_model})":=^60s}')
    model = GNN(in_feat = params_model["in_feat"], 
                past_steps=200,
                future_steps=65,
                future_channel=0,
                embs=[cardinality for cardinality, _ in emb.values()],
                cat_emb_dim=config.getint('model', 'cat_emb_dim', fallback=1),
                hidden_GNN=params_model["nfeat_out_gnn"],
                nfeat_out_gnn = params_model["nfeat_out_gnn"],
                num_layer1=params_model["num_layer1"],
                num_layer2=params_model["num_layer2"],
                hid_out_features1 = params_model["hid_out_features1"],
                hid_out_features2 = params_model["hid_out_features2"],
                emb = emb,
                past = 200, 
                future = 65, 
                device = str(device))
    
    model.to(model.device)
    optimizer = torch.optim.Adam(model.parameters(),    
                             lr = config.getfloat('optimizer','lr'), 
                             weight_decay = config.getfloat('optimizer','weight_decay'))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 
                                                       gamma = config.getfloat('optimizer','gamma'))
    train_gnn(model = model, 
            dataloader_train = dl['train'], 
            dataloader_validation = dl['validation'],
            optimizer = optimizer,
            scheduler = scheduler, 
            config = config)

    print(f'{" Creating the plot for the LAG ":=^60s}')

    path = os.path.join(config['paths']['net_weights_train'], f"GNN_{config['model']['id_model']}.pt")
    model.load_state_dict(torch.load(path))
    
    get_plot(model = model, 
             config = config,
             ds = ds)    



def run_legacy_tft(cluster: bool, run_train: bool, conf_params):
    import os
    import pickle as pkl
    from sklearn.preprocessing import StandardScaler
    from .data_management.tft import dataloading
    from .workflows.tft.support import load_paths, get_model

    # main used for training and inference
    # todo: only inference_script.py for only inference!
    
    #* Set path to get data and name to save the model 
    path_data, path_folder_works = load_paths(cluster)
    path_work_model = path_folder_works + conf_params.name + '/'
    #ex: '/home/andrea/timeseries/anmartinelli/works/Mix_Prec/'
    if not os.path.exists(path_work_model):
        os.makedirs(path_work_model)
    # path for MODEL FILES (ADD ONLY .PKL, .PT, .PNG)
    saving_path_model_name = path_work_model + conf_params.name

    #* SELECT MODEL
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = get_model(conf_params, device, saving_path_model_name)
    print(f'MODEL TRAINABLE PARAMETERS: < {model.count_parameters()} >')
    
    #* LOAD DATA
    scaler_type = StandardScaler()
    train_dl, val_dl, test_dl, scaler_y = dataloading(batch_size=conf_params.train_dict['bs'], 
                                                      batch_size_test = conf_params.test_dict['bs_t'], 
                                                      seq_len = conf_params.model_dict['seq_len'],
                                                      lag = conf_params.model_dict['lag'],
                                                      hour_learning = conf_params.train_dict['hour'], 
                                                      hour_inference = conf_params.test_dict['hour_test'], 
                                                      train_bool = run_train,
                                                      scaler_y = scaler_type,
                                                      step = 1,
                                                      path = path_data)
    
    #* Get training tools
    optimizer = conf_params.get_optim(model)
    loss_fun = conf_params.get_loss_fun()
    scheduler = conf_params.get_scheduler(optimizer)
    
    if run_train:
        with open(saving_path_model_name + '.pkl', 'wb') as f: #! PATH for PKL
            pkl.dump([conf_params, 'no_trained_yet'], f)
            f.close()
        #* TRAINING
        train_legacy_tft(model, epochs = conf_params.train_dict['epochs'], 
                    dl_training = train_dl, 
                    dl_validation = val_dl, 
                    cost_func = loss_fun, 
                    optimizer= optimizer, 
                    scheduler = scheduler)
    
    #* LEARNING
    model.inference(dl = test_dl, 
                    scaler = scaler_y)
    
