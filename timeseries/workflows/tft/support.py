"""Path resolution and model construction for legacy TFT workflows."""
import yaml
from timeseries.models.legacy_tft.model import Model
from .configuration import dictConfiguration


def load_paths(cluster: bool):
    config_path = (
        '/storage/DSIP/edison/anmartinelli/config.yaml'
        if cluster else '/home/andrea/timeseries/anmartinelli/config.yaml'
    )
    with open(config_path, encoding='utf-8') as config_file:
        conf = yaml.safe_load(config_file)
    paths = conf['cl' if cluster else 'loc']
    return paths['data'], paths['folder_works']


def get_model(conf_params: dictConfiguration, device, path: str) -> Model:
    model = Model(use_target_past = conf_params.strategy_dict['use_target_past'],
                use_yprec = conf_params.strategy_dict['use_yprec'],
                n_cat_var = conf_params.model_dict['n_cat_var'],
                n_target_var = conf_params.model_dict['n_target_var'],
                seq_len = conf_params.model_dict['seq_len'],
                lag = conf_params.model_dict['lag'],
                d_model = conf_params.model_dict['d_model'],
                n_enc = conf_params.model_dict['n_enc_layers'],
                n_dec = conf_params.model_dict['n_dec_layers'],
                head_size = conf_params.model_dict['head_size'],
                num_heads = conf_params.model_dict['num_heads'],
                fw_exp = conf_params.model_dict['fw_exp'],
                dropout = conf_params.model_dict['dropout'],
                num_lstm_layers = conf_params.model_dict['num_lstm_layers'],
                device = device,
                quantiles = conf_params.strategy_dict['quantiles'],
                path_model_save = path,
                model_name = conf_params.name).to(device)
    return model
        
