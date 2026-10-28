import os
import argparse
from configparser import ConfigParser

from .test import test
from timeseries.training import run_gnn as train


if __name__ == "__main__":
    # all standard stuffs here
    parser = argparse.ArgumentParser(description="Base model")
    parser.add_argument("-c", "--config", type=str, required = True, help="Config file")

    # parametri extra
    parser.add_argument("-lr", "--lr", type=str, help = "learning rate of the optimization step")
    parser.add_argument("-nl1", "--num_layer1", type=str, help = "number of layer of GNN")
    parser.add_argument("-nl2", "--num_layer2", type=str, help = "number of layer of GNN")
    parser.add_argument("-hl1", "--hid_out_features1", type=str, help = "hidden layer firts GNN")
    parser.add_argument("-hl2", "--hid_out_features2", type=str, help = "hidden layer second GNN")
    parser.add_argument("-hfg", "--nfeat_out_gnn", type=str, help = "hidden layer for the output of the GNN")
    parser.add_argument("-wd", "--weight_decay", type=str, help = "weight decay")

    # non posso avere più di un'operazione alla volta
    parser.add_argument("-d", "--dataset", action="store_true", help = "data manipulation")
    parser.add_argument("-t", "--train", action="store_true", help="Train model flag")
    parser.add_argument("-i", "--inference", action="store_true", help="Inference model flag")
    parser.add_argument("-e", "--test", action="store_true", help="Inference model flag")

    parser.add_argument("-p", "--get_plot", action="store_true", help="Train model flag")
    parser.add_argument("-m", "--name_model", type=str, help="Train model flag")
    

    args = parser.parse_args()
    config_path = args.config
    config = ConfigParser()
    config.read(config_path)
    
    print(f'{" Cheking the paths ":=^60s}')
    if 'paths' in config.keys():
        for path in config['paths']:
            if os.path.exists(config['paths'][path]) != True:
                os.makedirs(config['paths'][path])

    print(f'{" Paths controlled ":=^60s}')
    
    print(f'{" Setting new possible parameters for the model ":=^60s}')
    config['model']['id_model'] = ""
    for i, key in enumerate(["hid_out_features1", "hid_out_features2", "num_layer1", "num_layer2", "nfeat_out_gnn", "lr", "weight_decay"]):
        exists = True
        try:
            getattr(args, key)
        except:
            exists = False
        if exists:
            if getattr(args, key) is not None:
                if key not in ["lr", "weight_decay"]:
                    config['model'][key] = f"{getattr(args, key)}"
                else:
                    config['optimizer'][key] = f"{getattr(args, key)}"

        if i ==0:
            config['model']['id_model'] = f"{config['model'][key]}" 
        else:
            if key not in ["lr", "weight_decay"]:
                config['model']['id_model'] = f"{config['model']['id_model']}_{config['model'][key]}" 
            else:
                config['model']['id_model'] = f"{config['model']['id_model']}_{config['optimizer'][key]}" 
    
    
    print(f'{" Setting complete ":=^60s}')
    if args.get_plot:
        models = os.listdir(config['paths']['net_weights_train'])
        for n, model in enumerate(models):
            if model.startswith('GNN_') and model.endswith('.pt'):
                print(model[4:])
                test(config = config, 
                     name_model = model)
    if args.dataset:
        from timeseries.data_management.data_generation import crete_dataframe
        crete_dataframe(config_path)
    if args.train:
        train(config)
