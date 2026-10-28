"""Command line entry point for the shared legacy TFT workflow."""
import argparse
import os
import pickle as pkl

from timeseries.training import run_legacy_tft as main
from .configuration import dictConfiguration
from .support import load_paths, get_model


if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Transformer Model")
    parser.add_argument("-cl", "--cluster", action="store_true", help="Flag for scripts running on cluster or not")
    parser.add_argument("-t", "--train", action="store_true", help="Flag for scripts running on cluster or not")
    parser.add_argument("-m", "--model_name", type=str, default='default0', help='String to set the name of the model, must be unique in works/ to avoid overwriting')
    args = parser.parse_args()
    if args.train:
        dictConf = dictConfiguration(args.model_name)
        print(dictConf)
        main(args.cluster, args.train, dictConf)

    else:
        path_data, path_folder_works = load_paths(args.cluster)

        path_work_model = path_folder_works + args.model_name + '/'
        #ex: '/home/andrea/timeseries/anmartinelli/works/Mix_Prec/'
        if not os.path.exists(path_work_model):
            print(f'Wrong model name, check it carefully')
        # path for MODEL FILES (ADD ONLY .PKL, .PT, .PNG)
        saving_path_model_name = path_work_model + args.model_name
        
        with open(saving_path_model_name + '.pkl', 'rb') as f: #! PATH for PKL
            dict, _  = pkl.load(f)
            f.close()
        main(args.cluster, args.train, dict)
        
