"""Model lifecycle and inference helpers for prepared time series."""
import logging
import os
import pickle
from typing import Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from timeseries.base import Base
from timeseries.data_management import TimeSeries as TimeSeriesData
from timeseries.data_management.modifiers import VVADataset
from timeseries.data_management.utils import MyDataset


class TimeSeries(TimeSeriesData):
    """Time series with model attachment, training, inference and persistence."""

    checkpoint_file_best: str
    checkpoint_file_last: str
    losses: pd.DataFrame

    def __init__(
            self,
            name: str
    ):
        super().__init__(name)
        self.is_trained = False
        self.modifier = None
        self.check_custom = False
        self.split_params = {}
        self.dirpath: str | None = None

    def set_model(
            self,
            model: Base,
            config: dict | None = None
    ):
        """Set the model to train

        Args:
            model (Base): see `models`
            config (dict, optional): usually the configuration used by the model. Defaults to None.
        """
        self.model = model
        logging.info(
            '######################################################################################################'
        )
        logging.info(
            '###########LOOK THE INIT PROCEDURE IF YOU NEED A CUSTOM INITIALIZATION of the weighjts################'
        )
        logging.info(
            '######################################################################################################'
        )
        # self.model.apply(weight_init)
        self.config = config

        logging.info(
            '######################################################################################################'
        )
        logging.info(
            '######################################################MODEL###########################################'
        )
        logging.info(
            '######################################################################################################'
        )
        logging.info(model)

    def train_model(
            self,
            dirpath: str,
            split_params: dict,
            **kwargs
    ) -> float:
        """Delegate data preparation and training to the shared training module."""
        from timeseries.training import train_timeseries

        return train_timeseries(self, dirpath, split_params, **kwargs)

    def inference_on_set(
            self,
            batch_size: int = 100,
            num_workers: int = 4,
            split_params: Union[None, dict] = None,
            set: str = 'test',
            rescaling: bool = True,
            data: Union[None, torch.utils.data.Dataset] = None
    ) -> pd.DataFrame:
        """This function allows to get the prediction on a particular set (train, test or validation). 

        Args:
            batch_size (int, optional): barch sise. Defaults to 100.
            num_workers (int, optional): num workers. Defaults to 4.
            split_params (Union[None,dict], optional): if not None  the spliting procedure will use the given data
            otherwise it will use the same configuration used in train. Defaults to None.
            set (str, optional): trai, validation or test. Defaults to 'test'.
            rescaling (bool, optional):  If rescaling is true the output will be rescaled to the initial values. .
            Defaults to True.
            data (None or pd.DataFrame, optional). If not None the inference is performed on the given data. In the
            case of custom data please call inference because it will normalize the data for you!
        Returns:
            pd.DataFrame: the predicted values in a pandas format
        """

        if data is None:
            if split_params is None:
                logging.info(f'splitting using train parameters {self.split_params}')
                train, validation, test = self.split_for_train(**self.split_params)
            else:
                train, validation, test = self.split_for_train(**split_params)

        if set == 'test':
            if self.modifier is not None:
                test = self.modifier.transform(test)
            dl = DataLoader(
                test, batch_size = batch_size, shuffle = False, drop_last = False, num_workers = num_workers
            )
        elif set == 'validation':
            if self.modifier is not None:
                validation = self.modifier.transform(validation)
            dl = DataLoader(
                validation, batch_size = batch_size, shuffle = False, drop_last = False, num_workers = num_workers
            )
        elif set == 'train':
            if self.modifier is not None:
                train = self.modifier.transform(train)
            dl = DataLoader(
                train, batch_size = batch_size, shuffle = False, drop_last = False, num_workers = num_workers
            )
        elif set == 'custom':
            if data is None:
                raise ValueError("Provide a dataset for custom inference.")
            if self.check_custom:
                pass
            else:
                logging.info('RUN IT AT YOUR OWN RISK, I RECCOMEND TO CALL .inference INSTEAD')
            if self.modifier is not None:
                data = self.modifier.transform(data)
            dl = DataLoader(
                data, batch_size = batch_size, shuffle = False, drop_last = False, num_workers = num_workers
            )

        else:
            raise ValueError('Select train, test, validation, or custom.')
        self.model.eval()

        res = []
        real = []

        logging.info(f'Device used: {self.model.device}')

        with torch.no_grad():
            for batch in dl:
                batch = {key: value.to(self.model.device) for key, value in batch.items()}
                res.append(self.model.inference(batch).cpu().detach().numpy())
                real.append(batch['y'].cpu().detach().numpy())

        res = np.vstack(res)

        real = np.vstack(real)
        assert isinstance(dl.dataset, (MyDataset, VVADataset))
        time = dl.dataset.t

        if self.modifier is not None:
            res, real = self.modifier.inverse_transform(res, real)

        ## BxLxCx3
        if rescaling:
            logging.info('Scaling back')
            for i, c in enumerate(self.target_variables):
                real[:, :, i] = self.scaler_num[c].inverse_transform(real[:, :, i].reshape(-1, 1)).reshape(
                    -1, real.shape[1]
                )
                for j in range(res.shape[3]):
                    res[:, :, i, j] = self.scaler_num[c].inverse_transform(res[:, :, i, j].reshape(-1, 1)).reshape(
                        -1, res.shape[1]
                    )

        if self.model.use_quantiles:
            ##i+1 
            time = pd.DataFrame(time, columns = [i + 1 for i in range(res.shape[1])]).melt().rename(
                columns = {'value': 'time', 'variable': 'lag'}
            )
            tot = [time]
            for i, c in enumerate(self.target_variables):
                tot.append(
                    pd.DataFrame(real[:, :, i], columns = [i + 1 for i in range(res.shape[1])]).melt().rename(
                        columns = {'value': c}
                    ).drop(columns = ['variable'])
                )
                tot.append(
                    pd.DataFrame(res[:, :, i, 0], columns = [i + 1 for i in range(res.shape[1])]).melt().rename(
                        columns = {'value': c + '_low'}
                    ).drop(columns = ['variable'])
                )
                tot.append(
                    pd.DataFrame(res[:, :, i, 1], columns = [i + 1 for i in range(res.shape[1])]).melt().rename(
                        columns = {'value': c + '_median'}
                    ).drop(columns = ['variable'])
                )
                tot.append(
                    pd.DataFrame(res[:, :, i, 2], columns = [i + 1 for i in range(res.shape[1])]).melt().rename(
                        columns = {'value': c + '_high'}
                    ).drop(columns = ['variable'])
                )

            res = pd.concat(tot, axis = 1)


        ## BxLxCx1
        else:
            time = pd.DataFrame(time, columns = [i + 1 for i in range(res.shape[1])]).melt().rename(
                columns = {'value': 'time', 'variable': 'lag'}
            )
            tot = [time]
            for i, c in enumerate(self.target_variables):
                tot.append(
                    pd.DataFrame(real[:, :, i], columns = [i + 1 for i in range(res.shape[1])]).melt().rename(
                        columns = {'value': c}
                    ).drop(columns = ['variable'])
                )
                tot.append(
                    pd.DataFrame(res[:, :, i, 0], columns = [i + 1 for i in range(res.shape[1])]).melt().rename(
                        columns = {'value': c + '_pred'}
                    ).drop(columns = ['variable'])
                )
            res = pd.concat(tot, axis = 1)

        return res

    def inference(
            self,
            batch_size: int = 100,
            num_workers: int = 4,
            split_params: Union[None, dict] = None,
            rescaling: bool = True,
            data: pd.DataFrame | None = None,
            steps_in_future: int = 0
    ) -> pd.DataFrame:
        """same as inference_on_set
        only change is split_params that must contain this keys but using the default can be sufficient:
        'past_steps','future_steps','shift','keep_entire_seq_while_shifting','starting_point','skip_step'
        
        skip_step is set to 1 for convenience in case you want to use previous split parameters otherwise care to be
        coohernt
        """
        if data is None:
            raise ValueError("Provide data for custom inference.")
        self.check_custom = True
        ## enlarge the dataset in order to have all the rows needed
        freq = pd.to_timedelta(np.diff(data.time).min())
        logging.info(f'#############Detected minumum frequency: {freq}#############')
        empty = pd.DataFrame(
            {
                'time': pd.date_range(
                    data.time.min(), data.time.max() + freq * (
                            steps_in_future + self.split_params['past_steps'] + self.split_params['future_steps']),
                    freq = freq
                )
            }
        )
        dataset = empty.merge(data, how = 'left')

        if split_params is None:
            split_params = {}
            for c in self.split_params.keys():
                if c in ['past_steps', 'future_steps', 'shift', 'keep_entire_seq_while_shifting', 'starting_point']:
                    split_params[c] = self.split_params[c]
            split_params['skip_step'] = 1
            inference_data = self.create_data_loader(dataset, **split_params)
        else:
            inference_data = self.create_data_loader(data, **split_params)

        res = self.inference_on_set(
            batch_size = batch_size, num_workers = num_workers, split_params = None, set = 'custom',
            rescaling = rescaling,
            data = inference_data
        )
        self.check_custom = False
        return res

    def save(
            self,
            filename: str
    ) -> None:
        """save the timeseries object

        Args:
            filename (str): name of the file
        """
        logging.info('Saving')
        with open(f'{filename}.pkl', 'wb') as f:
            params = self.__dict__.copy()
            for k in ['model']:
                if k in params.keys():
                    _ = params.pop(k)
            pickle.dump(params, f)

    def load(
            self,
            model: type[Base],
            filename: str,
            load_last: bool = True,
            dirpath: Union[str, None] = None,
            weight_path: Union[str, None] = None
    ) -> None:
        """ Load a saved model

        Args:
            model (Base): class of the model to load (it will be initiated by pytorch-lightening)
            filename (str): filename of the saved model
            load_last (bool, optional): if true the last checkpoint will be loaded otherwise the best (in the
            validation set). Defaults to True.
            dirpath (Union[str,None], optional): if None we asssume that the model is loaded from the same pc where
            it has been trained, otherwise we can pass the dirpath where all the stuff has been saved . Defaults to
            None.
            weight_path (Union[str, None], optional): if None the standard path will be used. Defaults to None.
        """

        logging.info('################Loading#################################')
        self.modifier = None
        self.check_custom = False
        self.is_trained = True
        self.scalers_fitted = True
        with open(filename + '.pkl', 'rb') as f:
            params = pickle.load(f)
            for p in params:
                setattr(self, p, params[p])
        if self.config is None:
            raise ValueError("The saved model configuration is missing.")
        self.model = model(
            **self.config['model_configs'], optim_config = self.config['optim_config'],
            scheduler_config = self.config['scheduler_config']
        )

        if weight_path is not None:
            tmp_path = weight_path
        else:
            if self.dirpath is not None:
                directory = self.dirpath
            else:
                directory = dirpath

            if directory is None:
                raise ValueError("Provide a checkpoint directory or an explicit weight_path.")

            if load_last:

                try:
                    tmp_path = os.path.join(directory, self.checkpoint_file_last.split('/')[-1])
                except:
                    logging.info('checkpoint_file_last not defined try to load best')
                    tmp_path = os.path.join(directory, self.checkpoint_file_best.split('/')[-1])
            else:
                try:
                    tmp_path = os.path.join(directory, self.checkpoint_file_best.split('/')[-1])
                except:
                    logging.info('checkpoint_file_best not defined try to load best')
                    tmp_path = os.path.join(directory, self.checkpoint_file_last.split('/')[-1])
        checkpoint = torch.load(tmp_path, map_location = self.model.device, weights_only = True)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
