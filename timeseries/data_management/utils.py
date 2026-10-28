from enum import Enum
import pandas,numpy
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
from typing import Union
import torch
import os

def extend_df(x:Union[pandas.Series, numpy.ndarray],freq:str)-> pandas.DataFrame:
    """Utility for generating a full dataset and then merge the real data

    Args:
        x (Union[pd.Series, np.array]): array indicating the time
        freq (str): frequency (in pandas notation) of the resulting dataframe

    Returns:
        pd.DataFrame: a dataframe with the column time ranging from thr minumum of x to the maximum with frequency `freq`
    """

    empty = pd.DataFrame({'time':pd.date_range(x.min(),x.max(),freq=freq)})
    return empty


class MyDataset(Dataset):

    def __init__(self, data:dict,t:np.ndarray,idx_target:np.ndarray | list[int] | None)->None:
        """
            Extension of Dataset class. While training the returned item is a batch containing the standard keys

        Args:
            data (dict): a dictionary. Each key is a np.array containing the data. The keys are:
                y : the target variable(s)
                x_num_past: the numerical past variables
                x_num_future: the numerical future variables
                x_cat_past: the categorical past variables
                x_cat_future: the categorical future variables
                idx_target: index of target features in the past array
            t (np.array): the time array related to the target variables
            idx_target (Union[np.array,None]): you can specify the index in the past data that represent the input features (for differntial analysis or detrending strategies)

        Returns:
            torch.utils.data.Dataset: a torch Dataset to be used in a Dataloader
        """
        self.data = data
        self.t = t
        self.idx_target = np.array(idx_target) if idx_target is not None else None
    def __len__(self):
        return len(self.data['y'])

    def __getitem__(self, index):
        idxs = index
        sample = {}
        for k in self.data:
            sample[k] = self.data[k][idxs]
        if self.idx_target is not None:
            sample['idx_target'] = self.idx_target
        return sample

class ActionEnum(Enum):
    """action of categorical variable
    
    :meta private:
    """
    multiplicative = 'multiplicative'
    additive = 'additive'
