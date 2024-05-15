
from typing import Dict
from .base import Sampler
import numpy as np
import pandas as pd
from common import S
import torch
from .activelearner import ActiveLearner
from sklearn.model_selection import train_test_split

class Loader:
    def __init__(self, data_path: str):
        pass

    def load_data(self):
        pass


class PickleLoader(Loader):
    """
    Loads from different extensions
    """
    def __init__(self,data_path: str):
        """
        Args:
            data_path (str): The path where the data is stored. Accepted formats: csv, pkl, h5.
        """        
        self.data_path = data_path

    def load_data(self):
        return pd.read_pickle(self.data_path)

class CSVLoader(Loader):
    """
    Loads from different extensions
    """
    def __init__(self,data_path: str):
        """
        Args:
            data_path (str): The path where the data is stored. Accepted formats: csv, pkl, h5.
        """        
        self.data_path = data_path

    def load_data(self):
        return pd.read_csv(self.data_path)

class HDFLoader(Loader):
    # ToDo: some data might come with keys, this needs to be handled
    """
    Loads from different extensions
    """
    def __init__(self,data_path: str):
        """
        Args:
            data_path (str): The path where the data is stored. Accepted formats: csv, pkl, h5.
        """        
        self.data_path = data_path

    def load_data(self):
        return pd.read_hdf(self.data_path)


def data_split(df: pd.DataFrame, train_size: float, valid_size:float, test_size: float):
    if train_size>1 or valid_size>1 or test_size>1:
        raise ValueError("Allowed data_args must be <1")
    if train_size+valid_size+test_size>=1:
        raise ValueError("Sum of allowed data_args must be <1")
    poolsize = 1- train_size-valid_size-test_size
    poolsizeprime = poolsize/(1-train_size)
    validsizeprime = (valid_size)/(1-poolsizeprime)/(1-train_size)  
    train, tmp = train_test_split(df, test_size = 1-train_size, random_state = 42)
    pool, tmp = train_test_split(tmp, test_size = 1-poolsizeprime, random_state = 42)
    valid, test = train_test_split(tmp, test_size = 1-validsizeprime, random_state = 42)      
    return train,valid,test,pool


class ActiveLearningStaticPoolSampler(ActiveLearner):
    """
    Loads a database of runs that have already been collected. 
    This is useful for testing various acquisition strategies
    in Active Learning on fixed datasets. Uses Pandas under the hood.
    """
    sampler_interface = S.ACTIVEDB
    def __init__(self, data_path: str, total_budget: int, init_num_samples: int, **AL_kwargs):
        """
        Args:
            data_path (str): The path where the data is stored. Accepted formats: csv, pkl, h5.
        """
        super().__init(self, **AL_kwargs)
        allowed_extensions = ['csv','h5','pkl']
        self.extension = data_path.split('.')[-1]
        self.data_args = AL_kwargs['data_Args']

        if self.extension not in allowed_extensions:
            raise ValueError(f"Extension {self.extension} not allowed. Please use any of {allowed_extensions}")
        if self.extension=='csv':
            self.data = CSVLoader(data_path=data_path).load_data()
        if self.extension=='pkl':
            self.data = PickleLoader(data_path=data_path).load_data()
        if self.extension=='h5':
            self.data = HDFLoader(data_path=data_path).load_data() 

        self.total_budget = total_budget
        self.init_num_samples = init_num_samples
        self.train,self.valid,self.test,self.pool = data_split(self.data, **self.data_args)

        # somehow needs to handle cases 
        self.mapping_column_indices = self.get_column_idx_mapping()
        self.train = torch.tensor(self.train)
        self.valid = torch.tensor(self.valid)
        self.test = torch.tensor(self.test)
        self.pool = torch.tensor(self.pool)

    def get_column_idx_mapping(self):
        return {idx:col for col, idx in zip(self.data.columns, range(self.data.columns))}
    
    def get_initial_parameters(self):
        return self.data.sample(self.init_num_samples)
    
    def update_pool_and_train(self, new_idxs):
        pool_idxs = torch.as_tensor(self.pool.index.values)
        train_idxs = torch.as_tensor(self.train.index.values)
        logical_new_idxs = torch.zeros(pool_idxs.shape[-1], dtype=torch.bool)
        logical_new_idxs[new_idxs] = True
        # We now append the new indices to the training set
        train_idxs = torch.cat([train_idxs, pool_idxs[logical_new_idxs]], dim=-1)
        # and remove them from the pool set
        pool_idxs = pool_idxs[~logical_new_idxs]        
        self.train = self.train[train_idxs]
        self.pool = self.pool[pool_idxs]
        

        