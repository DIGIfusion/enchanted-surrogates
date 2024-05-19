from .base import Parser
import torch
import pandas as pd 
from common import data_split, CSVLoader, HDFLoader, PickleLoader, apply_scaler


class STATICPOOLparser(Parser):
    """ This is a dummy class as for the moment the sampler handles everything"""
    def __init__(self, data_path: str , *args, **kwargs): 
        init_num_samples = kwargs.get('init_num_samples')
        self.data_args = kwargs.get('data_args')
        self.target_keys = kwargs.get('target')
        self.input_keys = kwargs.get('inputs')
        self.keep_keys = self.target_keys+self.input_keys

        self.data = self.gather_data_from_storage(data_path)
        self.data = self.data[self.keep_keys]
        self.mapping_column_indices = self.get_column_idx_mapping()
        
        self.input_col_idxs = [self.mapping_column_indices[col_idx] for col_idx in self.input_keys]
        self.output_col_idxs = [self.mapping_column_indices[col_idx] for col_idx in self.target_keys]

        self.train, self.valid, self.test, self.pool = data_split(self.data, **self.data_args)
        self.train = self.train.sample(init_num_samples)
        self.train = torch.tensor(self.train.values)
        self.valid = torch.tensor(self.valid.values)
        self.test = torch.tensor(self.test.values)
        self.pool = torch.tensor(self.pool.values)    
        # print('\n train SIZE', self.train.shape, )
        # print('\n valid SIZE', self.valid.shape)
        # print('\n test SIZE', self.test.shape)
        # print('\n pool SIZE', self.pool.shape)
        self.train, self.valid, self.test, self.pool, self.scaler = apply_scaler(self.train, self.valid, self.test, self.pool, scaler=None, op='transform')
    
    def gather_data_from_storage(self, data_path) -> pd.DataFrame: 
        extension = data_path.split('.')[-1]
        if extension == 'csv':
            data = CSVLoader(data_path=data_path).load_data()
        elif extension == 'pkl':
            data = PickleLoader(data_path=data_path).load_data()
        elif extension == 'h5':
            data = HDFLoader(data_path=data_path).load_data()
        else: 
            raise ValueError(f"Extension {extension} not allowed. Please use any of {['csv','h5','pkl']}")
        return data

    def get_column_idx_mapping(self) -> dict:
        return {col:idx  for col, idx in zip(self.data.columns, range(len(self.data.columns)))}
        
    def get_train_valid(self) -> tuple[torch.Tensor, torch.Tensor]:
        # TODO this should be a method in the base Parser class
        return self.train, self.valid

    def get_train_valid_datasplit(self) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]: 
        """ returns (x_train, y_train), (x_valid, y_valid) """
        x_train, y_train = self.train[:, self.input_col_idxs].float(), self.train[:, self.output_col_idxs].float()
        x_valid, y_valid = self.valid[:, self.input_col_idxs].float(), self.valid[:, self.output_col_idxs].float()
        return (x_train, y_train), (x_valid, y_valid)

    def update_pool_and_train(self, new_idxs):
        """
        Updates the pool by removing sampled points and places them in train. 
        """
        pool_idxs = torch.arange(0, len(self.pool)) # (self.pool.index.values)
        train_idxs = torch.arange(0, len(self.train)) # (self.train.index.values)
        logical_new_idxs = torch.zeros(pool_idxs.shape[-1], dtype=torch.bool)
        logical_new_idxs[new_idxs] = True
        # We now append the new indices to the training set
        train_idxs = torch.cat([train_idxs, pool_idxs[logical_new_idxs]], dim=-1)
        train_new = torch.cat([self.train, self.pool[new_idxs]])
        # and remove them from the pool set
        pool_idxs = pool_idxs[~logical_new_idxs]
        self.train = train_new
        self.pool = self.pool[pool_idxs]


    def write_input_file(self):
        pass 