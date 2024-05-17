from .base import Parser
import torch
from common import data_split, CSVLoader, HDFLoader, PickleLoader, apply_scaler

class STATICPOOLparser(Parser):
    """ This is a dummy class as for the moment the sampler handles everything"""
    def __init__(self, data_path: str , *args, **kwargs): 
        self.extension = data_path.split('.')[-1]
        if self.extension not in ['csv','h5','pkl']:
            raise ValueError(f"Extension {self.extension} not allowed. Please use any of {['csv','h5','pkl']}")

        if self.extension == 'csv':
            self.data = CSVLoader(data_path=data_path).load_data()

        if self.extension == 'pkl':
            self.data = PickleLoader(data_path=data_path).load_data()

        if self.extension == 'h5':
            self.data = HDFLoader(data_path=data_path).load_data() 

        init_num_samples = kwargs.get('init_num_samples')
        self.data_args = kwargs.get('data_args')
        self.target_keys = kwargs.get('target')
        self.input_keys = kwargs.get('inputs')
        self.keep_keys = self.target_keys+self.input_keys
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
        print('\n train SIZE', self.train.shape)
        print('\n valid SIZE', self.valid.shape)
        print('\n test SIZE', self.test.shape)
        print('\n pool SIZE', self.pool.shape)
        self.train, self.valid, self.test, self.pool, self.scaler = apply_scaler(self.train, self.valid, self.test, self.pool, scaler=None, op='transform')
    

    def get_column_idx_mapping(self):
        return {col:idx  for col, idx in zip(self.data.columns, range(len(self.data.columns)))}
        
    def get_train_valid(self):
        # TODO this should be a method in the base Parser class
        return self.train, self.valid

    def update_pool_and_train(self, new_idxs):
        """
        This is essentially the read_outputs argument of  parser
        """
        # NOTE: I think we don't care exactly about the old indicies so we can reindex here? 

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