""" 
parsers/STATICPOOLparser.py 

A parser for a static pool. 

Needs functionality: 

- Given set of observations, update pool and training sets
    - update_pool_and_train()
- Gets train and validation datasets 
    - get_train_val_dataset() 

"""

from .base import Parser
import torch
import numpy as np
import pandas as pd
import os
from common import data_split, CSVLoader, HDFLoader, PickleLoader, apply_scaler


class LabelledPoolParser(Parser):
    """
    A static pool is considered to be data that is labeled and unnormalized. 
    At the moment, this parser handles the scaling of data, but 
    does not hold copies of the scaled data as attributes. 
    
    Attributes: 
        train: torch.Tensor, unnormalized current training set 
        valid: torch.Tensor, unnormalized current validation set
        test : torch.Tensor, unnormalized current test set
        pool : torch.Tensor, unnormalized current pool
        input_col_idxs: list of indices corresponding to the inputs columns
        output_col_idxs: list of indices corresponding to the output columns
    
    """

    def __init__(self, data_path: str, *args, **kwargs):
        """ 
        Sets the above class attributes 
        
        needs data_path as positional argument
        But would like as keyword arguments: 
        - data_args: dict for the initial valid/test data split
        - target: list of target column names
        - inputs: list of input column names
        """
        self.data_args = kwargs.get("data_args")
        self.target_keys = kwargs.get("target")
        self.input_keys = kwargs.get("inputs")

        self.data = self.gather_data_from_storage(data_path)
        self.mapping_column_indices = self.get_column_idx_mapping()

        self.input_col_idxs = [
            self.mapping_column_indices[col_idx] for col_idx in self.input_keys
        ]
        self.output_col_idxs = [
            self.mapping_column_indices[col_idx] for col_idx in self.target_keys
        ]

        self.keep_keys = self.target_keys + self.input_keys
        self.data = self.data[self.keep_keys]        

        self.train, self.valid, self.test, self.pool = data_split(
            self.data, **self.data_args
        )
        self.train = torch.tensor(self.train.values)
        self.valid = torch.tensor(self.valid.values)
        self.test = torch.tensor(self.test.values)
        self.pool = torch.tensor(self.pool.values)

    def gather_data_from_storage(self, data_path) -> pd.DataFrame:
        extension = data_path.split(".")[-1]
        if extension == "csv":
            data = CSVLoader(data_path=data_path).load_data()
        elif extension == "pkl":
            data = PickleLoader(data_path=data_path).load_data()
        elif extension == "h5":
            data = HDFLoader(data_path=data_path).load_data()
        else:
            raise ValueError(
                f"Extension {extension} not allowed. Please use any of {['csv','h5','pkl']}"
            )
        return data

    def get_column_idx_mapping(self) -> dict:
        return {
            col: idx
            for col, idx in zip(self.data.columns, range(len(self.data.columns)))
        }

    def scale_train_val_test_pool(
        self,
        train: torch.Tensor,
        valid: torch.Tensor,
        test: torch.Tensor,
        pool: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        train, valid, test, pool, _ = apply_scaler(
            train, valid, test, pool, scaler=None, op="transform"
        )
        return train, valid, test, pool

    def get_unscaled_train_valid_test_pool_from_self(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Returns the unnormalized train, valid, test, and pool tensors """
        return self.train, self.valid, self.test, self.pool

    def make_train_valid_datasplit(
        self,
        train: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        """ 
        takes two tensors and returns the x,y splits based on input_col_idxs and output_col_idxs attributes
        also casts tensors to float 
        """
        x_train, y_train = (
            train[:, self.input_col_idxs].float(),
            train[:, self.output_col_idxs].float(),
        )
        x_valid, y_valid = (
            valid[:, self.input_col_idxs].float(),
            valid[:, self.output_col_idxs].float(),
        )

        return (x_train, y_train), (x_valid, y_valid)

    def update_pool_and_train(self, new_idxs):
        """
        Updates the (self) pool by removing sampled points and places them in train.
        """
        pool_idxs = torch.arange(0, len(self.pool))   # (self.pool.index.values)
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

    def print_dset_sizes(
        self,
    ) -> str:
        return f"Pool Size: {len(self.pool)}, Train Size: {len(self.train)}, Valid size: {len(self.valid)}, Test size: {len(self.test)}"

    def __len__(self) -> int:
        return len(self.pool)

    def write_input_file(self, *args, **kwargs):
        pass




class StreamingLabelledPoolParserJETMock(LabelledPoolParser):
    def __init__(self,data_path: str, *args, **kwargs):

        #NOTE TO SELF: Remember to enact these cuts on the data prior to using it
        # data_master = data_master.query("cluster=='left' & is_hmode==False")
        # and also to remove NaNs
        # nonzeros = data_master.query('machtor!=0').index

        super().__init__(data_path=data_path, *args,**kwargs)
        self.data_basepath = os.path.join(os.path.dirname(data_path),'../')
        streaming_kwargs = kwargs.get("streaming") # should contain number of campaigns and sampled shots per campaign
        self.num_shots_per_campaign = streaming_kwargs['num_shots_per_campaign']
        self.num_campaigns =  streaming_kwargs['num_campaigns']
        self.use_only_new = streaming_kwargs['use_only_new'] # should tell whether only the new data is used for valid pool and test 
        self.shots_seen = []

    def get_unscaled_train_valid_test_pool_from_self(
        self,
        campaign_id,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Returns the unnormalized train, valid, test, and pool tensors, concatenated with the new data."""
        # NOTE: it is better to save the new shot inputs in different files. Appending to existing data is costly and not typical for the streaming case. 
        # Each folder will contain data from a different campaign. Each file is numbered using the shot number.
        campaign_folder = os.path.join(self.data_basepath,str(campaign_id))
        files_this_campaign = os.listdir(campaign_folder)
        count = 0
        while True:
            if count==len(files_this_campaign):
                print('Campaign has no more shots')
                return None, None, None, None
            data_path = np.random.choice(files_this_campaign)
            if data_path not in self.shots_seen:
                break
            else:
                count +=1
            
        data = self.gather_data_from_storage(data_path=data_path)
        data = data[self.keep_keys]        

        train, valid, test, pool = data_split(
            data, **self.data_args
        )
        train = torch.tensor(train.values)
        valid = torch.tensor(valid.values)
        test = torch.tensor(test.values)
        pool = torch.tensor(pool.values)    

        self.train = torch.cat((self.train, train))
        if not self.use_only_new:
            self.valid = torch.cat((self.valid, valid))
            self.test = torch.cat((self.test, test))
            self.pool = torch.cat((self.pool, pool))        
        return self.train, self.valid, self.test, self.pool
    