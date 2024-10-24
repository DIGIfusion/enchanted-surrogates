""" 
parsers/STATICPOOLparser.py 

A parser for a static pool. 

Needs functionality: 

- Given set of observations, update pool and training sets
    - update_pool_and_train()
- Gets train and validation datasets 
    - get_train_val_dataset() 

"""

from curses import meta
from .base import Parser
import torch
import numpy as np
import pandas as pd
from glob import glob
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
        self.use_only_new_for_train = kwargs.get('use_only_new_for_train',False) # should tell whether only the new data is used for valid pool and test 
        self.use_only_new_for_pool = kwargs.get('use_only_new_for_pool',False)
        self.data = self.gather_data_from_storage(data_path)
        self.keep_keys = self.target_keys + self.input_keys 
        self.data = self.data[self.keep_keys]

        self.mapping_column_indices = self.get_column_idx_mapping()

        self.input_col_idxs = [
            self.mapping_column_indices[col_idx] for col_idx in self.input_keys
        ]
        self.output_col_idxs = [
            self.mapping_column_indices[col_idx] for col_idx in self.target_keys
        ]


        self.train, self.valid, self.test, self.pool = data_split(
            self.data, **self.data_args
        )
        self.train = torch.tensor(self.train.values)
        self.valid = torch.tensor(self.valid.values)
        self.test = torch.tensor(self.test.values)
        self.pool = torch.tensor(self.pool.values)

        assert not torch.isnan(self.train).any()
        assert not torch.isnan(self.valid).any()
        assert not torch.isnan(self.test).any()
        assert not torch.isnan(self.pool).any()

        return 

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
        assert not torch.isnan(train).any()
        train, valid, test, pool, scaler = apply_scaler(
            train, valid, test, pool, scaler=None, op="transform"
        )
        self.scaler = scaler
        return train, valid, test, pool

    def has_nans_from_scaling(
        self   
    ) -> bool:
        """
        This is a helper function that identifies when NaNs appear because the batch has identical values in at least one of the columns
        """
        return self.scaler.find_zeros()


    def get_unscaled_train_valid_test_pool_from_self(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Returns the unnormalized train, valid, test, and pool tensors """
        return self.train, self.valid, self.test, self.pool

    def make_train_valid_test_datasplit(
        self,
        train: torch.Tensor,
        valid: torch.Tensor,
        test: torch.Tensor,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
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
        x_test, y_test = (
            test[:, self.input_col_idxs].float(),
            test[:, self.output_col_idxs].float(),
        )
        assert not torch.isnan(x_train).any()
        return (x_train, y_train), (x_valid, y_valid), (x_test, y_test)

    def update_pool_and_train(self, new_idxs):
        """
        Updates the (self) pool by removing sampled points and places them in train.
        """
        check_sum = len(self.train) + len(self.pool)
        pool_idxs = torch.arange(0, len(self.pool))   # (self.pool.index.values)
        logical_new_idxs = torch.zeros(pool_idxs.shape[-1], dtype=torch.bool)
        logical_new_idxs[new_idxs] = True
        train_new = self.pool[logical_new_idxs]
        if self.use_only_new_for_train:
            self.train = train_new
        else:        
            self.train = torch.cat([self.train,train_new])
        self.pool = self.pool[~logical_new_idxs]

        assert not torch.isnan(self.train).any()
        assert not torch.isnan(self.valid).any()
        assert not torch.isnan(self.test).any()
        assert not torch.isnan(self.pool).any()
        if not self.use_only_new_for_train:
            assert len(self.train) + len(self.pool) == check_sum

    def print_dset_sizes(
        self,
    ) -> str:
        return f"Pool Size: {len(self.pool)}, Train Size: {len(self.train)}, Valid size: {len(self.valid)}, Test size: {len(self.test)}"

    def __len__(self) -> int:
        return len(self.pool)

    def write_input_file(self, *args, **kwargs):
        pass




class StreamingLabelledPoolParserJETMock(LabelledPoolParser):
    """
    Parses data from a mock batched data stream, that has already 
    been labelled and that is unnormalised.
    At the moment, this parser handles the scaling of data, but 
    does not hold copies of the scaled data as attributes. 
    The unscaled  data is stored as attributes and it gets modified
    as data from the batched stream is processed.
    
    Attributes: 
        train: torch.Tensor, unnormalized current training set 
        valid: torch.Tensor, unnormalized current validation set
        test : torch.Tensor, unnormalized current test set
        pool : torch.Tensor, unnormalized current pool
        input_col_idxs: list of indices corresponding to the inputs columns
        output_col_idxs: list of indices corresponding to the output columns    
    """
    def __init__(self,data_path: str, *args, **kwargs):

        super().__init__(data_path=data_path, *args,**kwargs)
        
        streaming_kwargs = kwargs.get("streaming_kwargs") # should contain number of campaigns and sampled shots per campaign
        self.num_campaigns =  streaming_kwargs.get('num_campaigns', 10)
        self.num_shots_per_campaign = streaming_kwargs.get('num_acquisitions',3)
        self.data_basepath = os.path.join(os.path.dirname(data_path),'../')
        self.shots_seen = [self.data_basepath]
        self.all_shots_seen = [self.data_basepath]
        self.campaign_id = 0
            

    def get_unscaled_train_valid_test_pool_from_self(
        self
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Returns the unnormalized train, valid, test, and pool tensors, concatenated with the new data.
        """
        
        if len(self.shots_seen)>=self.num_shots_per_campaign:
            self.campaign_id+=1
            self.shots_seen = []
            if self.campaign_id==self.num_campaigns:
                return 'break',-1,-1,-1

        campaign_folder = os.path.join(self.data_basepath,"campaign_"+str(self.campaign_id))
        files_this_campaign = os.listdir(campaign_folder)

        count = 0
        while True:
            if count==len(files_this_campaign):
                print('Campaign has no more shots')
                self.campaign_id+=1
                self.shots_seen = []
                return None, None, None, None
            this_file = np.random.choice(files_this_campaign)
            data_path = os.path.join(campaign_folder,this_file)
            if data_path not in self.shots_seen:
                self.shots_seen.append(data_path)
                self.all_shots_seen.append(data_path)
                break
            else:
                count +=1
            
        print(f'Campaign number: {self.campaign_id}, shot number: {len(self.shots_seen)} ')
        data = self.gather_data_from_storage(data_path=data_path)
        data = data[self.keep_keys]                
        _, valid, test, pool = data_split(
            data, **self.data_args
        )
        valid = torch.tensor(valid.values)
        test = torch.tensor(test.values)
        pool = torch.tensor(pool.values)            
        self.valid = torch.cat((self.valid, valid))
        self.test = torch.cat((self.test, test))
        if self.use_only_new_for_pool:
            self.pool = pool
        else:
            self.pool = torch.cat((self.pool, pool))       
        # NOTE: self.train was set in update_pool_and_train(...)        
        return self.train, self.valid, self.test, self.pool        



class StreamingLabelledLumpedPoolParserJETMock(StreamingLabelledPoolParserJETMock):
    """
    Parses data from a mock batched data stream, that has already 
    been labelled and that is unnormalised.
    At the moment, this parser handles the scaling of data, but 
    does not hold copies of the scaled data as attributes. 
    The unscaled  data is stored as attributes and it gets modified
    as data from the batched stream is processed.
    
    Attributes: 
        train: torch.Tensor, unnormalized current training set 
        valid: torch.Tensor, unnormalized current validation set
        test : torch.Tensor, unnormalized current test set
        pool : torch.Tensor, unnormalized current pool
        input_col_idxs: list of indices corresponding to the inputs columns
        output_col_idxs: list of indices corresponding to the output columns    
    """
    def __init__(self,data_path: str, *args, **kwargs):
        """
        MAX_POOL_SIZE_PER_CAMPAIGN is needed as acquisition time increases greatly with pool size. 
        """
        
        self.data_args = kwargs.get("data_args")
        self.target_keys = kwargs.get("target")
        self.input_keys = kwargs.get("inputs")
        self.meta_keys = kwargs.get("meta", None)        
        self.use_only_new_for_train = kwargs.get('use_only_new_for_train',False) # should tell whether only the new data is used for valid pool and test 
        self.use_only_new_for_pool = kwargs.get('use_only_new_for_pool',False)
        streaming_kwargs = kwargs.get("streaming_kwargs") # should contain number of campaigns and sampled shots per campaign
        self.num_campaigns =  streaming_kwargs.get('num_campaigns', 10)
        self.num_acquisitions = streaming_kwargs.get('num_acquisitions', 5)
        self.MAX_POOL_SIZE_PER_CAMPAIGN = streaming_kwargs.get('MAX_POOL_SIZE_PER_CAMPAIGN', 5000)

        self.basepath = data_path
        self.acquisition_number = 0 
        self.campaign_id = 0
        self.batches = 0

        #TODO: obsolete lines below, remove    
        if self.meta_keys is None:
            raise ValueError("`meta_keys` must be defined in the config file for object `StreamingLabelledLumpedPoolParserJETMock`")
        self.keep_keys = self.target_keys + self.input_keys #+ self.meta_keys

        # Setup data
        train, valid, test, pool = self.gather_data_from_storage()
        self.mapping_column_indices = self.get_column_idx_mapping(train)
        self.input_col_idxs = [
            self.mapping_column_indices[col_idx] for col_idx in self.input_keys
        ]
        self.output_col_idxs = [
            self.mapping_column_indices[col_idx] for col_idx in self.target_keys
        ]
        # self.meta_col_idxs = [
        #     self.mapping_column_indices[col_idx] for col_idx in self.meta_keys
        # ]                        
        self.train, self.valid, self.test, self.pool = self._get_tensors(train,valid,test,pool)
        print('INSIDE INIT', self.print_dset_sizes())
        self.campaign_id +=1
        self.init_pool_size = len(self.pool)

    def _get_tensors(self, train, valid, test, pool):
        train = torch.tensor(train.values)
        valid = torch.tensor(valid.values)
        test = torch.tensor(test.values)
        pool = torch.tensor(pool.values)

        assert not torch.isnan(train).any()
        assert not torch.isnan(valid).any()
        assert not torch.isnan(test).any()
        assert not torch.isnan(pool).any()
        return train, valid, test, pool
    
    def get_column_idx_mapping(self, data) -> dict:
        return {
            col: idx
            for col, idx in zip(data.columns, range(len(data.columns)))
        }

    def gather_data_from_storage(self):
        data = super().gather_data_from_storage( os.path.join(self.basepath,f'campaign_{self.campaign_id}.csv'))
        data = data[self.keep_keys]
       # assert np.all(data[self.meta_keys]==self.campaign_id)
        train, valid, test, pool = data_split(
            data, **self.data_args
        )        
        if len(pool)>self.MAX_POOL_SIZE_PER_CAMPAIGN:
            pool = pool.sample(self.MAX_POOL_SIZE_PER_CAMPAIGN)
        
        return train, valid, test, pool

    # def filter_campaigns(self, data, campaign_id):
    #     meta = data[:,self.meta_col_idxs]
    #     idxs = torch.arange((len(meta)))
    #     idxs = torch.where(meta==campaign_id, idxs, -1)
    #     idxs = idxs[idxs>=0]
    #     data = data[idxs]                
    #     return data[idxs]
    
    def get_unscaled_train_valid_test_pool_from_self(
            self
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Returns the unnormalized train, valid, test, and pool tensors, concatenated with the new data.
        """
        if (self.acquisition_number>=self.num_acquisitions):
            self.acquisition_number = 0 
            self.campaign_id +=1
        elif (self.campaign_id>=self.num_campaigns):
            return 'break',-1,-1,-1   
        
        if self.acquisition_number==0:
            # NOTE: prevents data from the same campaign to be loaded again when performing multiple aquisitions in a given campaign
            train, valid, test, pool = self.gather_data_from_storage()
            train, valid, test, pool = self._get_tensors(train, valid, test, pool)
            # valid = self._filter_campaigns(self.valid, campaign_id=self.campaign_id)
            # test = self._filter_campaigns(self.test, campaign_id=self.campaign_id)
            # pool = self._filter_campaigns(self.pool, campaign_id=self.campaign_id)

            self.valid = torch.cat((self.valid, valid))
            self.test = torch.cat((self.test, test))
            if self.use_only_new_for_pool:
                self.pool = pool
            else:
                self.pool = torch.cat((self.pool, pool))       
            # NOTE: self.train was set in update_pool_and_train(...)     
            self.pool_size = len(self.pool)   

        # self.batches += 128*(self.acquisition_number+1)
        # print('DEBUG ================ ',self.acquisition_number, self.campaign_id, self.pool_size, self.batches)
        # assert len(self.pool) == self.pool_size - self.batches

        self.acquisition_number +=1
        return self.train, self.valid, self.test, self.pool       
