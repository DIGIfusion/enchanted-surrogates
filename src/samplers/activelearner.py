import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'bmdal'))

from common import S, CSVLoader, PickleLoader, HDFLoader, data_split
from bmdal.bmdal_reg.bmdal.algorithms import select_batch
from bmdal.bmdal_reg.bmdal.feature_data import TensorFeatureData
import torch
from .base import Sampler 
from typing import Dict
import parsers


class ActiveLearner(Sampler):
    sampler_interface = S.ACTIVE # NOTE: This could likely also be batch...
    def __init__(self, total_budget: int, parameters, parser_args, *args, **kwargs): 
        self.total_budget = total_budget
        self.bounds = kwargs.get('bounds')
        self.parameters = kwargs.get('parameters', ['NA']*len(self.bounds))
        self.batch_size = kwargs.get('batch_size', 1)
        self.init_num_samples = kwargs.get('num_initial_points', self.batch_size)
        # TODO: below is specific to the dataset, we should have the parser handle this stuff...., or introduce the parser
        self.parser = getattr(parsers, parser_args.pop('type'))(**parser_args)

    def get_initial_parameters(self):
        # random samples 
        batch_samples = [] 
        for _ in range(self.init_num_samples):
            params = [torch.distributions.Uniform(lb, ub).sample().item() for (lb, ub) in self.bounds]
            param_dict = {key: value for key, value in zip(self.parameters, params)}
            batch_samples.append(param_dict)
        return batch_samples

    def get_train_valid(self):
        return self.train, self.valid

    def get_next_parameter(self, model, train, pool) -> Dict[str, float]:
        # 
        # TODO: fix pseudocode

        train = TensorFeatureData(train.x)
        pool = TensorFeatureData(pool.x)
        y_train = train.y
        feature_data =  {'train': train, 'pool': pool}
        new_idxs, al_stats = select_batch(models=[model], data=feature_data, y_train=y_train,
                                        selection_method='lcmd', sel_with_train=True, 
                                        base_kernel='grad', kernel_transforms=[('rp', [512])])
        new_idxs, _ = None, None
        return new_idxs
    
    def update_pool_and_train(self):
        pass


class ActiveLearningStaticPoolSampler(ActiveLearner):
    """
    Loads a database of runs that have already been collected. 
    This is useful for testing various acquisition strategies
    in Active Learning on fixed datasets. Uses Pandas under the hood.
    """
    sampler_interface = S.ACTIVEDB
    def __init__(self, data_path: str, total_budget: int, init_num_samples: int, AL_kwargs, **kwargs):
        """
        Args:
            data_path (str): The path where the data is stored. Accepted formats: csv, pkl, h5.
        """
        super().__init__(self, **kwargs)
        
        #ToDd assert that the parser is an instance of STATICPOOLparser, or an enum
        self.extension = data_path.split('.')[-1]
        self.data_args = AL_kwargs['data_args']

        if self.extension not in ['csv','h5','pkl']:
            raise ValueError(f"Extension {self.extension} not allowed. Please use any of {['csv','h5','pkl']}")

        if self.extension == 'csv':
            self.data = CSVLoader(data_path=data_path).load_data()

        if self.extension == 'pkl':
            self.data = PickleLoader(data_path=data_path).load_data()

        if self.extension == 'h5':
            self.data = HDFLoader(data_path=data_path).load_data() 

        self.total_budget = total_budget
        self.init_num_samples = init_num_samples
        self.train, self.valid, self.test, self.pool = data_split(self.data, **self.data_args)

        # somehow needs to handle cases 
        self.train = torch.tensor(self.train.values)
        self.valid = torch.tensor(self.valid.values)
        self.test = torch.tensor(self.test.values)
        self.pool = torch.tensor(self.pool.values)
    
    def get_initial_parameters(self):
        return self.data.sample(self.init_num_samples)
    
    def get_next_parameter(self, model, train, pool) -> Dict[str, float]:
        idxs = super().get_next_parameter(model, train, pool)
        result = {'input': pool[idxs,self.parser.input_col_idxs], 'output': pool.loc[idxs,self.parser.output_col_idxs]}
        return result 
    
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
