# import sys
# import os
# from common import S, CSVLoader, PickleLoader, HDFLoader, data_split
from bmdal_reg.bmdal.algorithms import select_batch
from bmdal_reg.bmdal.feature_data import TensorFeatureData
import torch
from .base import Sampler 
from typing import Dict
from parsers import *
import numpy as np 
from common import S

class ActiveLearner(Sampler):
    sampler_interface = S.ACTIVE # NOTE: This could likely also be batch...
    def __init__(self, total_budget: int, parser_kwargs, model_kwargs, train_kwargs, *args, **kwargs): 
        self.total_budget = total_budget
        self.bounds = kwargs.get('bounds')
        self.parameters = kwargs.get('parameters', ['NA']*len(self.bounds))
        self.batch_size = kwargs.get('batch_size', 1)
        self.init_num_samples = kwargs.get('num_initial_points', self.batch_size)
        self.parser_kwargs = parser_kwargs
        self.model_kwargs = model_kwargs
        self.train_kwargs = train_kwargs
        
    def get_initial_parameters(self):
        # random samples 
        batch_samples = [] 
        for _ in range(self.init_num_samples):
            params = [torch.distributions.Uniform(lb, ub).sample().item() for (lb, ub) in self.bounds]
            param_dict = {key: value for key, value in zip(self.parameters, params)}
            batch_samples.append(param_dict)
        return batch_samples

    def _get_new_idxs_from_pool(self, model, train, pool) -> torch.Tensor:
        y_train = train[:, self.parser.output_col_idxs].float()
        train = train[:, self.parser.input_col_idxs].float()
        pool = pool[:, self.parser.input_col_idxs].float()
        print('\nData sizes', y_train.shape, train.shape, pool.shape)
        train_data = TensorFeatureData(train)
        pool_data = TensorFeatureData(pool)

        new_idxs, _ = select_batch(batch_size=self.batch_size, models=[model],
                           data={'train': train_data, 'pool': pool_data}, y_train=y_train,
                           selection_method='lcmd', sel_with_train=True,
                           base_kernel='grad', kernel_transforms=[('rp', [512])])
        
        return new_idxs
    
    def get_next_parameter(self, model, train, pool) -> list[dict[str, float]]:
        new_idxs = self._get_new_idxs_from_pool(model, train,pool)
        results = []
        for idx in new_idxs: 
            results.append({'input': self.parser.pool[idx, self.parser.input_col_idxs], 'output': None, 'pool_idxs':idx}) 
        # TODO: convert indicies to unscaled parameters
        return results 
    
        
    def collect_batch_results(self, results: list[dict[str, dict]]) -> torch.Tensor:
        outputs_as_tensor = torch.empty(len(results), len(self.parser.keep_keys))
        for n, result in enumerate(results): 
            x = result['inputs']
            y = result['output']
            outputs_as_tensor[n] = torch.cat((x, y))
        return outputs_as_tensor
    
    def update_pool_and_train(self):
        pass


class ActiveLearningStaticPoolSampler(ActiveLearner):
    """
    Loads a database of runs that have already been collected. 
    This is useful for testing various acquisition strategies
    in Active Learning on fixed datasets. Uses Pandas under the hood.
    """
    sampler_interface = S.ACTIVEDB
    def __init__(self, total_budget: int, init_num_samples: int, **kwargs):
        """
        Args:
            data_path (str): The path where the data is stored. Accepted formats: csv, pkl, h5.
        """
        super().__init__(self, **kwargs)
        self.parser = STATICPOOLparser(**self.parser_kwargs)
        #ToDd assert that the parser is an instance of STATICPOOLparser, or an enum
        self.total_budget = total_budget
        self.init_num_samples = init_num_samples
    
    def collect_batch_results(self, results: list[dict[str, dict]]) -> torch.Tensor:
        # iputs coming since we don't care about hte outputs in the static pool 
        print(results)
        idxs_as_tensor = torch.tensor([res['input']['pool_idxs'] for res in results])
        return idxs_as_tensor
    
    def get_initial_parameters(self) -> list[dict]:
        # get a set of random initial parameters
        params = []
        for _ in range(self.init_num_samples): 
            idxs = np.random.randint(0, len(self.parser.pool))
            # idxs = super().get_next_parameter(model, self.parser.train, self.parser.pool)
            result = {'input': self.parser.pool[idxs, self.parser.input_col_idxs], 'output': self.parser.pool[idxs, self.parser.output_col_idxs], 'pool_idxs':idxs}
            params.append(result)
        return params # self.parser.data.sample(self.init_num_samples)
    
    def get_next_parameter(self, model) -> list[Dict[str, float]]:
        idxs = self._get_new_idxs_from_pool(model, self.parser.train, self.parser.pool)
        results = []
        for idx in idxs: 
            results.append({'input': self.parser.pool[idx, self.parser.input_col_idxs], 'output': self.parser.pool[idx, self.parser.output_col_idxs], 'pool_idxs':idx}) 
        return results 

    def dump_results(self): 
        train_data = self.parser.train 
        for n in range(len(train_data)):
            print(train_data[n])