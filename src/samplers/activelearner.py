import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'bmdal'))

from common import S, CSVLoader, PickleLoader, HDFLoader, data_split
from bmdal.bmdal_reg.bmdal.algorithms import select_batch
from bmdal.bmdal_reg.bmdal.feature_data import TensorFeatureData
import torch
from .base import Sampler 
from typing import Dict
from parsers import *
import numpy as np 

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

    def get_next_parameter(self, model, train, pool) -> Dict[str, float]:
        # TODO: fix pseudocode

        y_train = train[:, self.parser.output_col_idxs].float()
        print(type(y_train), type(train), type(pool))
        print(y_train.shape, train[:, self.parser.input_col_idxs][:, None].shape, pool[:, self.parser.input_col_idxs].shape)
        print(type(model))
        print(type(model.model))
        train_data = TensorFeatureData(train[:, self.parser.input_col_idxs].float()[:, None])
        pool_data = TensorFeatureData(pool[:, self.parser.input_col_idxs].float()[:, None])
        # model = model.model.float()
        feature_data =  {'train': train_data, 'pool': pool_data}
        
        new_idxs, _ = select_batch(batch_size=50, models=[model.model], 
                                          data=feature_data, y_train=y_train,
                                          selection_method='random', sel_with_train=False,
                                          base_kernel='predictions', kernel_transforms=[])# [('rp', [512])])
        return new_idxs
    
    def collect_batch_results(self, results: list[dict[str, dict]]) -> torch.Tensor:
        outputs_as_tensor = torch.empty(len(results), len(self.parser.keep_keys))
        for n, result in enumerate(results): 
            x = result['inputs']
            y = result['output']
            outputs_as_tensor[n] = torch.concatenate((x, y))
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
        idxs = super().get_next_parameter(model, self.parser.train, self.parser.pool)
        # print(len(idxs))
        # if len(idxs) == 1: 
        results = []
        for idx in idxs: 
            results.append({'input': self.parser.pool[idx, self.parser.input_col_idxs], 'output': self.parser.pool[idx, self.parser.output_col_idxs], 'pool_idxs':idx}) 
        return results 
        
    def dump_results(self): 
        train_data = self.parser.train 
        for n in range(len(train_data)): 
            print(train_data[n])