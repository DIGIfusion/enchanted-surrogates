# sampler/random.py 

from .base import Sampler
from common import S
from typing import Union
import torch 

def get_parameter_distributions(bounds: list[list[Union[int,float]]], 
                                parameter_distributions: list[str]) -> list[torch.distributions.distribution.Distribution]: 
    param_distributions = [] 
    for param_dist_string, bound in zip(parameter_distributions, bounds): 
        lbound, upbound = bound
        param_dist = getattr(torch.distributions, param_dist_string)(lbound, upbound)
        param_distributions.append(param_dist)
    return param_distributions

class RandBatchSampler(Sampler): 
    sampler_interface = S.BATCH
    def __init__(self, 
                 bounds: list[[list[Union[int,float]]]], 
                 batch_size: int, 
                 total_budget: int,
                 parameters: list[str], ):

        self.total_budget = total_budget
        self.parameters = parameters
        self.bounds = bounds
        if batch_size <= 1: 
            raise ValueError('Batch size needs to be greator than 1, if 1 then use RandSampler')

        self.batch_size = self.num_initial_points = batch_size 
        parameter_strings = ['Uniform' for _ in range(len(bounds))]
        self.parameter_distributions = get_parameter_distributions(bounds, parameter_strings)
       
    def get_next_parameter(self, ):
        # batch version of get next parameter
        batch_samples = [] 
        for _ in range(self.batch_size):
            params = [dist.sample().item() for dist in self.parameter_distributions]
            param_dict = {key: value for key, value in zip(self.parameters, params)}
            batch_samples.append(param_dict)
        return batch_samples

    def get_initial_parameters(self, ): 
        return self.get_next_parameter()


class RandSampler(Sampler):
    sampler_interface = S.SEQUENTIAL
    def __init__(self, bounds, num_samples, parameters):
        self.parameters = parameters 
        self.bounds = bounds 
        self.num_initial_points = num_samples
        self.total_budget = num_samples
        parameter_strings = ['Uniform' for _ in range(len(bounds))]
        
        self.parameter_distributions = get_parameter_distributions(bounds, parameter_strings)

    def get_next_parameter(self, ): 
        params = [dist.sample().item() for dist in self.parameter_distributions]
        param_dict = {key: value for key, value in zip(self.parameters, params)}
        return param_dict

    def get_initial_parameters(self, ): 
        return [self.get_next_parameter() for _ in range(self.num_initial_points)]
