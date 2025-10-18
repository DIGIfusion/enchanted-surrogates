
from enchanted_surrogates.samplers.base_sampler import Sampler
from enchanted_surrogates.utils.precise_imports import import_sampler
from itertools import product
import numpy as np
import importlib
import os
import sys

class MultiSampler(Sampler):
    """
    A sampler tha combines other samplers together so each dimension can be sampled in a different way
    
    Attributes:

    Methods:
        
    """
    def __init__(self, samplers, sample_all_combinations=False, *args, **kwargs):
        """
        Args:
        """
        samplers_keys = samplers.keys
        samplers_types = [samplers[k]['type'] for k in samplers_keys()]
        samplers_config = [samplers[k] for k in samplers_keys()]
        self.all_samplers = [import_sampler(type=sampler_type, sampler_config=sampler_config) for sampler_type, sampler_config in zip(samplers_types,samplers_config)]
        self.sample_all_combinations = sample_all_combinations
        
        self.budget = np.inf
    def get_next_samples(self):
        # get all initial parameters
        samples = []
        for sampler in self.all_samplers:
            samples.append(sampler.get_next_samples())
        print('debug multisampler samples', samples)
        
        if self.sample_all_combinations:
            from itertools import product
            # Cartesian product of all sublists
            combinations = product(*samples)
            # Merge each tuple of dicts into a single dict
            combined = [dict(kv for d in combo for kv in d.items()) for combo in combinations]
        else:
            # check they all have the same number of parameters
            len_ip = [len(ip) for ip in samples]
            if not len(set(len_ip)) == 1 and not self.sample_all_combinations:
                raise ValueError('THE SAMPLERS PROVIDED DO NOT RETURN THE SAME NUMBER OF INITIAL PARAMETERS')
            # combine them together
            # Transpose the outer list so each group contains dicts at the same index
            transposed = zip(*samples)
            # Merge each group of dicts into one dict
            combined = [dict(kv for d in group for kv in d.items()) for group in transposed]
        
        # TODO: IMPLIMENT BATCH SAMPLING
        self.budget = len(combined)
        self.submitted = len(combined)+1
        return combined
    
    
    def register_future(self, future):
        """ Doesn't matter for random sampler TODO: Probably? """
        return None

    def register_futures(self, futures):
        return None
