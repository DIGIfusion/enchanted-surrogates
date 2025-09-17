# sampler/grid.py

# from .base import Sampler
from itertools import product
import numpy as np
import importlib
from enchanted_surrogates.utils.precise_imports import import_sampler
from enchanted_surrogates.samplers.base_sampler import Sampler


class NestedSampler(Sampler):
    """
    TODO: add docstring
    """

    def __init__(self, samplers, *args, **kwargs):
        """
        Args:
        """
        samplers_keys = samplers.keys
        samplers_types = [samplers[k]['type'] for k in samplers_keys()]
        samplers_kwargs = [samplers[k] for k in samplers_keys()]
        self.all_samplers = [import_sampler(sampler_type, sampler_kwargs) for sampler_type, sampler_kwargs in zip(samplers_types, samplers_kwargs)]
        self.budget = kwargs.get('budget')
        self.batch_size = kwargs.get('batch_size', self.budget)
        self.submitted = 0
        self.all_parameters = [param for sampler in self.all_samplers for param in sampler.parameters]
        
    def get_next_samples(self):
        # get all initial parameters
        initial_parameters = []
        for sampler in self.all_samplers:
            initial_parameters.append(sampler.get_next_samples())

        # Cartesian product of all sublists
        combinations = product(*initial_parameters)
        # Merge each tuple of dicts into a single dict
        combined = [dict(kv for d in combo for kv in d.items()) for combo in combinations]
        
        self.submitted += len(combined)
        # self.has_budget = False # nested sampler only supports one batch. For nested active sampling a bespoke nested sampler is needed for each application
        return combined
    
    def register_future(self, future):
        """ Doesn't matter for random sampler TODO: Probably? """
        return None

    def register_futures(self, futures):
        return None
