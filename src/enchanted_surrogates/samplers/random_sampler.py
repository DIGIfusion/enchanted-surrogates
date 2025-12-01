import numpy as np
from enchanted_surrogates.samplers.base_sampler import Sampler


class RandomSampler(Sampler):
    BATCH_SAMPLE_SIZE = 1

    def __init__(self, bounds, budget, parameters, **kwargs):
        self.budget = budget
        self.bounds = bounds
        self.parameters = parameters
        self.batch_size = kwargs.get("batch_size", self.budget)
        self.include_index = kwargs.get('include_index', False)

    def get_next_samples(self) -> list[dict]:
        # TODO not use uniform?
        # TODO batch samples
        samples = []
        for _ in range(self.batch_size):
            params = [np.random.uniform(low, high) for low, high in self.bounds]
            param_dict = {key: value for key, value in zip(self.parameters, params)}
            samples.append(param_dict)
        
        if self.include_index:
            samples = [{**samp, 'index': ind} for samp, ind in zip(samples, range(self.submitted,len(samples)))]
        
        self.submitted += len(samples)
        return samples

    def register_future(self, future):
        """ Doesn't matter for random sampler TODO: Probably? """
        return None

    def register_futures(self, futures):
        return None
