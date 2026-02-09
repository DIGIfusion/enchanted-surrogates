import numpy as np
from enchanted_surrogates.samplers.base_sampler import Sampler


class RandomSampler(Sampler):
    BATCH_SAMPLE_SIZE = 1

    def __init__(self, bounds, num_samples, parameters, **kwargs):
        self.num_samples = num_samples
        self.bounds = bounds
        self.parameters = parameters
        self.budget = kwargs.get('budget', num_samples)
        self.batch_size = kwargs.get("batch_size", self.num_samples)
        self.include_index = kwargs.get('include_index', False)
        self.num_repeats = kwargs.get('num_repeats', 1)
        self.seed = kwargs.get('seed', 42)
        # Create a generator with a fixed seed
        self.rng = np.random.default_rng(seed=self.seed)

    def get_next_samples(self) -> list[dict]:
        # TODO not use uniform?
        # TODO batch samples
        samples = []
        for _ in range(self.batch_size):
            # Use the generator instead of np.random
            params = [self.rng.uniform(low, high) for low, high in self.bounds]
            param_dict = {key: value for key, value in zip(self.parameters, params)}
            samples.append(param_dict)
        
        if self.num_repeats > 1:
            samples = samples * self.num_repeats

        if self.include_index:
            samples = [{**samp, 'index': ind} for samp, ind in zip(samples, range(self.submitted,self.submitted+len(samples)))]
        
        self.submitted += len(samples)
        return samples

    def register_future(self, future):
        """ Doesn't matter for random sampler TODO: Probably? """
        return None

    def register_futures(self, futures):
        return None
