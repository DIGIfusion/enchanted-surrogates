import numpy as np
from enchanted_surrogates.samplers.base_sampler import Sampler


class RandomSampler(Sampler):
    BATCH_SAMPLE_SIZE = 1

    def __init__(self, bounds, budget, parameters, **kwargs):
        self.budget = budget
        self.bounds = bounds
        self.parameters = parameters

    def get_next_samples(self) -> list[dict]:
        # TODO not use uniform?
        # TODO batch samples
        list_param_dicts = []
        for _ in range(self.BATCH_SAMPLE_SIZE):
            params = [np.random.uniform(low, high) for low, high in self.bounds]
            param_dict = {key: value for key, value in zip(self.parameters, params)}
            list_param_dicts.append(param_dict)
        self.submitted += len(list_param_dicts)
        return list_param_dicts

    def register_future(self, future):
        """ Doesn't matter for random sampler TODO: Probably? """
        return None

    def register_futures(self, futures):
        return None
