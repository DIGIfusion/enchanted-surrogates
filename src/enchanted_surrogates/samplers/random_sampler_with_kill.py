import numpy as np
from enchanted_surrogates.samplers.base_sampler import Sampler


class RandomSamplerWithKill(Sampler):
    BATCH_SAMPLE_SIZE = 1

    def __init__(self, bounds, budget, parameters, **kwargs):
        self.budget = budget
        self.bounds = bounds
        self.parameters = parameters
        self.batch_size = kwargs.get("batch_size", self.budget)
        self.kill_after = kwargs.get(
            "kill_after"
        )  # sampler dies after this many batches
        self.batches_done = 0

    def get_next_samples(self) -> list[dict]:
        # TODO not use uniform?
        # TODO batch samples
        if self.kill_after and self.kill_after <= self.batches_done:
            raise Exception("Time to die reached")

        list_param_dicts = []
        for _ in range(self.batch_size):
            params = [np.random.uniform(low, high) for low, high in self.bounds]
            param_dict = {key: value for key, value in zip(self.parameters, params)}
            list_param_dicts.append(param_dict)
        self.submitted += len(list_param_dicts)
        self.batches_done += 1

        return list_param_dicts

    def register_future(self, future):
        """Doesn't matter for random sampler TODO: Probably?"""
        return None

    def register_futures(self, futures):
        return None
