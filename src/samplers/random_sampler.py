# sampler/random_sampler.py

from .base import Sampler
import numpy as np
from itertools import product


class RandomSampler(Sampler):
    """
    Returns a number of random samples within the specified bounds.

    Number of samples should be specified as an integer.

    This throws errors if you are asking for something insane,
        e.g., 10 parameters  for 10 samples each -> 10 billion
        so hard limit at 100.000

    """

    def __init__(self, bounds, num_samples, parameters):
        if not isinstance(num_samples, int):
            raise TypeError(
                f"The input parameter num_samples should be an integer. {num_samples}"
            )

        self.total_budget = num_samples
        # check for stupidity
        if self.total_budget > 100000:
            raise Exception(
                (
                    "Can not do random sampling on more than 10000 samples, ",
                    f"you requested {self.total_budget}",
                )
            )

        self.parameters = parameters
        self.bounds = bounds
        self.num_initial_points = self.total_budget
        self.num_samples = num_samples
        self.samples = list(self.generate_parameters())
        self.current_index = 0

    def generate_parameters(self):
        samples = [
            [np.random.uniform(bound[0], bound[1]) for bound in self.bounds]
            for _ in range(self.num_samples)
        ]
        return samples

    def get_next_parameter(self):
        if self.current_index < len(self.samples):
            params = self.samples[self.current_index]
            self.current_index += 1
            param_dict = {key: value for key, value in zip(self.parameters, params)}
            return param_dict
        else:
            return None  # TODO: implement when done iterating!
