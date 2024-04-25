# sampler/grid.py

from .base import Sampler
import numpy as np
from itertools import product


class ArraySampler(Sampler):
    """
    Creates combinations of the given samples.
    The samples are directly specified in 'bounds' in the config file.

    For example:
        sampler:
          type: Array
          bounds: [[5, 7, 77, 199], [0.02, 0.2]]
          num_samples: [4, 2]
          parameters: ['a', 'b']

    would create the following sample space:
        [[5, 0.02], [5, 0.2], [7, 0.02], [7, 0.2], [77, 0.02],
        [77, 0.2], [199, 0.02], [199, 0.2]]

    This throws errors if you are asking for something insane,
        e.g., 10 parameters  for 10 samples each -> 10 billion
        so hard limit at 100.000

    """

    def __init__(self, bounds, num_samples, parameters):
        num_samples = [len(b) for b in bounds]

        # check for stupidity
        self.total_budget = np.prod(np.array(num_samples))
        if self.total_budget > 100000:
            raise Exception(
                (
                    "Can not do grid search on more than 10000 samples, ",
                    f"you requested {self.total_budget}",
                )
            )

        self.parameters = parameters
        self.bounds = bounds
        self.num_initial_points = np.prod(num_samples)
        self.num_samples = num_samples
        self.samples = list(self.generate_parameters())
        self.current_index = 0

    def generate_parameters(self):
        samples = self.bounds
        # Use itertools.product to create a Cartesian product of sample
        # points, representing the hypercube
        for params_tuple in product(*samples):
            # Convert tuples to list to ensure serializability
            yield list(params_tuple)

    def get_next_parameter(self):
        if self.current_index < len(self.samples):
            params = self.samples[self.current_index]
            self.current_index += 1
            param_dict = {key: value for key, value in zip(self.parameters, params)}
            return param_dict
        else:
            return None  # TODO: implement when done iterating!
