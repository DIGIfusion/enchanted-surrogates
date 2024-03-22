# sampler/hypercube.py

import numpy as np
from .base import Sampler
from itertools import product

class HypercubeSampler(Sampler):
    def __init__(self, bounds, num_samples, parameters):
        self.parameters = parameters
        self.bounds = bounds
        self.num_samples = num_samples
        self.hypercube_grid = list(self.sample_parameters())
        self.current_index = 0 

    def sample_parameters(self):
        samples = [np.linspace(bound[0], bound[1], self.num_samples) for bound in self.bounds]
        # Use itertools.product to create a Cartesian product of sample points, representing the hypercube
        for params_tuple in product(*samples):
            # Convert tuples to list to ensure serializability
            yield list(params_tuple)

    def get_next_parameter(self):
        if self.current_index < len(self.hypercube_grid):
            params = self.hypercube_grid[self.current_index]
            self.current_index += 1
            return params
        else:
            # Handle the case where all parameters have been iterated over
            return 1