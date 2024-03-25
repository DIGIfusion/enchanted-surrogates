# sampler/hypercube.py

import numpy as np
from .base import Sampler
from itertools import product

class HypercubeSampler(Sampler):
    def __init__(self, bounds, num_samples, parameters):
        # check for stupidity
        self.total_budget = (num_samples)**(len(parameters))
        if (num_samples)**(len(parameters)) > 100000: 
            raise Exception(f'Can not do grid search on more than 10000 samples, you requested {(num_samples)**(len(parameters))}')
        
        self.parameters = parameters
        self.bounds = bounds
        self.num_samples = self.num_initial_points = num_samples
        self.hypercube_grid = list(self.sample_parameters())
        self.current_index = 0 
        print(self.hypercube_grid)

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