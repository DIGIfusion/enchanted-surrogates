# sampler/grid.py

from .base import Sampler
import numpy as np
from itertools import product

class Grid(Sampler):
    """ 
    Creates a grid of equidistant points that spans bounds and num samples 
    i.e., (num_samples)**(len(parameters))

    This throws errors if you are asking for something insane, 
        e.g., 10 parameters  for 10 samples each -> 10 billion 
        so hard limit at 100.000

    """
    def __init__(self, bounds, num_samples, parameters):
        # check for stupidity
        self.total_budget = (num_samples)**(len(parameters))
        if (num_samples)**(len(parameters)) > 100000: 
            raise Exception(f'Can not do grid search on more than 10000 samples, you requested {(num_samples)**(len(parameters))}')
        
        self.parameters = parameters
        self.bounds = bounds
        self.num_samples = self.num_initial_points = num_samples
        self.samples = list(self.generate_parameters())
        self.current_index = 0 

    def generate_parameters(self):
        samples = [np.linspace(bound[0], bound[1], self.num_samples) for bound in self.bounds]
        # Use itertools.product to create a Cartesian product of sample points, representing the hypercube
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
            return None # TODO: implement when done iterating! 