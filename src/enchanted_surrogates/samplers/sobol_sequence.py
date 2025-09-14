import numpy as np
from enchanted_surrogates.samplers.base_sampler import Sampler
import numpy as np
from scipy.stats.qmc import Sobol


class SobolSequence(Sampler):
    def __init__(self, bounds, budget, parameters, **kwargs):
        self.budget = budget
        self.bounds = bounds
        self.parameters = parameters
        self.batch_size = kwargs.get("batch_size", self.budget)
        self.scramble = kwargs.get("scramble", True)
        self.batch_number = 0
        self.seed = kwargs.get("seed", 42)
        
        
        # must be last
        self.samples = self.generate_samples()
        
    def get_next_samples(self) -> list[dict]:
        # TODO not use uniform?
        # TODO batch samples
        samples = [{key: value for key, value in zip(self.parameters, params)} for params in self.samples[self.batch_number*self.batch_size  :  (self.batch_number+1)*self.batch_size]]
        self.batch_number += 1
        return samples
        
    def generate_samples(self):
                
        # Define the dimensionality
        dim = len(self.parameters)  # Change this for the number of dimensions

        # Define the bounds for each dimension
        lower_bounds = np.array(self.bounds).T[0]
        upper_bounds = np.array(self.bounds).T[1]

        # Create a Sobol sequence generator
        try:
            sobol = Sobol(d=dim, scramble=self.scramble, rng=self.seed)
        except AttributeError:
            sobol = Sobol(d=dim, scramble=self.scramble, seed=self.seed)
        
        power = int(np.log2(self.budget))
        self.num_samples = 2**power
        print(f'''
              GENERATING SOBOL SEQUENCE SAMPLES, NUM SAMPLES REQUESTED: {self.num_samples}, NUM SAMPLES: {2**power}\n
              PARAMETERS: {self.parameters}
              BOUNDS:{self.bounds}''')
        
        # Generate points in the unit hypercube [0, 1]^d
        points = sobol.random_base2(m=power)  # Generates 2^power points

        # Scale the points to the desired bounds
        scaled_points = lower_bounds + points * (upper_bounds - lower_bounds)
        
        return scaled_points.tolist()        


    def register_future(self, future):
        """ Doesn't matter for random sampler TODO: Probably? """
        return None

    def register_futures(self, futures):
        return None
