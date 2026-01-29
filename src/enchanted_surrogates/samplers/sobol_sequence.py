import numpy as np
from enchanted_surrogates.samplers.base_sampler import Sampler
import numpy as np
from scipy.stats.qmc import Sobol
import warnings

class SobolSequence(Sampler):
    def __init__(self, bounds, num_samples, parameters, **kwargs):
        self.num_samples = num_samples
        self.power = int(np.log2(self.num_samples))
        if self.num_samples != 2**self.power:
            warnings.warn(f'SOBOL SEQUENCE NUM_SAMPLES MUST BE A POWER OF 2 eg, 2,4,16,32... SETTING NUM_SAMPLES TO {2**self.power}')
        self.num_samples = 2**self.power
        self.num_repeats = kwargs.get("num_repeats", 1)
        self.budget = self.num_samples * self.num_repeats
        
        self.bounds = bounds
        self.parameters = parameters
        self.dim = len(self.parameters)  # Change this for the number of dimensions

        self.batch_size = kwargs.get("batch_size", self.budget)
        self.scramble = kwargs.get("scramble", True)
        self.batch_number = 0
        self.include_index = kwargs.get('include_index', False)
        self.seed = kwargs.get("seed", 42)
        self.fast_forward = kwargs.get('fast_forward', 0)
        
        # must be last
        self.samples = self.generate_samples()
        
        
    def get_next_samples(self) -> list[dict]:
        # TODO not use uniform?
        # TODO batch samples
        samples = [{key: value for key, value in zip(self.parameters, params)} for params in self.samples[self.batch_number*self.batch_size  :  min((self.batch_number+1)*self.batch_size, self.budget)]]
        samples = samples * self.num_repeats
        # samples = [{key: value for key, value in zip(self.parameters, params)} for params in self.samples[self.batch_number*self.batch_size  :  (self.batch_number+1)*self.batch_size]]
        
        self.batch_number += 1
        if self.include_index:
            samples = [
                {**samp, 'index': ind} for samp, ind in zip(samples, range(self.submitted, self.submitted + len(samples)))]
        self.submitted += len(samples)
        return samples
        
    def generate_samples(self):
        print('GENERATING SOBOL SEQUENCE SAMPLES')                
        # Define the dimensionality

        # Define the bounds for each dimension
        lower_bounds = np.array(self.bounds).T[0]
        upper_bounds = np.array(self.bounds).T[1]

        # Create a Sobol sequence generator
        try:
            sobol = Sobol(d=self.dim, scramble=self.scramble, rng=self.seed)
        except:
            sobol = Sobol(d=self.dim, scramble=self.scramble, seed=self.seed)

        if self.fast_forward > 0:
            sobol.fast_forward(self.fast_forward)

        # Generate points in the unit hypercube [0, 1]^d
        points = sobol.random_base2(m=self.power)  # Generates 2^power points

        # Scale the points to the desired bounds
        scaled_points = lower_bounds + points * (upper_bounds - lower_bounds)

        return scaled_points.tolist()


    def register_future(self, future):
        """ Doesn't matter for random sampler TODO: Probably? """
        return None

    def register_futures(self, futures):
        return None
