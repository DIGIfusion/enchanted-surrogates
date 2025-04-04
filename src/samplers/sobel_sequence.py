# sampler/grid.py

from .base import Sampler
import numpy as np
from itertools import product
from common import S
from dask.distributed import print
import numpy as np
from scipy.stats.qmc import Sobol


class SobelSequence(Sampler):
    """
    Sudo random sampler that achieves low discrepancy from uniform with few data points. 

    Attributes:
        bounds (list of tuple of float): The bounds of each parameter.
        num_samples (list of int): The number of samples, approximate as must be a power of two
        parameters (list of str): The names of the parameters.
        samples (list of list of float): The generated parameter combinations.
        current_index (int): The index of the current parameter combination.
        sampler_interface (S): The type of sampler interface.

    Methods:
        get_initial_parameters: Gets the initial parameters.
        generate_parameters: Generates the parameter combinations.
        get_next_parameter: Gets the next parameter combination.
    """

    sampler_interface = S.SEQUENTIAL

    def __init__(self, bounds, num_samples, parameters, *args, **kwargs):
        """
        Initializes the Grid sampler.

        Args:
            bounds (list of tuple of float): The bounds of each parameter.
            num_samples (list of int): The number of samples for each parameter.
            parameters (list of str): The names of the parameters.
        """
        
        self.parameters = parameters
        self.bounds = bounds
        print(self.parameters, self.bounds, len(self.parameters), len(self.bounds))
        if len(self.parameters) != len(self.bounds):
            raise ValueError('The number of bounds and parameters do not match. Please define the same number of bounds as parameters')
        self.num_samples = self.num_initial_points = float(num_samples)
        self.samples = self.generate_parameters()
        self.current_index = 0

    def get_initial_parameters(
        self,
    ):
        """
        Gets the initial parameters.

        Returns:
            list[dict[str, float]]: The initial parameters.
        """
        # self.samples[:self.num_initial_points]
        
        return [self.get_next_parameter() for _ in range(self.num_initial_points)]

    def generate_parameters(self):
        """
        Generates the parameter combinations.

        Yields:
            list of float: The next parameter combination.
        """
        
        # Define the dimensionality
        dim = len(self.parameters)  # Change this for the number of dimensions

        # Define the bounds for each dimension
        lower_bounds = np.array(self.bounds).T[0]
        upper_bounds = np.array(self.bounds).T[1]

        # Create a Sobol sequence generator
        sobol = Sobol(d=dim, scramble=False)

        print('num samp',type(self.num_samples), self.num_samples)
        power = int(np.log2(self.num_samples))
        self.num_samples = self.num_initial_points = 2**power
        print(f'''
              GENERATING SOBOL SEQUENCE SAMPLES, NUM SAMPLES REQUESTED: {self.num_samples}, NUM SAMPLES: {2**power}\n
              PARAMETERS: {self.parameters}
              BOUNDS:{self.bounds}''')
        
        # Generate points in the unit hypercube [0, 1]^d
        points = sobol.random_base2(m=power)  # Generates 2^power points

        # Scale the points to the desired bounds
        scaled_points = lower_bounds + points * (upper_bounds - lower_bounds)
        
        return scaled_points.tolist()        
        
    def get_next_parameter(self):
        """
        Gets the next parameter combination.

        Returns:
            dict or None: The next parameter combination, or None if all combinations have been used.
        """
        if self.current_index < len(self.samples):
            params = self.samples[self.current_index]
            self.current_index += 1
            param_dict = {key: value for key, value in zip(self.parameters, params)}
            return param_dict
        else:
            return None  # TODO: implement when done iterating!
