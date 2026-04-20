""" TODO: Documentation. """

import numpy as np
from scipy.stats.qmc import Sobol
import warnings
from enchanted_surrogates.samplers.base_sampler import Sampler


class SobolSequenceSampler(Sampler):
    """
    TODO: Description.

    """
    def __init__(self, bounds, budget, parameters, **kwargs):
        """Initialize the Sobol sequence sampler.

        Args:
            bounds (list[tuple]): A list of (min, max) tuples defining the
                bounds for each parameter.
            budget (int): The total number of samples to generate. Will be
                adjusted to the nearest power of 2.
            parameters (list[str]): The names of the parameters to sample.
            batch_size (int, optional): Number of samples to return per
                batch. Defaults to budget.
            scramble (bool, optional): Whether to apply scrambling to the
                Sobol sequence. Defaults to True.
            seed (int, optional): Random seed for reproducibility.
                Defaults to 42.
        """
        self.budget = budget
        self.power = int(np.log2(self.budget))
        if self.budget != 2**self.power:
            warnings.warn(f'SOBOL SEQUENCE BUDGET MUST BE A POWER OF 2 eg, 2,4,16,32... SETTING BUDGET TO {2**self.power}')
        self.budget = 2**self.power

        self.bounds = bounds
        self.parameters = parameters
        self.batch_size = kwargs.get("batch_size", self.budget)
        self.scramble = kwargs.get("scramble", True)
        self.batch_number = 0
        self.seed = kwargs.get("seed", 42)
        # must be last
        self.samples = self.generate_samples()

    def get_next_samples(self) -> list[dict]:
        """Get the next batch of samples from the Sobol sequence.

        Returns the next batch_size samples (or fewer if at the end of the
        sequence).
        Each sample is a dictionary mapping parameter names to their values.

        Returns:
            list[dict]: A list of dictionaries, where each dictionary maps
                parameter names to sampled values.
                Example: [{'param1': 0.5, 'param2': 0.3},
                {'param1': 0.7, 'param2': 0.9}]
        """
        # TODO not use uniform?
        # TODO batch samples
        samples = [{key: value for key, value in zip(self.parameters, params)} for params in self.samples[self.batch_number*self.batch_size  :  min((self.batch_number+1)*self.batch_size, self.budget)]]
        # samples = [{key: value for key, value in zip(self.parameters, params)} for params in self.samples[self.batch_number*self.batch_size  :  (self.batch_number+1)*self.batch_size]]

        self.batch_number += 1
        self.submitted += len(samples)
        return samples

    def generate_samples(self):
        """Generate a Sobol sequence of samples within the specified bounds.

        Creates a quasi-random sequence using scipy's Sobol implementation,
        then scales the sequence from [0, 1]^d to the specified bounds for
        each dimension.

        Returns:
            list[list[float]]: A list of samples, where each sample is a list
                of parameter values scaled to the specified bounds. The number
                of samples equals 2^power.
        """
        # Define the dimensionality
        dim = len(self.parameters)  # Change this for the number of dimensions

        # Define the bounds for each dimension
        lower_bounds = np.array(self.bounds).T[0]
        upper_bounds = np.array(self.bounds).T[1]

        # Create a Sobol sequence generator
        try:
            sobol = Sobol(d=dim, scramble=self.scramble, rng=self.seed)
        except:
            sobol = Sobol(d=dim, scramble=self.scramble, seed=self.seed)

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

    def skip(self, index):
        raise NotImplementedError(
            "skip not implemented for SobolSequenceSampler.")
