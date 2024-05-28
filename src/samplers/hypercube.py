# sampler/hypercube.py

import numpy as np
from .base import Sampler
from itertools import product
from common import S


class HypercubeSampler(Sampler):
    """
    Creates a hypercube grid of equidistant points that spans bounds and num samples.
    TODO: Is this implemenmtation finished?

    Attributes:
        bounds (list of tuple of float): The bounds of each parameter.
        num_samples (int): The number of samples for each parameter.
        parameters (list of str): The names of the parameters.
        total_budget (int): The total number of parameter combinations.
        num_initial_points (int): The number of initial points in the sample space.
        hypercube_grid (list of list of float): The generated hypercube grid.
        current_index (int): The index of the current parameter combination.
        sampler_interface (S): The type of sampler interface.

    This throws errors if you are asking for something insane,
        e.g., 10 parameters for 10 samples each -> 10 billion
        so hard limit at 100.000

    Methods:
        sample_parameters: Samples parameters from the hypercube grid.
        get_initial_parameters: Gets the initial parameters.
        get_next_parameter: Gets the next parameter combination.
    """

    sampler_interface = S.SEQUENTIAL

    def __init__(self, bounds, num_samples, parameters):
        """
        Initializes the HypercubeSampler.

        Args:
            bounds (list of tuple of float): The bounds of each parameter.
            num_samples (int): The number of samples for each parameter.
            parameters (list of str): The names of the parameters.
        """
        # check for stupidity
        self.total_budget = (num_samples) ** (len(parameters))
        if (num_samples) ** (len(parameters)) > 100000:
            raise Exception(
                (
                    "Can not do grid search on more than 10000 samples, ",
                    f"you requested {(num_samples)**(len(parameters))}",
                )
            )

        self.parameters = parameters
        self.bounds = bounds
        self.num_samples = self.num_initial_points = num_samples
        self.hypercube_grid = list(self.sample_parameters())
        self.current_index = 0
        print(self.hypercube_grid)

    def sample_parameters(self):
        """
        Samples parameters from the hypercube grid.

        Yields:
            list of float: The next parameter combination.
        """
        samples = [
            np.linspace(bound[0], bound[1], self.num_samples) for bound in self.bounds
        ]
        # Use itertools.product to create a Cartesian product of sample
        # points, representing the hypercube
        for params_tuple in product(*samples):
            # Convert tuples to list to ensure serializability
            yield list(params_tuple)

    def get_initial_parameters(
        self,
    ):
        """
        Gets the initial parameters.

        Returns:
            list[dict[str, float]]: The initial parameters.
        """
        return [self.get_next_parameter() for _ in range(self.num_initial_points)]

    def get_next_parameter(self):
        """
        Gets the next parameter combination.

        Returns:
            dict or None: The next parameter combination, or None if all combinations have been used.
        """
        if self.current_index < len(self.hypercube_grid):
            params = self.hypercube_grid[self.current_index]
            param_dict = {key: value for key, value in zip(self.parameters, params)}
            self.current_index += 1
            return param_dict
        else:
            # Handle the case where all parameters have been iterated over
            return 1
