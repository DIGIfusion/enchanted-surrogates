# sampler/grid.py

from .base import Sampler
import numpy as np
from itertools import product
from common import S


class Grid(Sampler):
    """
    Creates a grid of equidistant points that spans bounds and num samples.
    i.e., (num_samples)**(len(parameters))

    Attributes:
        bounds (list of tuple of float): The bounds of each parameter.
        num_samples (list of int): The number of samples for each parameter.
        parameters (list of str): The names of the parameters.
        total_budget (int): The total number of parameter combinations.
        num_initial_points (int): The number of initial points in the sample space.
        samples (list of list of float): The generated parameter combinations.
        current_index (int): The index of the current parameter combination.
        sampler_interface (S): The type of sampler interface.

    This throws errors if you are asking for something insane,
        e.g., 10 parameters for 10 samples each -> 10 billion
        so hard limit at 100.000

    Methods:
        get_initial_parameters: Gets the initial parameters.
        generate_parameters: Generates the parameter combinations.
        get_next_parameter: Gets the next parameter combination.
    """

    sampler_interface = S.SEQUENTIAL

    def __init__(self, bounds, num_samples, parameters):
        """
        Initializes the Grid sampler.

        Args:
            bounds (list of tuple of float): The bounds of each parameter.
            num_samples (list of int): The number of samples for each parameter.
            parameters (list of str): The names of the parameters.
        """
        if isinstance(num_samples, int):
            num_samples = [num_samples] * len(parameters)

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
        samples = [
            np.linspace(self.bounds[i][0], self.bounds[i][1], self.num_samples[i])
            for i in range(len(self.bounds))
        ]
        # Use itertools.product to create a Cartesian product of sample
        # points, representing the hypercube
        for params_tuple in product(*samples):
            # Convert tuples to list to ensure serializability
            yield list(params_tuple)

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
