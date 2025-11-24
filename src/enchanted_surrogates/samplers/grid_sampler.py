# samplers/grid_sampler.py

import numpy as np
from itertools import product
from enchanted_surrogates.samplers.base_sampler import Sampler


class GridSampler(Sampler):
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

    def __init__(self, bounds, num_samples, parameters, *args, **kwargs):
        """
        Initializes the Grid sampler.

        Args:
            bounds (list of tuple of float): The bounds of each parameter.
            num_samples (list of int): The number of samples for each parameter.
            parameters (list of str): The names of the parameters.
        """
        super().__init__()

        if isinstance(num_samples, int):
            num_samples = [num_samples] * len(parameters)

        self.parameters = parameters
        self.bounds = bounds
        self.num_samples = num_samples
        # check for stupidity
        self.batch_size = kwargs.get("batch_size", self.budget)

        self.num_repeats = kwargs.get('num_repeats', 1)
        self.budget = np.prod(np.array(num_samples))*self.num_repeats
        
        self.include_index = kwargs.get('include_index', False)
        
        if self.budget > 100000:
            raise Exception(
                (
                    "Can not do grid search on more than 10000 samples, ",
                    f"you requested {self.budget}",
                )
            )

        self.samples = list(self.generate_parameters())
        self.current_index = 0

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

    def get_next_samples(self) -> list[dict]:
        """
        Gets the next batch of parameter combinations.

        Returns:
            list[dict]: A batch of parameter combinations.
        """
        list_param_dicts = []

        for _ in range(self.batch_size):
            if self.current_index >= len(self.samples):
                break
            params = self.samples[self.current_index]
            self.current_index += 1
            param_dict = {k: v for k, v in zip(self.parameters, params)}
            list_param_dicts.append(param_dict)

        list_param_dicts = list_param_dicts * self.num_repeats

        if self.include_index:
            list_param_dicts = [
                {**samp, 'index': ind} for samp, ind in zip(list_param_dicts, range(self.submitted, self.submitted + len(list_param_dicts)))]

        self.submitted += len(list_param_dicts)
        return list_param_dicts

    def register_future(self, future):
        """Doesn't matter for grid sampler."""
        return None

    def register_futures(self, futures):
        return None
