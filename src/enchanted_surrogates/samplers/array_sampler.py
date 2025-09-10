# sampler/array.py

import numpy as np
from itertools import product
from enchanted_surrogates.samplers.base_sampler import Sampler


class ArraySampler(Sampler):
    """
    Creates combinations of the given samples.

    The samples are directly specified in 'bounds' in the config file.

    Attributes:
        bounds (list of list of float or int): The bounds of each parameter.
        num_samples (list of int): The number of samples for each parameter.
        parameters (list of str): The names of the parameters.
        total_budget (int): The total number of parameter combinations.
        num_initial_points (int): The number of initial points in the sample space.
        samples (list of list of float or int): The generated parameter combinations.
        current_index (int): The index of the current parameter combination.

    For example:
        sampler:
          type: Array
          bounds: [[5, 7, 77, 199], [0.02, 0.2]]
          num_samples: [4, 2]
          parameters: ['a', 'b']

    would create the following sample space:
        [[5, 0.02], [5, 0.2], [7, 0.02], [7, 0.2], [77, 0.02],
        [77, 0.2], [199, 0.02], [199, 0.2]]

    This throws errors if you are asking for something insane,
        e.g., 10 parameters  for 10 samples each -> 10 billion
        so hard limit at 100.000

    Methods:
        generate_parameters: Generates the parameter combinations.
        get_next_parameter: Gets the next parameter combination.

    """
    BATCH_SAMPLE_SIZE = 1

    def __init__(self, bounds, total_budget, parameters, **kwargs):
        """
        Initializes the ArraySampler.

        Args:
            bounds (list of list of float or int): The bounds of each parameter.
            num_samples (list of int): The number of samples for each parameter.
            parameters (list of str): The names of the parameters.
        """

        self.parameters = parameters
        self.bounds = bounds
        self.total_budget = np.prod(np.array([len(b) for b in bounds]))
        if self.total_budget > 100000:
            raise Exception(
                (
                    "Can not do array sampling on more than 10000 samples, ",
                    f"you requested {self.total_budget}",
                )
            )

        self.samples = list(self.generate_parameters())
        self.current_index = 0

    def generate_parameters(self):
        """
        Generates the parameter combinations.

        Yields:
            list of float or int: The next parameter combination.
        """
        samples = self.bounds
        # Use itertools.product to create a Cartesian product of sample
        # points, representing the hypercube
        for params_tuple in product(*samples):
            # Convert tuples to list to ensure serializability
            yield list(params_tuple)
    
    # def get_next_samples(self) -> list[dict]:
    #     """
    #     Gets the next parameter combination.

    #     Returns:
    #         dict or None: The next parameter combination,
    # or None if all combinations have been used.
    #     """
    #     list_param_dicts = []

    #     samples = self.generate_parameters()

    #     if self.current_index < len(samples):
    #         params = samples[self.current_index]
    #         self.current_index += 1
    #         param_dict = {key: value for key, value in zip(self.parameters, params)}
    #         return param_dict
    #     else:
    #         return None  # TODO: implement when done iterating!

    def get_next_samples(self) -> list[dict]:        
        """
        Gets the next batch of parameter combinations.

        Returns:
            list[dict]: List of parameter dicts for the batch,
                        or empty list if no samples remain.
        """
        list_param_dicts = []

        for _ in range(self.BATCH_SAMPLE_SIZE):
            if self.current_index >= len(self.samples):
                break
            params = self.samples[self.current_index]
            self.current_index += 1
            param_dict = {key: value for key, value in zip(self.parameters, params)}
            list_param_dicts.append(param_dict)
        self.submitted += len(list_param_dicts)

        return list_param_dicts

    def register_future(self, future):
        """ Doesn't matter for random sampler TODO: Probably? """
        return None

    def register_futures(self, futures):
        return None
