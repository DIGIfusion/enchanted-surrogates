# samplers/grid_sampler.py
"""
## Overview

Creates a grid of equidistant points that spans bounds and num samples.
i.e., (num_samples)**(len(parameters))

---
"""
import numpy as np
from itertools import product
from enchanted_surrogates.samplers.base_sampler import Sampler


class GridSampler(Sampler):
    """
    ## Configuration

    To use the grid sampler, you need to specify it in the configuration file as follows:

    ```yaml
       sampler:
          type: GridSampler
          parameters: ['x', 'y']
          bounds: [[1, 10], [0, 1]]
          num_samples: [4, 3]
    ```
    In this configuration:
    - parameter x is sampled at 4 evenly spaced points between 1 and 10
    - parameter y is sampled at 3 evenly spaced points between 0 and 1
    - resulting in a total of 4 × 3 = 12 samples.


    Attributes:
        bounds (list of tuple of float): The bounds of each parameter.
        num_samples (list of int): The number of samples for each parameter.
        parameters (list of str): The names of the parameters.
        total_budget (int): The total number of parameter combinations.
        num_initial_points (int): The number of initial points in the sample space.
        samples (list of list of float): The generated parameter combinations.
        current_index (int): The index of the current parameter combination.
        sampler_interface (S): The type of sampler interface.

    ---

    ## Assumptions and Notes

      - The sampler assumes continuous numeric parameters.
      - Parameter values are generated using numpy.linspace, resulting in evenly spaced points that include both bounds. The total number of samples (budget) is the product of num_samples across all parameters.
      - Grid size grows exponentially with the number of parameters; careful configuration is recommended. This throws errors if you are asking for something insane, e.g., 10 parameters for 10 samples each -> 10 billion. To prevent excessive memory usage, the sampler enforces a hard limit of 100,000 total samples.

    ---

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
        self.budget = np.prod(np.array(num_samples))
        self.batch_size = kwargs.get("batch_size", self.budget)

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

        self.submitted += len(list_param_dicts)
        return list_param_dicts

    def register_future(self, future):
        """Doesn't matter for grid sampler."""
        return None

    def register_futures(self, futures):
        return None
