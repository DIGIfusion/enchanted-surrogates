"""
---

## Overview

The random sampler generates samples randomly within the specified bounds
for each parameter.

---
"""
import numpy as np
from enchanted_surrogates.samplers.base_sampler import Sampler


class RandomSampler(Sampler):
    """
    ## Configuration

    To use the `RandomSampler`, specify it in the configuration file as in following example:

    ```yaml
      sampler:
        type: RandomSampler
        parameters: ['x', 'y']
        bounds: [[1, 10], [0, 1]]
        num_samples: 100
    ```

    Attributes:
        self.budget (int): Total number of samples that can be generated.
        self.bounds (list[tuple[float, float]]): Lower and upper bounds for each parameter.
        self.parameters (list[str]): Names of the parameters to be sampled.
        self.batch_size (int): Number of samples returned per batch (defaults to full budget).
        self.submitted (int): Counter tracking how many samples have been generated so far.
        BATCH_SAMPLE_SIZE (int): Class-level default batch size (currently 1).

    ---

    ## Assumptions and Notes

     - Parameter values are sampled independently using a uniform distribution
      (`np.random.uniform`) within the specified bounds.

     - The sampler generates samples in batches; the batch size is controlled
      by `batch_size`.

     - If `batch_size` is not provided, it defaults to the full sampling budget.
     - The sampler does not adapt based on previous evaluations.

     - The sampler maintains an internal counter (`self.submitted`) tracking
      the number of generated samples.

     - The current implementation assumes continuous parameter spaces.

    ---

    """

    BATCH_SAMPLE_SIZE = 1

    def __init__(self, bounds, budget, parameters, **kwargs):
        """
        Initializes the RandomSampler.

        Args:
          bounds (list[tuple[float, float]]): Lower and upper bounds for each parameter.
          budget (int): Total number of samples that can be generated.
          parameters (list[str]): Names of the parameters to be sampled.
              The order must correspond to the order of bounds.
          batch_size (int, optional): Number of samples returned per call
              to `get_next_samples`. Defaults to the full sampling budget.
          distribution (str, optional): The distribution used by the sampler.
              Options: uniform (default), loguniform, normal, int_uniform. 
        """
        self.budget = budget
        self.bounds = bounds
        self.parameters = parameters
        self.batch_size = kwargs.get("batch_size", self.budget)
        self.distribution = kwargs.get("distribution", "uniform")

    def get_next_samples(self) -> list[dict]:
        """
        Generates the next batch of randomly sampled parameter configurations.

        Each parameter value is sampled independently from a uniform distribution
        within the specified bounds.

        Returns:
           list[dict[str, float]]:
              A batch of parameter dictionaries, where each dictionary maps
              parameter names to sampled numeric values.
        """
        # TODO batch samples
        list_param_dicts = []
        for _ in range(self.batch_size):
            params = [
              self.sample_from_distribution(low, high)
              for low, high in self.bounds
            ]
            param_dict = {key: value for key, value in zip(self.parameters, params)}
            list_param_dicts.append(param_dict)
        self.submitted += len(list_param_dicts)
        return list_param_dicts

    def register_future(self, future):
        """
        Registers a completed or scheduled evaluation.

        This method is part of the sampler interface but is not used by
        the RandomSampler, as sampling does not depend on evaluation results.

        Args:
          future:
              A future or handle representing an asynchronous evaluation.

        Returns:
          None
        """
        return None

    def register_futures(self, futures):
        """
        Registers multiple completed or scheduled evaluations.

        This method is part of the sampler interface but is not used by
        the RandomSampler. It is implemented as a no-op.

        Args:
          futures:
            An iterable of futures or handles representing asynchronous
            evaluations.

        Returns:
          None
        """
        return None

    def sample_from_distribution(self, low, high):
        """ Sample from different distributions. 
        
        Args:
          low:
            A float or integer defining the lower boundary of the distribution.
          high:
            A float or integer defining the higher boundary of the distribution.

        Returns:
          sample:
            A float or integer.
        """
        if self.distribution == "uniform":
            return np.random.uniform(low, high)

        elif self.distribution == "loguniform":
            # useful when parameters span orders of magnitude
            return np.exp(np.random.uniform(np.log(low), np.log(high)))

        elif self.distribution == "normal":
            # interpret bounds as 3std limits (common convention)
            mean = (low + high) / 2
            std = (high - low) / 6
            return np.random.normal(mean, std)

        elif self.distribution == "int_uniform":
            return np.random.randint(low, high + 1)

        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")
