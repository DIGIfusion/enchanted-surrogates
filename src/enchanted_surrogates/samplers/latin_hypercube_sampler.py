"""
---

## Overview

The latin hypercube sampler creates a hypercube grid of equidistant points that spans
bounds and num samples.

---
"""
import numpy as np
from enchanted_surrogates.samplers.base_sampler import Sampler


class LatinHypercubeSampler(Sampler):
    """
    ## Configuration

    To use the `LatinHypercubeSampler`, specify it in the configuration file as in following example:

    ```yaml
      samplers:
        s1:
          type: LatinHypercubeSampler
          parameters: ['c1', 'c2']
          bounds: [[1, 10], [0, 1]]
          budget: 100
          batch_size: 10 # optional

    ```

    Attributes:
        self.budget (int): Total number of samples that can be generated.
        self.bounds (list[tuple[float, float]]): Lower and upper bounds for each parameter.
        self.parameters (list[str]): Names of the parameters to be sampled.
        self.batch_size (int): Number of samples returned per batch (defaults to full budget).
        self.submitted (int): Counter tracking how many samples have been generated so far.

    ---

    ## Assumptions and Notes
     - Parameter values are sampled in such way that each parameter bound range is split into 
      n = `batch_size` gaps, and each parameter gets one value from one gap
     - `LatinHypercubeSampler` generates samples in batches, the batch size is controlled by `batch_size`.
      if `batch_size` is not provided, it defaults to the full sampling `budget`.
     - If the budget is not divisible by `batch_size`, the last batch will have less gaps than other
      batches.
     - `LatinHypercubeSampler` does not adapt based on previous evaluations
     - `LatinHypercubeSampler` maintains an internal counter (`self.submitted`) tracking
      the number of generated samples.
     - The current implementation assumes continuous parameter spaces (`bounds`).

    ---

    """

    def __init__(self, bounds, budget, parameters, **kwargs):
        """
        Initializes the LatinHypercubeSampler.

        Args:
          bounds (list[tuple[float, float]]): Lower and upper bounds for each parameter.
          budget (int): Total number of samples that can be generated.
          parameters (list[str]): Names of the parameters to be sampled. The order must correspond to the order of bounds.
          batch_size (int, optional): Number of samples returned per call to `get_next_samples`. Defaults to the full sampling budget.
        """
        self.budget = budget
        self.bounds = bounds
        self.parameters = parameters
        self.batch_size = kwargs.get("batch_size", self.budget)
        self.submitted = 0

    def get_next_samples(self) -> list[dict]:
        """
        Generates the next batch of parameters according to Latin Hypercube Sampling method.
        Parameter bounds are divided to `batch_size` amount of gaps, and each parameter gets
        assigned random value within the gap. 

        Returns:
           list[dict[str, float]]:
              A batch of parameter dictionaries, where each dictionary maps
              parameter names to sampled numeric values.
        """
        n = min(self.batch_size, self.budget - self.submitted)

        if n <= 0:
            return []

        result = np.zeros((n, len(self.parameters)))
        for dim in range(len(self.parameters)):
            low, high = self.bounds[dim]
            cuts = np.linspace(0,1,n+1) # first do cuts in [0,1] and then scale it
            points = np.random.uniform(cuts[:-1],cuts[1:])
            np.random.shuffle(points)
            result[:,dim] = low + points * (high-low) # scale

        list_param_dicts = [
            {key: result[i, dim] for dim, key in enumerate(self.parameters)}
            for i in range(n)
        ]

        self.submitted += len(list_param_dicts)
        return list_param_dicts

    def register_future(self, future):
        """
        Registers a completed or scheduled evaluation.

        This method is part of the sampler interface but is not used by
        the LatinHypercubeSampler, as sampling does not depend on evaluation results.

        Args:
          future:
              A future or handle representing an asynchronous evaluation.

        Returns:
          None
        """
        return None
