# sampler/grid.py
"""
## Overview
`NestedSampler` composes multiple samplers into a single sampler by
generating the Cartesian product of their initial sample sets.

Each nested sampler is initialized independently, and all combinations
of their initial samples are merged into joint parameter configurations.
The sampler produces exactly one batch of samples and does not support
iterative or adaptive sampling.

---
"""

# from .base import Sampler
from itertools import product
import numpy as np
import importlib
from enchanted_surrogates.utils.precise_imports import import_sampler
from enchanted_surrogates.samplers.base_sampler import Sampler


class NestedSampler(Sampler):
    """
    ## Configuration

    The `NestedSampler` is configured by specifying a dictionary of sub-samplers,
    each with its own type and configuration.

    Example configuration:

    ```yaml
    sampler:
      type: NestedSampler
      samplers:
        sampler_a:
          type: GridSampler
          parameters: ['x']
          bounds: [[0, 1]]
          num_samples: [4, 3]

        sampler_b:
          type: RandomSampler
          parameters: ['y', 'z']
          bounds: [[0, 1], [10, 20]]
          num_samples: 2

      budget: 24
    ```
    In this example:

      - `sampler_a` generates 12 samples for `(x)`

      - `sampler_b` generates 2 samples for `(y, z)`

      - The nested sampler produces `12 × 2 = 24` combined configurations.
    

    Attributes:
        all_samplers (list[Sampler]): List of instantiated nested samplers.
        budget (int): Total number of samples allowed across the nested sampling process.
        batch_size (int): Number of samples returned per call to `get_next_samples`.
        submitted (int): Counter tracking the number of generated combined samples.
        all_parameters (list[str]): Flattened list of parameter names from all nested samplers.
        initial_parameters (list[list[dict]]): Initial samples collected from each nested sampler.
        current_batch (int): Tracks whether the first (and only) batch has been returned.

    ---

    ## Assumptions and notes

    - Each nested sampler must implement `get_next_samples()` and expose a `parameters` attribute.
    - All samples are generated eagerly during initialization.
    - The total number of generated samples grows multiplicatively with the number of nested samplers.

    ---

    """

    def __init__(self, samplers, *args, **kwargs):

        """
        Initializes the NestedSampler and all nested samplers.

        Each nested sampler is instantiated using its provided configuration,
        and its initial samples are collected immediately. The full Cartesian
        product of these samples defines the nested search space.

        Args:
            samplers (dict): Mapping of sampler names to sampler configurations.
                Each configuration must specify the sampler type and its
                corresponding parameters.
            budget (int, optional): Total number of samples allowed across the
                nested sampling process.
            batch_size (int, optional): Number of samples returned per batch.
                Defaults to the total budget.

        """
        samplers_keys = samplers.keys
        samplers_types = [samplers[k]['type'] for k in samplers_keys()]
        samplers_config = [samplers[k] for k in samplers_keys()]
        self.all_samplers = [import_sampler(sampler_type, sampler_config) for sampler_type, sampler_config in zip(samplers_types, samplers_config)]
        
        default_budget = 1
        for sampler in self.all_samplers:
            default_budget = default_budget * sampler.budget
        self.budget = kwargs.get('budget', default_budget)
        self.batch_size = kwargs.get('batch_size', self.budget)
        self.submitted = 0
        self.all_parameters = [param for sampler in self.all_samplers for param in sampler.parameters]

        self.initial_parameters = []
        for sampler in self.all_samplers:
            self.initial_parameters.append(sampler.get_next_samples())

        # Split first sampler (batched) from the rest (static)
        self._first_sampler_samples = self.initial_parameters[0]
        self._other_sampler_samples = self.initial_parameters[1:]

        # Precompute Cartesian product of all non-first samplers
        if len(self._other_sampler_samples) > 0:
            self._other_product = [
                dict(kv for d in combo for kv in d.items())
                for combo in product(*self._other_sampler_samples)
            ]
        else:
            self._other_product = [dict()]  # identity element

        self._num_combinations = len(self._all_combinations)
        self._cursor = 0

        self.current_batch = 0 
        
    def get_next_samples(self):
        """
        Prepares batching structures for nested sampling.

        This setup enables batching over the first nested sampler while
        keeping all other samplers static. The first sampler's initial
        samples are consumed in batches, and each batch is combined with
        the full Cartesian product of the remaining samplers' initial
        samples. This guarantees that no two batches reuse the same
        parameter values from the first sampler.

        Attributes initialized:
            _first_sampler_samples (list[dict]):
                Initial samples from the first nested sampler. These are
                consumed in batches according to `batch_size`.

            _other_sampler_samples (list[list[dict]]):
                Initial samples from all remaining samplers.

            _other_product (list[dict]):
                Precomputed Cartesian product of all non-first samplers'
                initial samples. Acts as a static multiplier for each
                batch of the first sampler.

            _cursor (int):
                Index into `_first_sampler_samples` indicating how many
                samples have already been consumed.

        Notes:
            - This batching strategy ensures uniqueness of the first
              sampler's parameters across batches.
            - The total number of returned configurations remains equal
              to the full Cartesian product of all samplers.
        """

        # If first sampler exhausted → stop
        if self._cursor >= len(self._first_sampler_samples):
            return None

        # Slice batch from first sampler
        start = self._cursor
        end = min(self._cursor + self.batch_size, len(self._first_sampler_samples))
        first_batch = self._first_sampler_samples[start:end]

        # Cartesian product: (first_batch) × (other_product)
        batch = []
        for a in first_batch:
            for rest in self._other_product:
                batch.append({**a, **rest})

        self._cursor = end
        self.submitted += len(batch)

        return batch
    
    def register_future(self, future):
        """
        Registers a completed or scheduled evaluation.

        This method is part of the sampler interface but is not used by the
        NestedSampler, as it does not adapt based on evaluation results.

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

        This method is part of the sampler interface but is implemented as
        a no-op for the NestedSampler.

        Args:
            futures:
              An iterable of futures or handles representing evaluations.

        Returns:
            None
        """

        return None
