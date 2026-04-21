"""
## Overview

The `RandomCategoricalSampler` generates random categorical configurations by
independently sampling from a predefined set of allowed values for each parameter.

This sampler is useful for discrete or symbolic search spaces where each parameter
has a finite set of valid categories.

---
"""
import numpy as np
from enchanted_surrogates.samplers.base_sampler import Sampler


class RandomCategoricalSampler(Sampler):
    """
    ## Configuration

    To use the `RandomCategoricalSampler`, specify it in the configuration file:

    ```yaml
      sampler:
        type: RandomCategoricalSampler
        parameters: ['optimizer', 'activation']
        categories:
          - ['sgd', 'adam', 'rmsprop']
          - ['relu', 'tanh', 'gelu']
        num_samples: 50
        num_repeats: 1
        include_index: true
    ```

    Attributes:
        parameters (list[str]):
            Names of the parameters to be sampled.

        categories (list[list[Any]]):
            Allowed categorical values for each parameter. The order must match
            the order of `parameters`.

        num_samples_requested (int):
            Number of samples drawn per batch.

        num_repeats (int):
            Number of times each sampled configuration is duplicated.

        include_index (bool):
            Whether to attach a monotonically increasing integer index to each
            returned sample.

        batch_size (int):
            Number of samples drawn per call to `get_next_samples`. Defaults to
            `num_samples_requested`.

        batch_number (int):
            Counter tracking how many batches have been generated.

        submitted (int):
            Total number of samples returned so far.

        global_index (int):
            Persistent counter used when `include_index=True` to ensure that
            each sample across all batches receives a unique index.

    ---

    ## Assumptions and Notes

     - Each parameter is sampled independently from its categorical set using
       `np.random.choice`.

     - The sampler supports repeated batches; each call to `get_next_samples`
       produces a fresh random batch.

     - If `include_index=True`, the sampler attaches a unique, monotonically
       increasing index to each sample across all batches.

     - The sampler does not adapt based on evaluation results and does not
       require feedback from futures.

    ---
    """

    def __init__(self, parameters, categories, num_samples, **kwargs):
        """
        Initializes the RandomCategoricalSampler.

        Args:
          parameters (list[str]):
              Names of the parameters to be sampled.

          categories (list[list[Any]]):
              Allowed categorical values for each parameter.

          num_samples (int):
              Number of samples to draw per batch.

          num_repeats (int, optional):
              Number of times to duplicate each sampled configuration.
              Defaults to 1.

          include_index (bool, optional):
              Whether to attach a unique index to each sample. Defaults to False.

          batch_size (int, optional):
              Number of samples returned per call to `get_next_samples`.
              Defaults to `num_samples`.
        """
        self.parameters = parameters
        self.categories = categories
        self.num_samples_requested = num_samples

        self.num_repeats = kwargs.get("num_repeats", 1)
        self.include_index = kwargs.get("include_index", False)
        self.batch_size = kwargs.get("batch_size", num_samples)

        self.batch_number = 0
        self.submitted = 0

        # Persistent index across all batches
        self.global_index = 0

    # ---------------------------------------------------------
    # INTERNAL: generate a full batch (draw + format)
    # ---------------------------------------------------------
    def _generate_batch(self):
        """
        Draws a fresh batch of categorical samples and formats them into
        a list of dictionaries. If indexing is enabled, each sample receives
        a unique, monotonically increasing index.

        Returns:
          list[dict]:
              A list of sampled parameter configurations.
        """
        # Draw raw samples (shape: batch_size × num_parameters)
        raw = np.column_stack([
            np.random.choice(cat, size=self.batch_size, replace=True)
            for cat in self.categories
        ])

        # Convert to list of dicts
        samples = [
            {key: value for key, value in zip(self.parameters, row)}
            for row in raw
        ] * self.num_repeats

        # Apply persistent indexing
        if self.include_index:
            indexed = []
            for s in samples:
                indexed.append({**s, "index": self.global_index})
                self.global_index += 1
            samples = indexed

        return samples

    # ---------------------------------------------------------
    # PUBLIC: always generate a new batch
    # ---------------------------------------------------------
    def get_next_samples(self):
        """
        Generates the next batch of categorical samples.

        Returns:
          list[dict]:
              A batch of parameter dictionaries, optionally including a
              persistent index field.
        """
        samples = self._generate_batch()
        self.submitted += len(samples)
        self.batch_number += 1
        return samples

    # Compatibility no-ops
    def register_future(self, future):
        """No-op: Random sampling does not depend on evaluation results."""
        return None

    def register_futures(self, futures):
        """No-op: Random sampling does not depend on evaluation results."""
        return None
