"""
## Overview

The `SampleWiseJoinSampler` joins multiple sub‑samplers by merging their
sampled configurations *element‑wise*. Each sub‑sampler produces a list of
parameter dictionaries, and this sampler zips those lists together to form
unified sample dictionaries.

This sampler is useful when different parameters must be sampled using
different strategies (e.g., categorical, continuous, grid‑based), while
preserving sample‑wise alignment across batches.

---
"""
from enchanted_surrogates.samplers.base_sampler import Sampler
from enchanted_surrogates.utils.precise_imports import import_sampler
from enchanted_surrogates.utils.logger import get_logger
import numpy as np

log = get_logger(__name__)


class SampleWiseJoinSampler(Sampler):
    """
    ## Configuration

    Example YAML configuration:

    ```yaml
      sampler:
        type: SampleWiseJoinSampler
        samplers:
          sampler_x:
            type: RandomSampler
            parameters: ['x']
            bounds: [[0, 1]]
            num_samples: 10

          sampler_y:
            type: RandomCategoricalSampler
            parameters: ['y']
            categories: [['a', 'b', 'c']]
            num_samples: 10

        fixed_parameters:
          seed: 42
    ```

    Attributes:
        all_samplers (list[Sampler]):
            Instantiated sub‑samplers used to generate partial samples.

        fixed_parameters (dict):
            Key‑value pairs included in every sample.

        parameters (list[str]):
            Combined parameter names from all sub‑samplers plus fixed parameters.

        submitted (int):
            Total number of samples returned so far.

        budget (int):
            Total number of samples that can be produced. May be infinite if
            sub‑samplers are streaming.

    ---

    ## Assumptions and Notes

     - All sub‑samplers must return the **same number of samples per batch**.
     - Samples are merged *sample‑wise* using Python’s `zip`.
     - Fixed parameters are injected into every sample.
     - The sampler supports **multiple batches**, continuing until any
       sub‑sampler reports no remaining budget.
     - Before generating a batch, the sampler checks `sampler.has_budget()`.
       If any sampler is exhausted, a debug message is logged and `None` is returned.
     - The sampler does not adapt based on evaluation results.

    ---
    """

    def __init__(self, samplers, fixed_parameters=None, *args, **kwargs):
        """
        Initializes the SampleWiseJoinSampler.

        Args:
          samplers (dict):
              A mapping of sampler names to their configuration dictionaries.
              Each entry must contain a `type` field and any sampler‑specific
              configuration.

          fixed_parameters (dict, optional):
              Static key‑value pairs included in every sample. Defaults to `{}`.
        """
        self.fixed_parameters = fixed_parameters or {}

        # Instantiate sub‑samplers
        sampler_keys = list(samplers.keys())
        sampler_types = [samplers[k]["type"] for k in sampler_keys]
        sampler_configs = [samplers[k] for k in sampler_keys]

        self.all_samplers = [
            import_sampler(type=stype, sampler_config=config)
            for stype, config in zip(sampler_types, sampler_configs)
        ]

        # Collect parameter names for metadata
        sub_params = [
            p for sampler in self.all_samplers for p in sampler.parameters
        ]
        self.parameters = sub_params + list(self.fixed_parameters.keys())

        self.submitted = 0
        self.budget = np.inf  # streaming by default

    # ---------------------------------------------------------
    # PUBLIC: generate next merged batch
    # ---------------------------------------------------------
    def get_next_samples(self):
        """
        Generates the next batch of merged samples by joining sub‑sampler
        outputs element‑wise.

        Returns:
          list[dict] or None:
              A list of merged sample dictionaries, or `None` if any
              sub‑sampler has no remaining budget.
        """
        # Check budget of all sub‑samplers
        for sampler in self.all_samplers:
            if not sampler.has_budget():
                log.debug(
                    f"Sub‑sampler {sampler.__class__.__name__} has no remaining budget. "
                    "Stopping SampleWiseJoinSampler."
                )
                return None

        # Pull next batch from each sub‑sampler
        sub_batches = [sampler.get_next_samples() for sampler in self.all_samplers]

        # If any sampler is exhausted → stop
        if any(batch is None for batch in sub_batches):
            log.debug(
                "One or more sub‑samplers returned None. "
                "Stopping SampleWiseJoinSampler."
            )
            return None

        # Validate equal lengths
        lengths = [len(b) for b in sub_batches]
        if len(set(lengths)) != 1:
            raise ValueError(
                "All sub‑samplers must return the same number of samples "
                "for sample‑wise joining."
            )

        # Merge sample‑wise
        merged = []
        for group in zip(*sub_batches):
            merged_dict = {**self.fixed_parameters}
            for d in group:
                merged_dict.update(d)
            merged.append(merged_dict)

        self.submitted += len(merged)
        return merged

    # ---------------------------------------------------------
    # No‑op registration hooks
    # ---------------------------------------------------------
    def register_future(self, future):
        """No‑op: joining does not depend on evaluation results."""
        return None

    def register_futures(self, futures):
        """No‑op: joining does not depend on evaluation results."""
        return None
