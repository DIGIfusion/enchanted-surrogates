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
    ...
    ## Additional Notes

     - If sub‑samplers declare different `batch_size` values, an error is raised.
     - The sampler's `budget` is set to the **minimum** budget among all sub‑samplers.
     - If `remove_non_unique=True`, duplicate merged samples are removed
       after joining. A warning is logged if any are removed.
    ...
    """

    def __init__(self, samplers, fixed_parameters=None, remove_non_unique=False, *args, **kwargs):
        """
        Initializes the SampleWiseJoinSampler.

        Args:
          samplers (dict):
              Mapping of sampler names to configuration dictionaries.

          fixed_parameters (dict, optional):
              Static key‑value pairs included in every sample.

          remove_non_unique (bool, optional):
              If True, duplicate merged samples are removed after joining.
              Defaults to False.
        """
        self.fixed_parameters = fixed_parameters or {}
        self.remove_non_unique = remove_non_unique

        # Instantiate sub‑samplers
        sampler_keys = list(samplers.keys())
        sampler_types = [samplers[k]["type"] for k in sampler_keys]
        sampler_configs = [samplers[k] for k in sampler_keys]

        self.all_samplers = [
            import_sampler(sampler_type=stype, sampler_config=config)
            for stype, config in zip(sampler_types, sampler_configs)
        ]

        # Validate batch sizes
        batch_sizes = []
        for sampler in self.all_samplers:
            if not hasattr(sampler, "batch_size"):
                raise AttributeError(
                    f"Sub‑sampler {sampler.__class__.__name__} must define `batch_size`."
                )
            batch_sizes.append(sampler.batch_size)

        if len(set(batch_sizes)) != 1:
            raise ValueError(
                "All sub‑samplers must define the same `batch_size` for sample‑wise joining. "
                f"Found batch sizes: {batch_sizes}"
            )

        self.batch_size = batch_sizes[0]

        # Collect parameter names
        sub_params = [p for sampler in self.all_samplers for p in sampler.parameters]
        self.parameters = sub_params + list(self.fixed_parameters.keys())

        # Budget = minimum of all sampler budgets
        sampler_budgets = []
        for sampler in self.all_samplers:
            if not hasattr(sampler, "budget"):
                raise AttributeError(
                    f"Sub‑sampler {sampler.__class__.__name__} must define `budget`."
                )
            sampler_budgets.append(sampler.budget)

        self.budget = min(sampler_budgets)
        self.submitted = 0


    # ---------------------------------------------------------
    # PUBLIC: generate next merged batch
    # ---------------------------------------------------------
    def get_next_samples(self):
        """
        Generates the next batch of merged samples by joining sub‑sampler
        outputs element‑wise.

        If `remove_non_unique=True`, duplicate merged samples are removed.
        A warning is logged if any are removed.

        Returns:
          list[dict] or None:
              A list of merged sample dictionaries, or `None` if any
              sub‑sampler has no remaining budget.
        """
        # Check budget
        for sampler in self.all_samplers:
            if not sampler.has_budget:
                log.debug(
                    f"Sub‑sampler {sampler.__class__.__name__} has no remaining budget. "
                    "Stopping SampleWiseJoinSampler."
                )
                return None

        # Pull next batch
        sub_batches = [sampler.get_next_samples() for sampler in self.all_samplers]

        if any(batch is None for batch in sub_batches):
            log.debug("One or more sub‑samplers returned None. Stopping.")
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

        # Remove duplicates if requested
        if self.remove_non_unique:
            before = len(merged)
            unique = []
            seen = set()

            for sample in merged:
                key = tuple(sorted(sample.items()))
                if key not in seen:
                    seen.add(key)
                    unique.append(sample)

            removed = before - len(unique)
            if removed > 0:
                log.warning(
                    f"Removed {removed} duplicate merged samples "
                    "due to `remove_non_unique=True`."
                )

            merged = unique

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
