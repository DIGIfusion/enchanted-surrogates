"""
## Overview

`ConstrainedSobolSampler` extends the plain Sobol sequence sampler with:
- inequality constraints between named parameters (e.g. pedestal-top density
  must stay above the separatrix density),
- categorical parameters (e.g. an impurity species drawn from a fixed set of
  choices rather than a continuous range),
- and derived parameters computed from the sampled ones, for anything that
  should stay purely a function of this sampler's own bounds/choices (not
  for values that depend on external config, like a reference quantity
  defined in a downstream runner's parameter file - those are better
  derived by that runner/parser, which actually has the reference value in
  scope; see GeneParser.write_input_file's zimp -> mass_imp handling for an
  example of that pattern).

Points violating a constraint are rejected; the underlying Sobol pool is
doubled in size and redrawn until `budget` valid points are found, up to
`max_pool_size`.
"""

import warnings

import numpy as np
from scipy.stats.qmc import Sobol

from enchanted_surrogates.samplers.base_sampler import Sampler


class ConstrainedSobolSampler(Sampler):
    """
    ## Configuration

    ```yaml
    helena_sampler:
      type: constrained_sobol_sampler
      parameters: ['T_eped', 'n_eped', 'n_esep', 'Ti_ov_Te_ped', 'Ti_ov_Te_sep']
      bounds: [[0.4, 1.5], [3, 11], [1.5, 6], [0.7, 1.7], [2.5, 8]]
      budget: 16
      seed: 42
      constraints:
        - "n_eped - n_esep > 0"
        - "T_eped * Ti_ov_Te_ped - 0.104 * Ti_ov_Te_sep > 0"
      categorical_parameters:
        zimp: [4, 10]   # dominant impurity charge: beryllium (4) or neon (10)
    ```

    `budget` is simply the number of valid samples you want back — unlike
    `SobolSequenceSampler` it does not need to be a power of 2. Internally,
    the sampler draws a Sobol pool of `budget` points, filters it through
    `constraints`, and if that isn't enough, doubles the pool size and
    redraws (discarding the smaller pool) until it has `budget` valid points
    or the pool reaches `max_pool_size`.

    `categorical_parameters` is a dict mapping a parameter name to a list of
    discrete choices. Each such parameter consumes one extra Sobol dimension
    (appended after `parameters`/`bounds`), which is binned into
    `len(choices)` equal-width intervals over [0, 1] to pick a choice —
    keeping the whole draw on a single low-discrepancy sequence rather than
    mixing in independent RNG. Categorical values are included in the dicts
    returned by `get_next_samples` alongside the continuous parameters.

    `constraints` is a list of python expressions evaluated with the sampled,
    categorical, and derived parameter names bound in the local namespace; a
    point is kept only if every expression evaluates truthy.

    `derived_parameters` is a dict mapping a new parameter name to a python
    expression (using the sampled/categorical parameter names, and `np` for
    numpy) that is evaluated once per candidate point, before constraints
    are checked. Derived values are included in the dicts returned by
    `get_next_samples` alongside the sampled parameters.

    Attributes mirror `SobolSequenceSampler`, plus:

    self.constraints (list[str]):
        Expressions (using parameter/derived-parameter names) that must all
        be truthy for a candidate point to be accepted.

    self.categorical_parameters (dict[str, list]):
        Name -> list of discrete choices, sampled uniformly via one extra
        Sobol dimension each.

    self.derived_parameters (dict[str, str]):
        Expressions computing additional parameters from the sampled ones.

    self.max_pool_size (int):
        Upper limit on how large the internal Sobol pool (rounded up to the
        nearest power of 2) is allowed to grow while searching for enough
        valid points. Defaults very high since it rarely needs tuning; only
        very tight constraints should ever approach it. If reached without
        finding `budget` valid points, a warning is raised and however many
        valid points were found are returned.
    """

    def __init__(self, bounds, budget, parameters, **kwargs):
        self.budget = budget
        self.bounds = bounds
        self.parameters = parameters
        self.batch_size = kwargs.get("batch_size", self.budget)
        self.scramble = kwargs.get("scramble", True)
        self.batch_number = 0
        self.seed = kwargs.get("seed", 42)
        self.constraints = kwargs.get("constraints", [])
        self.categorical_parameters = kwargs.get("categorical_parameters", {})
        self.derived_parameters = kwargs.get("derived_parameters", {})
        self.max_pool_size = kwargs.get("max_pool_size", 2**24)
        # must be last
        self.samples = self.generate_samples()

    def _with_derived(self, sample: dict) -> dict:
        sample = dict(sample)
        for name, expr in self.derived_parameters.items():
            sample[name] = eval(expr, {"__builtins__": {}, "np": np}, dict(sample))
        return sample

    def _satisfies_constraints(self, sample: dict) -> bool:
        for expr in self.constraints:
            if not eval(expr, {"__builtins__": {}, "np": np}, dict(sample)):
                return False
        return True

    def get_next_samples(self) -> list[dict]:
        samples = self.samples[
            self.batch_number * self.batch_size : min(
                (self.batch_number + 1) * self.batch_size, self.budget
            )
        ]
        self.batch_number += 1
        self.submitted += len(samples)
        return samples

    def generate_samples(self):
        """Draw Sobol points, resolve categorical choices, compute derived
        parameters, and keep those satisfying the configured constraints,
        doubling the pool size and redrawing from scratch until `budget`
        valid points are found or `max_pool_size` is reached.
        """
        cat_names = list(self.categorical_parameters.keys())
        dim = len(self.parameters) + len(cat_names)
        lower_bounds = np.array(self.bounds).T[0]
        upper_bounds = np.array(self.bounds).T[1]

        pool_power = int(np.ceil(np.log2(max(1, self.budget))))
        max_pool_power = int(np.ceil(np.log2(max(1, self.max_pool_size))))

        while True:
            try:
                sobol = Sobol(d=dim, scramble=self.scramble, rng=self.seed)
            except TypeError:
                sobol = Sobol(d=dim, scramble=self.scramble, seed=self.seed)

            points = sobol.random_base2(m=pool_power)
            continuous_points = points[:, : len(self.parameters)]
            scaled_points = lower_bounds + continuous_points * (
                upper_bounds - lower_bounds
            )
            categorical_points = points[:, len(self.parameters) :]

            valid_samples = []
            for point, cat_point in zip(scaled_points, categorical_points):
                sample = dict(zip(self.parameters, point))
                for name, unit_value in zip(cat_names, cat_point):
                    choices = self.categorical_parameters[name]
                    idx = min(int(unit_value * len(choices)), len(choices) - 1)
                    sample[name] = choices[idx]
                sample = self._with_derived(sample)
                if self._satisfies_constraints(sample):
                    valid_samples.append(sample)
                if len(valid_samples) == self.budget:
                    return valid_samples

            if pool_power >= max_pool_power:
                warnings.warn(
                    f"ConstrainedSobolSampler only found {len(valid_samples)}/"
                    f"{self.budget} points satisfying constraints after "
                    f"reaching max_pool_size ({2**pool_power} candidates). "
                    f"Constraints may be too tight relative to bounds, or "
                    f"increase max_pool_size."
                )
                return valid_samples

            pool_power += 1

    def register_future(self, future):
        return None

    def register_futures(self, futures):
        return None

    def skip(self, index):
        raise NotImplementedError(
            "skip not implemented for ConstrainedSobolSampler."
        )
