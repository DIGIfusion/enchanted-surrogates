"""
## Overview

The `SobolSequenceSampler` generates **quasi‑random**, low‑discrepancy samples
using a Sobol sequence. Unlike purely random sampling, Sobol sequences
systematically fill the parameter space, providing improved coverage and
reduced clustering — especially valuable for high‑dimensional or
sample‑efficient surrogate modelling.

---
"""

import numpy as np
from scipy.stats.qmc import Sobol
import warnings
from enchanted_surrogates.samplers.base_sampler import Sampler


class SobolSequenceSampler(Sampler):
    """
    ## Configuration

    To use the `SobolSequenceSampler`, specify it in the configuration file:

    ```yaml
      sampler:
        type: SobolSequenceSampler
        parameters: ['x', 'y']
        bounds: [[1, 10], [0, 1]]
        num_samples: 64
        scramble: true
        seed: 42
    ```

    Attributes:
        self.budget (int):
            Total number of Sobol samples. Automatically rounded **down** to the
            nearest power of two (required by `scipy.stats.qmc.Sobol`).

        self.power (int):
            The exponent such that `budget = 2**power`.

        self.bounds (list[tuple[float, float]]):
            Lower and upper bounds for each parameter.

        self.parameters (list[str]):
            Names of the parameters to be sampled.

        self.batch_size (int):
            Number of samples returned per batch. Defaults to the full budget.

        self.scramble (bool):
            Whether to apply Owen scrambling to the Sobol sequence, improving
            uniformity and reducing structural artifacts.

        self.seed (int):
            Seed for reproducibility when scrambling is enabled.

        self.batch_number (int):
            Internal counter tracking which batch is returned next.

        self.samples (list[list[float]]):
            Pre‑generated Sobol samples scaled to the specified bounds.

    ---

    ## Assumptions and Notes

    - Sobol sequences require sample counts of the form `2^m`.  
      If the user provides a non‑power‑of‑two budget, it is automatically
      adjusted downward with a warning.

    - Samples are generated **once** at initialization and served in batches.

    - Sobol sequences provide deterministic, quasi‑random coverage of the
      parameter space; scrambling introduces controlled randomness while
      preserving low‑discrepancy properties.

    - Parameter dimensions are assumed to be continuous.

    - The sampler does not adapt based on evaluation results.

    ---

    ## Why Sobol?

    Sobol sequences are particularly effective when:
    - exploring high‑dimensional spaces,
    - requiring uniform coverage with minimal clustering,
    - performing global sensitivity analysis,
    - training surrogate models with limited budgets.

    Their structured exploration often outperforms purely random sampling in
    convergence and stability.

    ---
    """
    def __init__(self, bounds, budget, parameters, **kwargs):
        self.budget = budget
        self.power = int(np.log2(self.budget))
        if self.budget != 2**self.power:
            warnings.warn(f'SOBOL SEQUENCE BUDGET MUST BE A POWER OF 2 eg, 2,4,16,32... SETTING BUDGET TO {2**self.power}')
        self.budget = 2**self.power

        self.bounds = bounds
        self.parameters = parameters
        self.batch_size = kwargs.get("batch_size", self.budget)
        self.scramble = kwargs.get("scramble", True)
        self.batch_number = 0
        self.seed = kwargs.get("seed", 42)
        # must be last
        self.samples = self.generate_samples()

    def get_next_samples(self) -> list[dict]:
        """Get the next batch of samples from the Sobol sequence.

        Returns the next batch_size samples (or fewer if at the end of the
        sequence).
        Each sample is a dictionary mapping parameter names to their values.

        Returns:
            list[dict]: A list of dictionaries, where each dictionary maps
                parameter names to sampled values.
                Example: [{'param1': 0.5, 'param2': 0.3},
                {'param1': 0.7, 'param2': 0.9}]
        """
        # TODO not use uniform?
        # TODO batch samples
        samples = [{key: value for key, value in zip(self.parameters, params)} for params in self.samples[self.batch_number*self.batch_size  :  min((self.batch_number+1)*self.batch_size, self.budget)]]
        # samples = [{key: value for key, value in zip(self.parameters, params)} for params in self.samples[self.batch_number*self.batch_size  :  (self.batch_number+1)*self.batch_size]]

        self.batch_number += 1
        self.submitted += len(samples)
        return samples

    def generate_samples(self):
        """Generate a Sobol sequence of samples within the specified bounds.

        Creates a quasi-random sequence using scipy's Sobol implementation,
        then scales the sequence from [0, 1]^d to the specified bounds for
        each dimension.

        Returns:
            list[list[float]]: A list of samples, where each sample is a list
                of parameter values scaled to the specified bounds. The number
                of samples equals 2^power.
        """
        # Define the dimensionality
        dim = len(self.parameters)  # Change this for the number of dimensions

        # Define the bounds for each dimension
        lower_bounds = np.array(self.bounds).T[0]
        upper_bounds = np.array(self.bounds).T[1]

        # Create a Sobol sequence generator
        try:
            sobol = Sobol(d=dim, scramble=self.scramble, rng=self.seed)
        except:
            sobol = Sobol(d=dim, scramble=self.scramble, seed=self.seed)

        # Generate points in the unit hypercube [0, 1]^d
        points = sobol.random_base2(m=self.power)  # Generates 2^power points

        # Scale the points to the desired bounds
        scaled_points = lower_bounds + points * (upper_bounds - lower_bounds)

        return scaled_points.tolist()

    def register_future(self, future):
        """
        Registers a completed result for the sampler to use it to train a model.

        This method is part of the sampler interface but is not used by
        the Sobol Sequence Sampler, as sampling does not depend on previous evaluation results.

        Args:
          future:
              A completed result from a simulation

        Returns:
          None
        """
        return None

    def register_futures(self, futures):
        """
        Registers multiple completed results.

        This method is part of the sampler interface but is not used by
        the Sobol Sequence Sampler.

        Args:
          futures:
            An iterable of completed results

        Returns:
          None
        """
        return None

    def skip(self, index):
        raise NotImplementedError(
            "skip not implemented for SobolSequenceSampler.")
