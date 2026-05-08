import numpy as np
from enchanted_surrogates.samplers.base_sampler import Sampler
import warnings

class LatinHypercubeSampler(Sampler):
    def __init__(self, bounds, num_samples, parameters, **kwargs):
        self.num_samples = int(num_samples)
        if self.num_samples < 1:
            raise ValueError("num_samples must be >= 1")

        self.num_repeats = kwargs.get("num_repeats", 1)
        self.budget = self.num_samples * self.num_repeats

        self.bounds = bounds
        self.parameters = parameters
        self.dim = len(self.parameters)

        self.batch_size = kwargs.get("batch_size", self.budget)
        self.include_index = kwargs.get("include_index", False)
        self.seed = kwargs.get("seed", 42)

        self.batch_number = 0
        self.submitted = 0

        # must be last
        self.samples = self.generate_samples()

    def get_next_samples(self) -> list[dict]:
        start = self.batch_number * self.batch_size
        end = min((self.batch_number + 1) * self.batch_size, self.budget)

        samples = [
            {key: value for key, value in zip(self.parameters, params)}
            for params in self.samples[start:end]
        ]

        samples = samples * self.num_repeats
        self.batch_number += 1

        if self.include_index:
            samples = [
                {**samp, "index": idx}
                for samp, idx in zip(samples, range(self.submitted, self.submitted + len(samples)))
            ]

        self.submitted += len(samples)
        return samples

    def generate_samples(self):
        print("GENERATING LATIN HYPERCUBE SAMPLES")

        rng = np.random.default_rng(self.seed)

        # Extract bounds
        lower_bounds = np.array(self.bounds).T[0]
        upper_bounds = np.array(self.bounds).T[1]

        N = self.num_samples
        D = self.dim

        # ------------------------------------------------------------
        # 1. Allocate the LHS matrix in the unit hypercube
        # ------------------------------------------------------------
        points = np.zeros((N, D))

        # ------------------------------------------------------------
        # 2. For each dimension:
        #      - create N strata
        #      - sample one random point inside each stratum
        #      - permute those N samples
        # ------------------------------------------------------------
        for dim in range(D):

            # Stratum indices: 0, 1, ..., N-1
            strata = np.arange(N)

            # Random offset inside each stratum
            offsets = rng.random(N)

            # Random point inside each stratum
            # value[i] ∈ [i/N, (i+1)/N]
            dim_samples = (strata + offsets) / N

            # Shuffle to enforce Latin property
            perm = rng.permutation(N)
            points[:, dim] = dim_samples[perm]

        # ------------------------------------------------------------
        # 3. Scale from [0,1] to user-specified bounds
        # ------------------------------------------------------------
        scaled = lower_bounds + points * (upper_bounds - lower_bounds)

        return scaled.tolist()

    def register_future(self, future):
        return None

    def register_futures(self, futures):
        return None
