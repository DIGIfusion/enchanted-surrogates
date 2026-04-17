import os
import sys
import numpy as np
from time import sleep
from .base_runner import Runner
from enchanted_surrogates.utils.logger import get_logger
import pandas as pd

log = get_logger(__name__)

class FriedmanLinear(Runner):
    """
    FriedmanLinear: An implementation of the Friedman #1 regression benchmark 
    with optional Gaussian noise.

    This is ment to be a part of the 3 runner series of friedman interaction, quadratic and linear in order
    to test nested active learning

    Overview:
    - Implements the non-linear Friedman #1 function.
    - Supports the injection of Gaussian noise to simulate measurement error,
      which is a critical component for testing Active Learning robustness.

    Mathematical Definition:
    y_true = 10 * sin(pi * x1 * x2) + 20 * (x3 - 0.5)^2 + 10 * x4 + 5 * x5
    y_observed = y_true + N(0, noise_std^2)

    Initialization parameters (via kwargs):
    - sleep_sec (number or two-element iterable, default 0.01): Duration to pause 
      after calculation.
    - fail_prob (float in [0, 1], default 0): Probability of synthetic RuntimeError.
    - noise_std (float, default 0.0): The standard deviation of the Gaussian noise 
      added to the output. Set to 0 for a deterministic function.
    """

    def __init__(self, *args, **kwargs):
        self.sleep_sec = kwargs.get("sleep_sec", 0.01)
        self.fail_prob = kwargs.get("fail_prob", 0)
        self.noise_std = float(kwargs.get("noise_std", 0.0))

    def get_sleep_sec(self) -> float:
        """
        Derives a float sleep duration from the configured sleep_sec.
        """
        if isinstance(self.sleep_sec, (int, float, np.number)):
            return float(self.sleep_sec)
        
        if hasattr(self.sleep_sec, "__iter__"):
            seq = list(self.sleep_sec)
            if len(seq) != 2:
                raise ValueError(f"Random sleep bounds must have length 2. Got {len(seq)}")
            low, high = seq
            if float(low) > float(high):
                raise ValueError("Lower bound must be <= upper bound.")
            return float(np.random.uniform(float(low), float(high)))
            
        raise TypeError("sleep_sec must be a number or a two-element iterable.")

    def single_code_run(self, run_dir: str, params: dict = None) -> dict:
        """
        Calculates the Friedman #1 value, adds Gaussian noise, and manages the lifecycle.

        Args:
            run_dir (str): Directory where 'output.txt' will be stored.
            params (dict): Dictionary containing keys 'x1' through 'x5'.

        Returns:
            dict: {
                "output": float,         # The noisy/observed Friedman value
                "true_value": float,     # The underlying noiseless value (for benchmarking)
                "success": bool          # True if processed without failure
            }
        """
        os.makedirs(run_dir, exist_ok=True)
        params = params or {}
        outfile = os.path.join(run_dir, "output.txt")

        # 1. Strict Parameter Extraction
        required_keys = ['friedman_quadratic_run_dir'] + [f"x{i}" for i in range(4, 6)]
        missing_keys = [k for k in required_keys if k not in params]
        
        if missing_keys:
            raise ValueError(
                f"FriedmanRunner missing required parameters: {missing_keys}. "
                f"Provided keys: {list(params.keys())}"
            )

        try:
            x = [float(params[k]) for k in required_keys[1:]]
        except (ValueError, TypeError) as exc:
            raise TypeError(
                f"FriedmanRunner requires numeric inputs for x4-x5. "
                f"Check parameter types in: {params}. Error: {exc}"
            )
        
        # 2. Deterministic Calculation (y_true)
        # term_interaction = 10 * np.sin(np.pi * x[0] * x[1])
        # term_quadratic   = 20 * (x[2] - 0.5)**2
        parent_output_path = os.path.join(params['friedman_quadratic_run_dir'], 'friedman_quadratic.csv')
        df = pd.read_csv(parent_output_path)
        term_interaction_plus_quadratic = df['friedman_interaction_plus_quadratic_noisy'].iloc[0]
        term_linear      = (10 * x[0]) + (5 * x[1])
        
        y_true = term_interaction_plus_quadratic + term_linear

        # 3. Noise Injection
        # We sample from a normal distribution with mean 0 and standard deviation self.noise_std
        noise = 0.0
        if self.noise_std > 0:
            noise = np.random.normal(0, self.noise_std)
        
        y_observed = y_true + noise

        # 4. Latency Simulation
        sleep_duration = self.get_sleep_sec()
        sleep(sleep_duration)

        result = {
            "output": y_observed, 
            "friedman_linear_true": term_linear,
            "success": True
        }

        # 7. Failure Injection
        if self.fail_prob > 0:
            if np.random.uniform() < float(self.fail_prob):
                raise RuntimeError(f"Synthetic failure injected: fail_prob={self.fail_prob}")

        return result