import os
from .base_runner import Runner
import numpy as np
import shutil 
from enchanted_surrogates.utils.print_stats_table import print_stats_table

class AdditiveLinearRunner(Runner):
    """
    AdditiveLinearRunner: evaluates a linear additive function f(x) = sum(w_i * x_i)
    over arbitrary dimensions with analytical mean, variance, and Sobol indices.

    Usage:
    - Initialize with weights `w` (list of floats), one per input dimension.
    - Provide input parameters as a dictionary with keys like "x1", "x2", ..., matching `w`.
    - Inputs must be in [0, 1] for analytical formulas to hold.
    """

    def __init__(self, *args, **kwargs):
        self.w = kwargs.get('w', None)
        if not isinstance(self.w, (list, tuple, np.ndarray)):
            raise TypeError("Weights 'w' must be a list, tuple, or numpy array of floats with length equal to the number of sampled dimensions.")

    def analytical_stats(self):
        """
        Returns analytical mean, variance, and Sobol indices for the additive model.

        Returns:
        - dict with keys: "mean", "variance", "sobol_indices", "sobol_total_indices"
        """
        w = np.array(self.w)
        mean = np.sum(w) / 2.0
        var = np.sum(w**2) / 12.0
        S = (w**2 / np.sum(w**2)).tolist()
        return {
            "mean": mean,
            "std": np.sqrt(var),
            "sobol_indices": S,
            "sobol_total_indices": S  # equal for additive models
        }

    def single_code_run(self, run_dir: str, params: dict = None) -> dict:
        """
        Evaluates f(x) = sum(w_i * x_i) with input validation.

        Raises:
        - ValueError if any input is outside [0, 1]
        """
        exclusive_params = [f"x{i+1}" for i in range(len(params))]
        x = [float(params[k]) for k in sorted(params.keys()) if k in exclusive_params]
        assert len(x) > 0, f"Parameters must include at least one of {exclusive_params}." 
        for xi in x:
            if not (0.0 <= xi <= 1.0):
                raise ValueError(
                    f"Invalid input {xi}. Additive Linear Runner expects all inputs in [0, 1]."
                )
        # Evaluate linear function
        output = float(np.dot(self.w[:len(x)], x))
        return {"output": output, "success": True}

    def light_post_processing(self, base_run_dir):
        stats = self.analytical_stats()
        stats['header'] = 'ANALYTICAL UQ QUANTITIES'
        stats['subheader'] = f'Additive Linear Model with weights: {self.w}'
        table = print_stats_table(stats)
        with open(os.path.join(base_run_dir, 'true_uq_stats.txt'),'w') as file:
            file.write(table)
    
    def clean(self, run_dir):
        shutil.rmtree(run_dir)
