import os
from .base_runner import Runner
import numpy as np
import shutil 
from enchanted_surrogates.utils.print_stats_table import print_stats_table
class SobolGRunner(Runner):
    """
    SobolGRunner: evaluates the Sobol G-function for a given input vector and sensitivity parameters.

    Usage:
    - Initialize with sensitivity parameters `a`, list of floats from 0 to 1
        - 0 means low sensitivity and so low sobol indicies for that dimension
        - 1 means high sensitivity and so high sobol indicies for that dimension
        - typical value for a could be [0.01,0.0.1,0.5,0.99,0.99] where each value is for a different dimension
    - Samples must have parameters like "x1", "x2", ..., matching the length of `a`.
    - Samples must be in the range [0, 1].
    - The runner computes the function value and returns a dictionary with "output" and "success".
    - The mean is 1 and it has analytically computable variance and sobol indices. This is usefull for testing UQ samplers.

    Example:
    runner = SobolGRunner(a=[1.0, 2.0, 5.0])
    result = runner.single_code_run("/tmp/run", params={"x1": 0.3, "x2": 0.7, "x3": 0.5})
    """

    def __init__(self, *args, **kwargs):
        self.a = kwargs.get('a', None)
        if not isinstance(self.a, (list, tuple, np.ndarray)):
            raise TypeError("Parameter 'a' must be a list, tuple, or numpy array of floats with length equal to the number of sampled dimensions. Each float must be between 0 and 1.")

    def sobol_g(self, x):
        return np.prod([(np.abs(4 * xi - 2) + ai) / (1 + ai) for xi, ai in zip(x, self.a)])

    def analytical_stats(self):
        """
        Returns analytical mean, variance, first-order and total-order Sobol indices.

        Returns:
        - dict with keys: "mean", "variance", "sobol_indices", "sobol_total_indices"
        """
        a = np.array(self.a)
        V = 1.0 / (3.0 * (1.0 + a)**2)  # individual variances
        D = np.prod(1.0 + V)            # total variance + 1
        var = D - 1.0                   # total variance

        # First-order Sobol indices
        S = V / var

        # Total-order Sobol indices
        ST = []
        for i in range(len(a)):
            D_wo_i = np.prod(1.0 + np.delete(V, i))
            ST_i = 1.0 - D_wo_i / D
            ST.append(ST_i)

        return {
            "mean": 1.0,
            "std": np.sqrt(var),
            "sobol_indices": S.tolist(),
            "sobol_total_indices": ST
        }


    def single_code_run(self, run_dir: str, params: dict = None) -> dict:
        # Validate input domain
        exclusive_params = [f"x{i+1}" for i in range(len(self.a))]
        x = [float(params[k]) for k in sorted(params.keys()) if k in exclusive_params]
        assert len(x) > 0, f"Parameters must include at least one of {exclusive_params}." 
        for xi in x:
            if not (0.0 <= xi <= 1.0):
                raise ValueError(
                    f"Invalid input {xi}. Sobol G-function expects all inputs in [0, 1]."
                )
        output = self.sobol_g(x)
        self.clean(run_dir)
        return {"output": output, "success": True}
    
    def light_post_processing(self, base_run_dir):
        stats = self.analytical_stats()
        stats['header'] = 'ANALYTICAL UQ QUANTITIES'
        stats['subheader'] = f'SOBOL G FUNCTION | a:{self.a}'
        table = print_stats_table(stats)
        with open(os.path.join(base_run_dir, 'true_uq_stats.txt'),'w') as file:
            file.write(table)
    
    def clean(self, run_dir):
        shutil.rmtree(run_dir)
