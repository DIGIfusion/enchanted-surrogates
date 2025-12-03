import os
from .base_runner import Runner
import numpy as np
from enchanted_surrogates.utils.print_stats_table import print_stats_table
import pandas as pd
class IshigamiRunner(Runner):
    """
    IshigamiRunner: evaluates the Ishigami function for three input variables.
    It has a mean of 0 and analytically computable variance and sobol indicies.
    This is useful for testing UQ workflows to ensure they get the correct values.

    Usage:
    - Initialize with parameters `a` and `b` (floats).
    - Provide input parameters as a dictionary with keys "x1", "x2", "x3".
    - The runner computes the function value and returns a dictionary with "output" and "success".
    - samples must be in the range [-π, π].
    Example:
    runner = IshigamiRunner(a=7.0, b=0.1)
    result = runner.single_code_run("/tmp/run", params={"x1": 1.0, "x2": 2.0, "x3": 3.0})
    """

    def __init__(self, *args, **kwargs):
        self.a = kwargs.get('a', 7.0)
        self.b = kwargs.get('b', 0.1)

    def ishigami(self, x1, x2, x3):
        return np.sin(x1) + self.a * np.sin(x2)**2 + self.b * x3**4 * np.sin(x1)

    def analytical_stats(self):
        """
        Returns analytical mean, variance, first-order and total-order Sobol indices.

        Returns:
        - dict with keys: "mean", "variance", "sobol_indices", "sobol_total_indices"
        """
        a, b = self.a, self.b

        # Known analytical components
        mean = 0.0
        var = a**2 / 8 + b * np.pi**4 / 5 + b**2 * np.pi**8 / 18

        # First-order indices
        S1 = (0.5 + b * np.pi**4 / 5) / var
        S2 = (a**2 / 8) / var
        S3 = 0.0

        # Total-order indices
        ST1 = (0.5 + b * np.pi**4 / 5 + b**2 * np.pi**8 / 18) / var
        ST2 = S2
        ST3 = (b**2 * np.pi**8 / 18) / var

        return {
            "mean": mean,
            "std": np.sqrt(var),
            "sobol_indices": [S1, S2, S3],
            "sobol_total_indices": [ST1, ST2, ST3]
        }

    def single_code_run(self, run_dir: str, params: dict = None) -> dict:
            # Validate input domain
        assert 'x1' in params and 'x2' in params and 'x3' in params, "Parameters must include 'x1', 'x2', and 'x3'."
        x=[params['x1'],params['x2'],params['x3']]
        for xi in x:
            if not (-np.pi <= xi <= np.pi):
                raise ValueError(
                    f"Invalid input {xi}. Sobol IshigamiRunner expects all inputs inbetween -pi and pi. Where pi is approximately {np.pi}."
                )
        output = self.ishigami(x1=params['x1'], x2=params['x2'], x3=params['x3'])
        return {"output": output, "success": True}

    def light_post_processing(self, base_run_dir):
        stats = self.analytical_stats()
        stats['header'] = 'ANALYTICAL UQ QUANTITIES'
        stats['subheader'] = f'ISHIGAMI\n a:{self.a}'
        table = print_stats_table(stats)
        with open(os.path.join(base_run_dir, 'true_uq_stats.txt'),'w') as file:
            file.write(table)
        self.plot_slices(base_run_dir)
            
    def plot_slices(self, base_run_dir, res=100):
        from enchanted_surrogates.samplers.slices_sampler_2d import SlicesSampler2D
        save_dir = os.path.join(base_run_dir, 'ishigami_true_slice_plots')
        os.makedirs(save_dir, exist_ok=True)
        dim = 3
        budget = (dim*(dim-1) / 2) * res**2
        # res = int(budget / (dim*(dim-1) / 2))
        parameters=['x1','x2','x3']
        slice_samp = SlicesSampler2D(parameters=['x1','x2','x3'], bounds=[[-3.14,3.14],[-3.14,3.14],[-3.14,3.14]], base_run_dir=save_dir, res=res, budget=budget)
        samples = slice_samp.get_samples()
        df = pd.DataFrame(samples)
        X_slice = df[parameters].to_numpy()
        Y_slice, _ = self.ishigami(X_slice)
        print('debug len y len df', len(Y_slice), len(df))
        df_plot = pd.DataFrame(samples)
        df_plot['Ishigami'] = Y_slice
        slice_samp.plot_full_grid(df=df_plot, name=f'ishigami_')

