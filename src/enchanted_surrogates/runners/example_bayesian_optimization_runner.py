import os
import numpy as np

from .base_runner import Runner

class ExampleBayesianOptimizationRunner(Runner):
    """
    ExampleBayesianOptimizationRunner: a simple Runner that implements a  1D 
    double Gaussian with a background slope to test 
    the BayesianOptimizationSampler.

    kwargs:
        model_parameters - vector of model parameters 
                           [a, g11, g12, g13, g21, g22, g23]:
                           y = a*x + g11*np.exp(-(x - g13)**2/g12) 
                               + g21*np.exp(-(x - g23)**2/g22)

    Returns a dictionary of containing the output and True for success. 
    Future developments can include synthetic failure implementations.
    """

    def __init__(self, *args, **kwargs):
        self.model_parameters = kwargs.get('model_parameters', 
                                           [0.2, 1.0, 0.001, 0.2, 
                                            0.6, 0.01, 0.7])

    def single_code_run(self, run_dir: str, params: dict = None) -> dict:
        """
        Execute a single example run in `run_dir` and
        return a simple results dictionary.
        """

        # Ensure run_dir exists
        os.makedirs(run_dir, exist_ok=True)

        if params is None:
            params = {}

        outfile = os.path.join(run_dir, "output.txt")

        x = params['x']
        x = float(x)

        a   = self.model_parameters[0]
        g11 = self.model_parameters[1]
        g12 = self.model_parameters[2]
        g13 = self.model_parameters[3]
        g21 = self.model_parameters[4]
        g22 = self.model_parameters[5]
        g23 = self.model_parameters[6]
        
        y = a*x 
        y += self.gfunc(x, g1=g11, g2=g12, g3=g13)
        y += self.gfunc(x, g1=g21, g2=g22, g3=g23)

        with open(outfile, 'a') as f:
            result = y
            f.write(str(result))

        result = {"output": y, "success": True}

        return result

    def gfunc(self, x, g1=1.0, g2=1.0, g3=1.0):
        return g1*np.exp(-(x - g3)**2/g2)
