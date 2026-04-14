"""
Example runner that implements a simple synthetic function for testing
and demonstration purposes.
"""
import os
import numpy as np

from .base_runner import Runner


class SyntheticDoubleGaussianRunner(Runner):
    """
    A simple Runner that implements a double Gaussian
    with a background slope.

    kwargs:
        dimensions (int): Number of dimensions (1 or 2).
        model_parameters - vector of model parameters
                           [a, g11, g12, g13, g21, g22, g23]:
                           y = a*x + g11*np.exp(-(x - g13)**2/g12)
                               + g21*np.exp(-(x - g23)**2/g22)

    Returns a dictionary of containing the output and True for success.
    Future developments can include synthetic failure implementations.
    """

    def __init__(self, *args, **kwargs):
        self.dimensions = kwargs.get("dimensions", 1)

        if self.dimensions == 1:
            self.model_parameters = kwargs.get(
                "model_parameters",
                [0.2, 1.0, 0.001, 0.2, 0.6, 0.01, 0.7]
            )

        elif self.dimensions == 2:
            self.model_parameters = kwargs.get(
                "model_parameters",
                [0.2, 0.1,   # plane slopes ax, ay
                 1.0, 0.02, 0.3, 0.3,   # hill 1 (centered near middle)
                 0.8, 0.04, 0.75, 0.6]  # hill 2 (off-center)
            )
        else:
            raise ValueError("Only dimensions=1 or 2 supported.")

    def single_code_run(self, run_dir: str, params: dict = None) -> dict:
        """
        Run the synthetic function with given parameters.

        Args:
            run_dir (str): Directory to save outputs.
            params (dict): Dictionary of parameter values.
                           For 1D: {'x': value}
                           For 2D: {'x': value, 'y': value}

        Returns:
            dict: A dictionary containing the output and success status.
        """
        if params is None:
            params = {}

        outfile = os.path.join(run_dir, "output.txt")

        if self.dimensions == 1:
            x = float(params["x"])
            a, g11, g12, g13, g21, g22, g23 = self.model_parameters

            z = a * x
            z += g11 * np.exp(-(x - g13)**2 / g12)
            z += g21 * np.exp(-(x - g23)**2 / g22)

        elif self.dimensions == 2:
            x = float(params["x"])
            y = float(params["y"])

            (ax, ay,
             g11, g12, g1x, g1y,
             g21, g22, g2x, g2y) = self.model_parameters

            # background plane
            z = ax * x + ay * y

            # Gaussian hills
            z += self.gaussian_2d(x, y, g11, g12, g1x, g1y)
            z += self.gaussian_2d(x, y, g21, g22, g2x, g2y)

        with open(outfile, "a") as f:
            f.write(str(z))

        return {"output": z, "success": True}

    def gaussian_2d(self, x, y, amp, width, x0, y0):
        r2 = (x - x0)**2 + (y - y0)**2
        return amp * np.exp(-r2 / width)