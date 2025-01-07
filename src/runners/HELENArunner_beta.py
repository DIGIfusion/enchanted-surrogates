"""
# runners/HELENA.py

Defines the HELENArunner class for running HELENA simulations.

"""

import numpy as np
from .base import Runner
from parsers import HELENAparser
import subprocess
import os
from dask.distributed import print

# import logging


class HELENArunner_beta(Runner):
    """
    Class for running HELENA.


    Attributes
    ----------
    executable_path : str
        the path to the pre-compiled executable HELENA binary


    Methods
    -------
    single_code_run()
        Runs HELENA after copying and writing the input files

    """

    def __init__(
        self,
        executable_path,
        other_params: dict,
        *args,
        **kwargs,
    ):
        """
        Initializes the HELENArunner object.

        Args:
            executable_path (str): The path to the pre-compiled executable HELENA binary.
            other_params (dict): Dictionary containing other parameters for initialization.

        other_params:
            namelist_path : str
                The namelist constaining the HELENA values to be kept constant
                during the run.

            only_generate_files: bool
                Flag for either only creating input files or creating the files
                and running HELENA.

        """
        self.parser = HELENAparser()
        self.executable_path = (
            executable_path  # "/scratch/project_2009007/HELENA/bin/hel13_64"
        )
        self.namelist_path = other_params["namelist_path"]
        self.only_generate_files = other_params["only_generate_files"]

        self.pre_run_check()

    def single_code_run(self, params: dict, run_dir: str, tolerance=1e-4):
        """
        Runs HELENA simulation.

        Args:
            params (dict): Dictionary containing parameters for the simulation.
            run_dir (str): Directory where HELENA is run.

        Returns:
            bool: True if the simulation is successful, False otherwise.

        """
        print(f"single_code_run: {run_dir}", flush=True)
        self.parser.write_input_file(params, run_dir, self.namelist_path)

        os.chdir(run_dir)
        # run code
        if not self.only_generate_files:
            beta_target = params['beta_N']
            self.parser.modify_fast_ion_pressure('fort.10', 0.0)
            subprocess.call([self.executable_path])
            output_vars = self.parser.get_real_world_geometry_factors_from_f20('fort.20')
            beta_n0 = 1e2*output_vars['BETAN']
            self.parser.modify_fast_ion_pressure('fort.10', 0.1)
            subprocess.call([self.executable_path])
            output_vars = self.parser.get_real_world_geometry_factors_from_f20('fort.20')
            beta_n01 = 1e2*output_vars['BETAN']
            apftarg = (beta_target - beta_n0)*0.1/(beta_n01 - beta_n0)
            self.parser.modify_fast_ion_pressure('fort.10', apftarg)
            subprocess.call([self.executable_path])
            output_vars = self.parser.get_real_world_geometry_factors_from_f20('fort.20')
            beta_n = 1e2*output_vars['BETAN']
            while np.abs(beta_target - beta_n) > tolerance*beta_target:
                apftarg = (beta_target - beta_n0)*apftarg/(beta_n - beta_n0)
                self.parser.modify_fast_ion_pressure('fort.10', apftarg)
                subprocess.call([self.executable_path])
                output_vars = self.parser.get_real_world_geometry_factors_from_f20('fort.20')
                beta_n = 1e2*output_vars['BETAN']

        # process output
        # self.parser.read_output_file(run_dir)
        self.parser.write_summary(run_dir, params)
        self.parser.clean_output_files(run_dir)

        return True

    def pre_run_check(self):
        """
        Performs pre-run checks to ensure necessary files exist before running the simulation.

        Raises:
            FileNotFoundError: If the executable path or the namelist path is not found.

        """
        # Does executable exist?
        if not os.path.isfile(self.executable_path):
            raise FileNotFoundError(
                f"The executable path ({self.executable_path}) provided to the HELENA runner is not found. Exiting."
            )
        # Does base namelist exist?
        if not os.path.isfile(self.namelist_path):
            raise FileNotFoundError(
                f"The namelist path ({self.namelist_path}) provided to the HELENA runner is not found. Exiting."
            )
        # TODO: Does namelist contain paramters that this structure can handle or that makes sense?
        # TODO: neped > nesep
        return
