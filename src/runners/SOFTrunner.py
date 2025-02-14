"""
# runners/HELENA.py

Defines the HELENArunner class for running HELENA simulations.

"""

import numpy as np
from .base import Runner
from parsers import SOFTparser
import subprocess
import os
from dask.distributed import Client, wait

# import logging


class SOFTrunner(Runner):
    """
    Class for running SOFT.


    Attributes
    ----------
    executable_path : str
        the path to the pre-compiled executable HELENA binary


    Methods
    -------
    single_code_run()
        Runs SOFT after copying and writing the input files

    """

    def __init__(
        self,
        executable_path,
        other_params: dict,
        *args,
        **kwargs,
    ):
        """
        Initializes the SOFTrunner object.

        Args:
            executable_path (str): The path to the pre-compiled executable SOFT binary.
            other_params (dict): Dictionary containing other parameters for initialization.

        other_params:
            number_of_mpi : int
                Number of MPI processes to launch.

        """
        self.parser = SOFTparser()
        self.executable_path = executable_path
        self.number_of_mpi = other_params["number_of_mpi"]
        self.only_generate_files = other_params["only_generate_files"]

    def single_code_run(self, params: dict, run_dir: str):
        """
        Runs SOFT simulation.

        Args:
            params (dict): Dictionary containing parameters for the simulation.
            run_dir (str): Directory where SOFT is run.
            tolerance (float): Tolerance for beta iteration

        Returns:
            bool: True if the simulation is successful, False otherwise.

        """
        print(f"single_code_run: {run_dir}", flush=True)

        self.parser.write_input_file(params, run_dir)

        os.chdir(run_dir)
        # run code
        if not self.only_generate_files:
            if self.number_of_mpi > 1:
                subprocess.call(['mpirun','-n',str(self.number_of_mpi),self.executable_path,'soft_input'])
            else:
                subprocess.call([self.executable_path, 'soft_input'])

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
                f"The executable path ({self.executable_path}) provided to the HELENA runner ",
                "is not found. Exiting.",
            )
        
        # TODO: Does namelist contain paramters that this structure can handle or that makes sense?
        # TODO: neped > nesep
        return
