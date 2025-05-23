"""
# runners/DREAM.py

Defines the DREAMrunner class for running DREAM simulations within enchanted-surrogates.

Attributes:
    Runner: Abstract base class for running codes.
    DREAMparser: DREAM parser module from the parsers package.
    subprocess: Subprocess module for running external commands.
    os: Operating system-specific module for interacting with the operating system.

"""

import os
# import numpy as np
from .base import Runner
import subprocess
from parsers import DREAMparser, SOFTparser


class DREAMrunner(Runner):
    """
    Runner class for DREAM simulations.

    Attributes:
        executable_path (str): The path to the executable binary.

    Methods:
        __init__(executable_path: str, *args, **kwargs)
            Initializes the DREAMrunner object.
        single_code_run(params: dict, run_dir: str) -> str
            Runs a DREAM simulation.

    """

    def __init__(
        self, 
        executable_path: str, 
        other_params: dict,
        *args, 
        **kwargs,
    ):
        """
        Initializes the DREAMrunner object.

        Args:
            executable_path (str): The path to the executable binary.
            other_params (dict): Dictionary containing other parameters for
                                 DREAM initialization.

        other_params:
            base_input_file_path: str
                Path to a file that contains a base DREAM input setup.
            only_generate_files: bool
                Flag for either only creating input files or creating the 
                input files and running DREAM.

        """
        self.parser = DREAMparser()
        self.executable_path = executable_path
        
        self.only_generate_files = other_params['only_generate_files']
        if "base_input_file_path" in other_params:
            self.base_input_file_path = other_params['base_input_file_path']
        else:
            self.base_input_file_path = ' '
        if "nt" in other_params:
            self.nt = other_params['nt']
        else:
            self.nt = nt = 800
        if "non_lin_solve" in other_params:
            self.non_lin_solve = other_params['non_lin_solve']
        else:
            self.non_lin_solve = True
        if "CQ" in other_params:
            self.CQ = True
        else:
            self.CQ = False
        if "exp_file_path" in other_params:
            self.exp_file_path = other_params['exp_file_path']
        else:
            self.exp_file_path = 'None'
        if "re_grid" in other_params:
            self.re_grid = other_params['re_grid']
        else:
            self.re_grid = False
        if "F0" in other_params:
            self.F0 = other_params['F0']
        else:
            self.F0 = True

    def single_code_run(self, params: dict, run_dir: str):
        """
        Runs a DREAM simulation.

        Args:
            params (dict): Dictionary containing parameters for the simulation.
            run_dir (str): Directory where the test program is run.

        Returns:
            str: Result of running the test program.

        """
        self.parser.write_input_file(params, 
                                     run_dir, 
                                     base_input_file_path = self.base_input_file_path,
                                     nt = self.nt, 
                                     non_lin_solve = self.non_lin_solve,
                                     CQ = self.CQ,
                                     exp_file_path = self.exp_file_path,
                                     re_grid = self.re_grid,
                                     F0 = self.F0,
                                     )

        input_path = os.path.join(run_dir, 'input.h5')

        if not self.only_generate_files:
            subprocess.call([self.executable_path, input_path])
        
        return run_dir

