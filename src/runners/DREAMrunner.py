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
        self.base_input_file_path = other_params['base_input_file_path']
        self.only_generate_files = other_params['only_generate_files']
        #if 'soft' in other_params:
        #    soft_params = other_params['soft']
        #    self.run_soft = (
        #        False
        #        if "run_soft" not in soft_params
        #        else soft_params['run_soft']
        #    )
        #    if self.run_soft:
        #        self.soft_parser = SOFTparser()
        #        self.soft_runner = SOFTrunner(
        #            soft_params['executable_path'],
        #            other_params=soft_params['other_params'],
        #        )

    def single_code_run(self, params: dict, run_dir: str):
        """
        Runs a DREAM simulation.

        Args:
            params (dict): Dictionary containing parameters for the simulation.
            run_dir (str): Directory where the test program is run.

        Returns:
            str: Result of running the test program.

        """
        self.parser.write_input_file(params, run_dir, self.base_input_file_path)

        os.chdir(run_dir)

        if not self.only_generate_files:
            subprocess.call([self.executable_path, 'input.h5'])

        return run_dir

