"""
# runners/SIMPLE.py

Defines the SIMPLErunner class for running simple test scenarios.

Attributes:
    Runner: Abstract base class for running codes.
    SIMPLEparser: SIMPLE parser module from the parsers package.
    subprocess: Subprocess module for running external commands.
    os: Operating system-specific module for interacting with the operating system.

"""

import os

# import numpy as np
from .base import Runner
import subprocess
from parsers import SIMPLEparser


class SIMPLErunner(Runner):
    """
    Runner class for simple test scenarios.
    Attributes:
        executable_path (str): The path to the executable binary.
    Methods:
        __init__(executable_path: str, *args, **kwargs)
            Initializes the SIMPLErunner object.
        single_code_run(params: dict, run_dir: str) -> str
            Runs a simple test program like a bash script.
    """
    def __init__(self, executable_path: str, *args, **kwargs):
        """
        Initializes the SIMPLErunner object.
        Args:
            executable_path (str): The path to the executable binary.
        """
        self.parser = SIMPLEparser()
        self.executable_path = executable_path
        self.type = kwargs.get('type')

    # in the pipeline executor we do not need to call this.
    def parse_params(self, params:dict, run_dir:str):
        self.parser.write_input_file(params, run_dir)
    
    def single_code_run(self, run_dir: str, params=None):
        """
        Runs a simple test program like a bash script.
        Args:
            params (dict): Dictionary containing parameters for the simulation.
            run_dir (str): Directory where the test program is run.
        Returns:
            str: Result of running the test program.
        """
        os.chdir(run_dir)
        if type(params) != type(None):
            self.parse_params(params, run_dir)
            #If we are after the first code run in a pipeline then the inputfile parsing
            # parsing will be handeled by the executor and no params will be provided
         
        params = self.parser.read_input_file(run_dir)
        # command to run the code in the terminal which will be carried out on the workers
        subprocess.run(["bash", self.executable_path, f"{params}", f"{run_dir}"])
        res = self.parser.read_output_file(run_dir)
        return res
