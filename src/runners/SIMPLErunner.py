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

    def single_code_run(self, params: dict, run_dir: str):
        """
        Runs a simple test program like a bash script.

        Args:
            params (dict): Dictionary containing parameters for the simulation.
            run_dir (str): Directory where the test program is run.

        Returns:
            str: Result of running the test program.

        """
        # os.chdir(run_dir)
        self.parser.write_input_file(params, run_dir)
        subprocess.run(["bash", self.executable_path, f"{params}", f"{run_dir}"])
        res = self.parser.read_output_file(params, run_dir)
        return res
