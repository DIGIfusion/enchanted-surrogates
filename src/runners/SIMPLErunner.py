# runners/SIMPLE.py
import os

# import numpy as np
from .base import Runner
import subprocess
from parsers import SIMPLEparser


class SIMPLErunner(Runner):
    """
    Runner class for simple test scenarios.
    """

    def __init__(self, executable_path: str, *args, **kwargs):
        self.parser = SIMPLEparser()
        self.executable_path = executable_path

    def single_code_run(self, params: dict, run_dir: str):
        """Logic to run a simple test program like a bash script"""
        # os.chdir(run_dir)
        self.parser.write_input_file(params, run_dir)
        subprocess.run(["bash", self.executable_path, f"{params}", f"{run_dir}"])
        res = self.parser.read_output_file(params, run_dir)
        return res
