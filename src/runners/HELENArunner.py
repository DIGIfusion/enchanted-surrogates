# runners/HELENA.py

# import numpy as np
from .base import Runner
from parsers import HELENAparser
import subprocess
import os


class HELENArunner(Runner):
    def __init__(self, *args, **kwargs):
        self.parser = HELENAparser()
        self.executable_path = "/scratch/project_2009007/HELENA/bin/hel13_64"
        pass

    def single_code_run(self, params: dict, run_dir: str):
        """Logic to run HELENA"""
        self.parser.write_input_file(params, run_dir)

        # run code
        os.chdir(run_dir)
        subprocess.call([self.executable_path])

        # process output
        # self.parser.read_output_file(run_dir)

        return True
