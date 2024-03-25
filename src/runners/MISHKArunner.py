# runners/MISHKA.py

import numpy as np
from .base import Runner
import parsers.MISHKAparser as MISHKAparser
import subprocess

class MISHKArunner(Runner):
    def __init__(self, *args, **kwargs): 
        self.parser = MISHKAparser()
        self.executable_path = "/scratch/project_2009007/MISHKA/bin/mishka1fast"
        pass 

    def single_code_run(self, params: dict, run_dir: str):
        """ Logic to run MISHKA """
        self.parser.write_input_file(params, run_dir)

        # run code
        subprocess.call([self.executable_path])

        # process output
        # self.parser.read_output_file(run_dir)

        return True 

    