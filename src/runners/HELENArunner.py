# runners/HELENA.py

import numpy as np
from .base import Runner
import parsers.HELENAparser as helenaparser

class HELENArunner(Runner):
    def __init__(self, *args, **kwargs): 
        self.parser = helenaparser()
        pass 

    def single_code_run(self, params: dict, run_dir: str):
        """ Logic to run HELENA """
        self.parser.write_input_file(params, run_dir)
        return True 

    