# runners/TGLF.py

import numpy as np
from .base import Runner
import parsers.TGLFparser as tglfparser

class TGLFrunner(Runner):
    def __init__(self, *args, **kwargs): 
        self.parser = tglfparser()
        pass 

    def single_code_run(self, params: dict, run_dir: str):
        """ Logic to run TGLF """
        self.parser.write_input_file(params, run_dir)
        return True 

    