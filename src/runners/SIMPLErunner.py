# runners/SIMPLE.py

# import numpy as np
from .base import Runner
import parsers.SIMPLEparser as SIMPLEparser
import subprocess


class SIMPLErunner(Runner):
    def __init__(self, *args, **kwargs):
        self.parser = SIMPLEparser()
        pass

    def single_code_run(self, params: dict, run_dir: str):
        """ Logic to run a simple test program like a bash script """
        self.parser.write_input_file(params, run_dir)
        subprocess.run([
            'bash', '/pathtoexecutablescript/simple.sh', f'{params}'])

        return True
