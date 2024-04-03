# runners/TGLF.py

# import numpy as np
from .base import Runner
from parsers import TGLFparser as tglfparser
import subprocess


class TGLFrunner(Runner):
    def __init__(self, *args, **kwargs):
        self.parser = tglfparser()

    def single_code_run(self, params: dict, run_dir: str):
        """Logic to run TGLF"""

        # write input file
        self.parser.write_input_file(params, run_dir)

        # process input file
        tglf_sim_dir = "/".join(run_dir.split("/")[-2:])
        subprocess.run(["tglf", "-i", f"{tglf_sim_dir}"])

        # run TGLF
        subprocess.run(["tglf", "-e", f"{tglf_sim_dir}"])

        # process TGLF
        self.parser.read_output_file(run_dir)

        # return fluxes
        output = self.parser.fluxes
        return output
