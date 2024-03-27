# runners/TGLF.py

# import numpy as np
from .base import Runner
import parsers.TGLFparser as tglfparser
import subprocess


class TGLFrunner(Runner):
    def __init__(self, *args, **kwargs):
        self.parser = tglfparser()

    def single_code_run(self, params: dict, run_dir: str):
        """ Logic to run TGLF """
        """
        # TODO: change run_dir?
        run_dir, as it is now, is the full path,
        so we need to extract just the db_dir + simdir TGLF
        """
        # write input file
        self.parser.write_input_file(params, run_dir)

        # process input file
        tglf_sim_dir = '/'.join(run_dir.split('/')[-2:])
        subprocess.run(['tglf', '-i', f'{tglf_sim_dir}'])

        # run code
        subprocess.run(['tglf', '-e', f'{tglf_sim_dir}'])

        # process code
        self.parser.read_output_file(run_dir)

        output = self.parser.fluxes
        return output
