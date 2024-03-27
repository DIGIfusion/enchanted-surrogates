# runners/MISHKA.py

# import numpy as np
from .base import Runner
import parsers.MISHKAparser as MISHKAparser
import subprocess
import os
import shutil


class MISHKArunner(Runner):
    """
    Assumes that pre-compiled MISHKA binaries exist for m=(21,31,41,51,71).
    Adding "_m" at the end of the name of the defined executable path.


    For example, the executable path in the config file is
        executable_path: "/bin/mishka1fast"
    and the toroidal mode number is n=20, which according to the europed
    standards gives the poloidal mode number m=71.
    The runner will expect the executable to be "/bin/mishka1fast_71".

    """
    def __init__(
            self, executable_path: str, other_params: dict, *args,
            **kwargs):
        self.parser = MISHKAparser()
        self.executable_path = executable_path
        self.input_namelist = other_params['input_namelist']
        self.input_fort12 = other_params['input_fort12']
        self.input_density = other_params['input_density']

        if not os.path.exists(self.input_namelist):
            raise FileNotFoundError(
                f"Couldn't find {self.input_namelist}. ",
                f"other_params: {other_params}")

        if not os.path.exists(self.input_fort12):
            raise FileNotFoundError(
                f"Couldn't find {self.input_fort12}. ",
                f"other_params: {other_params}")

        # MISHKA can run without density file
        if not os.path.exists(self.input_density):
            self.input_density = None
            print(f"Couldn't find {self.input_density}")

    def single_code_run(self, params: dict, run_dir: str):
        """ Logic to run MISHKA """
        # check if equilibrium files exist and copy them to run_dir
        self.get_equilibrium_files(run_dir)

        # write input file
        self.parser.write_input_file(params, run_dir)
        mpol = self.get_mpol(params[0])

        # run code
        os.chdir(run_dir)
        subprocess.call([f"{self.executable_path}_{mpol}"])

        # TODO: if iteration 20 at instability found, restart with new guess

        # process output
        # self.parser.read_output_file(run_dir)

        return True

    def get_equilibrium_files(self, run_dir: str):
        shutil.copy(self.input_fort12, run_dir)
        if self.input_density is not None:
            shutil.copy(self.input_density, run_dir)
        return

    def get_mpol(self, n):
        # Europed model set_harmonic(self,n):
        nint = int(n)
        if nint < 4:
            harmonic = 21
        elif nint < 6:
            harmonic = 31
        elif nint < 10:
            harmonic = 41
        elif nint < 15:
            harmonic = 51
        else:
            harmonic = 71
        return harmonic
