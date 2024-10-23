"""
# runners/TGLF.py

Defines the TGLFrunner class for running TGLF codes.

"""

# import numpy as np
from .base import Runner
from parsers import TGLFparser as tglfparser
import subprocess
import os


class TGLFrunner(Runner):
    """
    Class for running TGLF codes.

    Methods:
        __init__(*args, **kwargs)
            Initializes the TGLFrunner object.
        single_code_run(params: dict, run_dir: str) -> dict
            Runs a single TGLF code simulation.

    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the TGLFrunner object.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        """
        self.parser = tglfparser()

    def single_code_run(self, params: dict, run_dir: str):
        """
        Runs a single TGLF code simulation.

        Args:
            params (dict): Dictionary containing parameters for the code run.
            run_dir (str): Directory path for storing the run output.

        Returns:
            dict: Dictionary containing the output fluxes from the TGLF simulation.

        """

        # write input file
        os.mkdir(run_dir)

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
