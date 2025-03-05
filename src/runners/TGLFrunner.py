"""
# runners/TGLF.py

Defines the TGLFrunner class for running TGLF codes.

"""

# import numpy as np
from .base import Runner
from parsers import TGLFparser as tglfparser
import subprocess


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
        self.parser.write_input_file(params, run_dir)


        # # process input file
        # tglf_sim_dir = "/".join(run_dir.split("/")[-2:])
        print()

        # run TGLF
        try: 
            # print('Runnning tglf')
            exe_string = "tglf -e"
            result = subprocess.run(exe_string, 
                        cwd=run_dir, 
                        shell=True, 
                        check=True, 
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True
                        )
            print(f"TGLF outpu t at {run_dir}:\n{result.stdout}")
        except subprocess.CalledProcessError as e: 
            print(f"Error running executable:\n{e.stderr}")

        # process TGLF
        output = self.parser.read_output_file(run_dir)

        # return fluxes
        return output
