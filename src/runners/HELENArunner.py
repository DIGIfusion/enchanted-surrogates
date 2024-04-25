# runners/HELENA.py

# import numpy as np
from .base import Runner
from parsers import HELENAparser
import subprocess
import os
from dask.distributed import print

# import logging


class HELENArunner(Runner):
    """
    Class for running HELENA.


    Attributes
    ----------
    executable_path : str
        the path to the pre-compiled executable MISHKA binary (without "_m")


    Methods
    -------
    single_code_run()
        Runs MISHKA after copying and writing the input files

    """

    def __init__(
        self,
        executable_path,
        other_params: dict,
        *args,
        **kwargs,
    ):
        """
        Initialie HELENA runner class.

        Parameters
        ----------
        executable_path : str
            The path to where the executable binary.

        namelist_path : str
            The namelist constaining the HELENA values to be kept constant
            during the run.

        only_generate_files: bool
            Flag for either only creating input files or creating the files
            and running HELENA.

        Returns
        -------
        None
        """
        self.parser = HELENAparser()
        self.executable_path = (
            executable_path  # "/scratch/project_2009007/HELENA/bin/hel13_64"
        )
        self.namelist_path = other_params["namelist_path"]
        self.only_generate_files = other_params["only_generate_files"]

        self.pre_run_check()

    def single_code_run(self, params: dict, run_dir: str):
        """
        Logic to run HELENA.

        Parameters
        ----------
        run_dir : str
            The directory in where HELENA is run.

        Returns
        -------
        None
        """
        print(f"single_code_run: {run_dir}", flush=True)
        self.parser.write_input_file(params, run_dir, self.namelist_path)

        os.chdir(run_dir)
        # run code
        if not self.only_generate_files:
            subprocess.call([self.executable_path])

        # process output
        # self.parser.read_output_file(run_dir)
        self.parser.write_summary(run_dir, params)
        self.parser.clean_output_files(run_dir)

        return True

    def pre_run_check(self):
        # Does executable exist?
        if not os.path.isfile(self.executable_path):
            raise FileNotFoundError(
                f"The executable path ({self.executable_path}) provided to the HELENA runner is not found. Exiting."
            )
        # Does base namelist exist?
        if not os.path.isfile(self.namelist_path):
            raise FileNotFoundError(
                f"The namelist path ({self.namelist_path}) provided to the HELENA runner is not found. Exiting."
            )
        # TODO: Does base namelist contain paramters that this structure can handle or that makes sense?
        # TODO: neped > nesep
        return
