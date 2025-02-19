# runners/MISHKA.py

# import numpy as np
# import json
import os
import shutil
import subprocess
from parsers import MISHKAparser
from .base import Runner
from dask.distributed import print


class MISHKArunner(Runner):
    """
    Class for running MISHKA.
    Requires that pre-compiled MISHKA binaries exist for m=(21,31,41,51,71).
    Adding "_m" at the end of the name of the defined executable path.

    For example, if the executable path in the config file is
        executable_path: "/bin/mishka1fast"
    and the toroidal mode number is n=20, which according to the europed
    standards gives the poloidal mode number m=71, then
    the runner will expect the executable to be found at "/bin/mishka1fast_71".

    Either define the fort.12 file in the runner config file or define the
    path to the (HELENA output) directory in the params.

    Attributes
    ----------
    executable_path : str
        the path to the pre-compiled executable MISHKA binary (without "_m")
    other_params : dict
        a dictinoary of other parameters defined in the config file

    Methods
    -------
    single_code_run()
        Runs MISHKA after copying and writing the input files

    get_equilibrium_files
        Copies the fort.12 file specified path input_fort12 (out from HELENA)
        to the run_dir.

    """

    def __init__(self, executable_path: str, other_params: dict, *args, **kwargs):
        self.parser = MISHKAparser(default_namelist=other_params["namelist_path"])
        self.executable_path = executable_path
        self.default_namelist = other_params["namelist_path"]
        self.input_fort12 = (
            "" if "input_fort12" not in other_params else other_params["input_fort12"]
        )
        self.input_density = (
            "" if "input_density" not in other_params else other_params["input_density"]
        )
        self.pre_run_check()

    def single_code_run(self, params: dict, run_dir: str):
        """
        Logic to run MISHKA

        Parameters
        ----------
        run_dir : str
            The directory in where MISHKA is run.

        Returns
        -------
        None
        """
        print(run_dir, params)
        # check if equilibrium files exist and copy them to run_dir
        if not os.path.exists(self.default_namelist):
            print(
                f"Couldn't find {self.default_namelist}. \n",
                f"params: {params}",
            )
            return

        if len(self.input_fort12) > 0 and not os.path.exists(self.input_fort12):
            print(
                f"Couldn't find the defined {self.input_fort12}. \n",
                f"params: {params}",
            )
            return

        self.get_equilibrium_files(run_dir, params)

        # write input file
        self.parser.write_input_file(params, run_dir)
        mpol = self.get_mpol(params["ntor"])

        # run code
        os.chdir(run_dir)
        subprocess.call([f"{self.executable_path}_{mpol}"])

        # process output
        # self.parser.read_output_file(run_dir)
        self.parser.write_summary(run_dir, mpol, params)
        self.parser.clean_output_files(run_dir)

        return True

    def get_equilibrium_files(self, run_dir: str, params: dict):
        """
        Copies the equilibirum files to the run directory.
        - fort.12 is needed
        - density (fort.17) is optional (it is actually not used at all in our
          MISHKA versions?)

        Parameters
        ----------
        run_dir : str
            The run directory to where the input file is copied.

        Returns
        -------
        None
        """
        if "helena_dir" in params:
            file_path = os.path.join(params["helena_dir"], "fort.12")
        else:
            file_path = self.input_fort12
        print(f"copying {file_path}")
        shutil.copy(file_path, run_dir)

        # if self.input_density is not None:
        #     shutil.copy(self.input_density, run_dir)
        return

    def get_mpol(self, n):
        """
        Chooses the maximum poloidal harmonic to use in MISHKA.
        As this class assumes that the MISHKA verions are
        pre-compiled, this function chooses which version to use.
        Implementation following the Europed model set_harmonic(self,n)
        for europed input parameter 0.

        Parameters
        ----------
        n : int
            The toroidal mode number

        Returns
        -------
        harmonic: int
            The poloidal harmonic
        """
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

    def pre_run_check(self):
        """
        Performs pre-run checks to ensure necessary files exist before running the simulation.

        Raises:
            FileNotFoundError: If the executable path or the namelist path is not found.

        """
        ntors = [21, 31, 41, 51, 71]
        for ntor in ntors:
            if not os.path.isfile(f"{self.executable_path}_{ntor}"):
                raise FileNotFoundError(
                    f"The executable path ({self.executable_path}_{ntor}) provided to the MISHKA ",
                    "runner is not found. Exiting.",
                )

        return
