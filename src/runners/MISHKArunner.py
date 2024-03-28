# runners/MISHKA.py

# import numpy as np
from .base import Runner
from parsers import MISHKAparser
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
        self.parser = MISHKAparser()
        self.executable_path = executable_path
        self.default_namelist = other_params["default_namelist"]
        self.input_fort12 = other_params["input_fort12"]
        self.input_density = other_params["input_density"]

        if not os.path.exists(self.default_namelist):
            raise FileNotFoundError(
                f"Couldn't find {self.default_namelist}. ",
                f"other_params: {other_params}",
            )

        if not os.path.exists(self.input_fort12):
            raise FileNotFoundError(
                f"Couldn't find {self.input_fort12}. ", f"other_params: {other_params}"
            )

        # MISHKA can run without density file
        if not os.path.exists(self.input_density):
            self.input_density = None
            print(f"Couldn't find {self.input_density}")

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
        print(params)
        # check if equilibrium files exist and copy them to run_dir
        self.get_equilibrium_files(run_dir)

        # write input file
        self.parser.write_input_file(params, run_dir)
        mpol = self.get_mpol(params[0])

        # run code
        os.chdir(run_dir)
        subprocess.call([f"{self.executable_path}_{mpol}"])

        # process output
        # self.parser.read_output_file(run_dir)

        return True

    def get_equilibrium_files(self, run_dir: str):
        """
        Copies the equilibirum files to the run directory.
        - fort.12 is needed
        - density (fort.17) is optional (not used in all MISHKA versions?)

        Parameters
        ----------
        run_dir : str
            The run directory to where the input file is copied.

        Returns
        -------
        None
        """
        shutil.copy(self.input_fort12, run_dir)
        if self.input_density is not None:
            shutil.copy(self.input_density, run_dir)
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
