# import numpy as np
# import json
import os
import shutil
import subprocess
from datetime import datetime
from parsers import CASTORparser
from .base import Runner
from dask.distributed import print


class CASTORrunner(Runner):
    """
    Class for running CASTOR.
    Requires that pre-compiled CASTOR binary exist for m=(21,31,41,51,71).
    Adding "_m" at the end of the name of the defined executable path.

    For example, if the executable path in the config file is
        executable_path: "/bin/cas12"
    and the toroidal mode number is n=20, which according to the europed
    standards gives the poloidal mode number m=71, then
    the runner will expect the executable to be found at "/bin/cas12_71".

    Attributes
    ----------
    executable_path : str
        the path to the pre-compiled executable CASTOR binary (without "_m")
    other_params : dict
        a dictinoary of other parameters defined in the config file

    Methods
    -------
    single_code_run()
        Runs CASTOR after copying and writing the input files

    get_equilibrium_files
        Copies the fort.12 file specified path input_fort12 (out from HELENA)
        to the run_dir.

    """

    def __init__(self, executable_path: str, other_params: dict, *args, **kwargs):
        self.parser = CASTORparser(namelist_path=other_params["namelist_path"])

        self.executable_path = executable_path
        self.namelist_path = other_params["namelist_path"]
        self.eigenvalue_tracing = other_params["eigenvalue_tracing"]
        self.resistivity_type = other_params["resistivity_type"]
        self.pre_run_check()

    def single_code_run(self, params: dict, run_dir: str):
        """
        Logic to run CASTOR

        Parameters
        ----------
        run_dir : str
            The directory in where CASTOR is run.

        Returns
        -------
        None
        """
        start_time = datetime.now()
        print(f"CASTOR run starting... ({run_dir}) ({start_time})", flush=True)

        # Get files from HELENA equilibrium
        self.get_equilibrium_files(run_dir, params, self.resistivity_type)

        # Write input file
        self.parser.write_input_file(params, run_dir)
        mpol = self.get_mpol(params["ntor"])

        # Run code
        os.chdir(run_dir)
        subprocess.call([f"{self.executable_path}_{mpol}"])

        # Process output
        # self.parser.read_output_file(run_dir)
        success, growthrate = self.parser.write_summary(run_dir, mpol, params)
        self.parser.clean_output_files(run_dir)

        print(
            f"CASTOR run finished in {datetime.now() - start_time}. ({run_dir})",
            flush=True,
        )
        return success, growthrate

    def get_equilibrium_files(
        self, run_dir: str, params: dict, resistivity_type: str = "spitzer"
    ):
        """
        Copies the equilibirum files to the run directory.

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
            shutil.copy(file_path, run_dir)
            self.parser.extract_resistivity(
                filepath=os.path.join(params["helena_dir"], "fort.20"),
                outputpath=os.path.join(run_dir, "fort.14"),
                resistivity_type=resistivity_type,
            )
        else:
            raise FileNotFoundError(
                "No helena_dir specified in the params. Exiting.",
            )
        return

    def get_mpol(self, n):
        """
        Chooses the maximum poloidal harmonic to use in CASTOR.
        As this class assumes that the CASTOR verions are
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
                    f"The executable path ({self.executable_path}_{ntor}) provided to the CASTOR ",
                    "runner is not found. Exiting.",
                )
        if not os.path.exists(self.namelist_path):
            raise FileNotFoundError(
                f"Couldn't find {self.namelist_path}. Exiting.",
            )

        return
