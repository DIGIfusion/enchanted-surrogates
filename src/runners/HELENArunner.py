"""
# runners/HELENA.py

Defines the HELENArunner class for running HELENA simulations.

"""

import numpy as np
from .base import Runner
from runners.MISHKArunner import MISHKArunner
from parsers import HELENAparser
import subprocess
import os
import shutil
from dask.distributed import print


class HELENArunner(Runner):
    """
    Class for running HELENA.


    Attributes
    ----------
    executable_path : str
        the path to the pre-compiled executable HELENA binary


    Methods
    -------
    single_code_run()
        Runs HELENA after copying and writing the input files

    """

    def __init__(
        self,
        executable_path,
        other_params: dict,
        *args,
        **kwargs,
    ):
        """
        Initializes the HELENArunner object.

        Args:
            executable_path (str): The path to the pre-compiled executable HELENA binary.
            other_params (dict): Dictionary containing other parameters for initialization.

        other_params:
            namelist_path : str
                The namelist constaining the HELENA values to be kept constant
                during the run.

            only_generate_files: bool
                Flag for either only creating input files or creating the files
                and running HELENA.

            beta_iteration: bool
                Flag for iterating HELENA to find target beta normalized. Requires that 'beta_N'
                exists in the sampler parameters.

            beta_tolerance: float
                (Only relevant when beta_iteration is True.)
                Tolerance criteria for calculated betan vs target betan.

            max_beta_iterations: float
                (Only relevant when beta_iteration is True.)
                The maximum number of beta iterations.

            input_parameter_type: int
                0: EPED/MTANH profiles using direct helena input ADE, BDE, etc.
                1: Europed profiles generated using 'pedestal_delta', 'n_eped', 'bvac', 'ip',
                'teped_multip'
                2: using a scaling law for changing ATE and CTE, requires input parameter
                "scaling_factor"

            run_mishka: bool
                Flag for running MISHKA after the HELENA run. If run_mishka is True
                the following other parameters need to be defined in other_params:
                - mishka_executable_path: str
                - mishka_namelist_path: str
                - ntor: list of int

        """
        print("HELENArunner initializing...")
        self.parser = HELENAparser()
        self.executable_path = executable_path
        self.namelist_path = other_params["namelist_path"]
        self.only_generate_files = (
            False
            if "only_generate_files" not in other_params
            else other_params["only_generate_files"]
        )
        self.beta_iteration = (
            False
            if "beta_iteration" not in other_params
            else other_params["beta_iteration"]
        )
        self.beta_tolerance = (
            1e-4
            if "beta_tolerance" not in other_params
            else other_params["beta_tolerance"]
        )
        self.max_beta_iterations = (
            10
            if "max_beta_iterations" not in other_params
            else other_params["max_beta_iterations"]
        )
        self.input_parameter_type = (
            0
            if "input_parameter_type" not in other_params
            else other_params["input_parameter_type"]
        )
        self.pedestal_width_scan = (
            False
            if "pedestal_width_scan" not in other_params
            else other_params["pedestal_width_scan"]
        )

        self.pedestal_width_scan_params = (
            {
                "min_width": 0.02,
                "max_width": 0.06,
                "max_iterations": 5,
            }
            if "pedestal_width_scan_params" not in other_params
            else other_params["pedestal_width_scan_params"]
        )

        self.run_mishka = False
        self.mishka_runner = None

        if "mishka" in other_params:
            mishka_params = other_params["mishka"]
            self.run_mishka = (
                False
                if "run_mishka" not in mishka_params
                else mishka_params["run_mishka"]
            )
            if self.run_mishka:
                if (
                    "executable_path" not in mishka_params
                    or "ntor" not in mishka_params
                ):
                    print(
                        'Parameters "executable_path" or "ntor" missing in mishka section ',
                        "in the config file. Cannot initiate MISHKA runner. ",
                        "Only the HELENA part will run.",
                    )
                    self.run_mishka = False
                else:
                    self.mishka_runner = MISHKArunner(
                        executable_path=mishka_params["executable_path"],
                        other_params=mishka_params["other_params"],
                    )
                    self.mishka_ntor_samples = mishka_params["ntor"]

        self.beta_iterations_afp = (
            False
            if "beta_iterations_afp" not in other_params
            else other_params["beta_iterations_afp"]
        )

        self.pre_run_check()

    def single_code_run(self, params: dict, run_dir: str):
        """
        Runs HELENA simulation.

        Args:
            params (dict): Dictionary containing parameters for the simulation.
            run_dir (str): Directory where HELENA is run.
            tolerance (float): Tolerance for beta iteration

        Returns:
            bool: True if the simulation is successful, False otherwise.

        """
        print(f"single_code_run: {run_dir}", flush=True)
        if self.input_parameter_type == 0:
            self.parser.write_input_file(params, run_dir, self.namelist_path)
        elif self.input_parameter_type == 1:
            self.parser.write_input_file_europed(params, run_dir, self.namelist_path)
        elif self.input_parameter_type == 2:
            self.parser.write_input_file_scaling(params, run_dir, self.namelist_path)
        elif self.input_parameter_type == 3:
            self.parser.write_input_file_europed2(params, run_dir, self.namelist_path)
        elif self.input_parameter_type == 4:
            self.parser.write_input_file_noKBM(params, run_dir, self.namelist_path)
        else:
            self.parser.write_input_file(params, run_dir, self.namelist_path)

        # Exit here
        if self.only_generate_files:
            return True

        # Check input parameters
        if self.beta_iteration:
            if "beta_N" not in params:
                print(
                    "The parameter configuration does not include 'beta_N'.",
                    "This it needed for beta iteration. EXITING.",
                )
                return False
            if "apf" in params:
                print(
                    "The core ion pressure parameter APF should not be ",
                    "part of the input parameters during beta iteration. EXITING",
                )
                return False

        # Needed for HELENA
        os.chdir(run_dir)

        # Single equilibrium run
        if not self.pedestal_width_scan:
            if self.beta_iteration:
                self.run_helena_with_beta_iteration(params)
            else:
                subprocess.call([self.executable_path])

            # process output
            run_successful = self.parser.write_summary(run_dir, params)
            self.parser.clean_output_files(run_dir)

            # run MISHKA
            if run_successful:
                if self.run_mishka:
                    self.run_mishka_for_ntors(run_dir)
                    self.parser.collect_growthrates_from_mishka(run_dir, save=True)
            else:
                print("HELENA run not success. MISHKA is not being run.")

        # Run for multiple pedestal widths
        else:
            sampling_method = 1

            if self.input_parameter_type == 3:
                beta_target = params["beta_N"]
                if sampling_method == 0:
                    # Evenly spaced sample points
                    d_ped_scans = np.linspace(
                        self.pedestal_width_scan_params["min_width"],
                        self.pedestal_width_scan_params["max_width"],
                        self.pedestal_width_scan_params["max_iterations"],
                    )

                elif sampling_method == 1:
                    # Uniform random sampling
                    d_ped_scans = np.random.uniform(
                        self.pedestal_width_scan_params["min_width"],
                        self.pedestal_width_scan_params["max_width"],
                        self.pedestal_width_scan_params["max_iterations"],
                    )

                    for _i, d_ped_scan in enumerate(d_ped_scans):
                        try:
                            # scan_dir = os.path.join(run_dir, f"scan_{_i}")
                            scan_dir = f"{run_dir}_scan_{_i}"
                            os.mkdir(scan_dir)
                            os.chdir(scan_dir)
                            self.parser.update_pedestal_delta(
                                d_ped_scan, beta_target, run_dir, scan_dir
                            )
                            subprocess.call([self.executable_path])

                            at1 = self.parser.find_new_at1(
                                output_dir=scan_dir, beta_target=beta_target, h=0.01
                            )
                            self.parser.update_at1(
                                namelist_path=os.path.join(scan_dir, "fort.10"), at1=at1
                            )

                            subprocess.call([self.executable_path])
                            run_successful = self.parser.write_summary(scan_dir, params)
                            self.parser.clean_output_files(scan_dir)
                            if run_successful:
                                if self.run_mishka:
                                    self.run_mishka_for_ntors(scan_dir)
                                    self.parser.collect_growthrates_from_mishka(
                                        scan_dir, save=True
                                    )
                        except Exception as exc:
                            print(exc)
                elif sampling_method == 2:
                    # Binary search for stability boundary
                    # self.pedestal_width_scan_params["max_iterations"]
                    min_width = self.pedestal_width_scan_params["min_width"]
                    max_width = self.pedestal_width_scan_params["max_width"]
                    d_ped_initial_scan = np.random.uniform(
                        min_width,
                        max_width,
                        1,
                    )
                    # TODO: implement stability search
                    # d_ped_scans = np.linspace(0.025, 0.045, 3)
                    # for _i, d_ped_scan in enumerate(d_ped_scans):
                    #     try:
                    #         # scan_dir = os.path.join(run_dir, f"scan_{_i}")
                    #         scan_dir = f"{run_dir}_scan_{_i}"
                    #         os.mkdir(scan_dir)
                    #         os.chdir(scan_dir)
                    #         self.parser.update_pedestal_delta(
                    #             d_ped_scan, beta_target, run_dir, scan_dir
                    #         )
                    #         subprocess.call([self.executable_path])

                    #         at1 = self.parser.find_new_at1(
                    #             output_dir=scan_dir, beta_target=beta_target, h=0.01
                    #         )
                    #         self.parser.update_at1(
                    #             namelist_path=os.path.join(scan_dir, "fort.10"), at1=at1
                    #         )

                    #         subprocess.call([self.executable_path])
                    #         run_successful = self.parser.write_summary(scan_dir, params)
                    #         self.parser.clean_output_files(scan_dir)
                    #         if run_successful:
                    #             if self.run_mishka:
                    #                 self.run_mishka_for_ntors(scan_dir)
                    #                 self.parser.collect_growthrates_from_mishka(
                    #                     scan_dir, save=True
                    #                 )
                    #     except Exception as exc:
                    #         print(exc)

                else:
                    print("ERROR: Sampling method not implemented.")
            else:
                print(
                    "ERROR: Pedestal width scan is only implemented for input_parameter_type = 3."
                )
            print("HELENArunner finished.")

        return True

    def run_helena_with_beta_iteration(self, params):
        beta_target = params["beta_N"]
        print(f"BETA ITERATION STARTED. Target betaN = {beta_target}\n")
        if self.beta_iterations_afp:
            self.parser.modify_fast_ion_pressure("fort.10", 0.0)
        else:
            self.parser.modify_at1("fort.10", 0.0)
        subprocess.call([self.executable_path])
        output_vars = self.parser.get_real_world_geometry_factors_from_f20("fort.20")
        beta_n0 = 1e2 * output_vars["BETAN"]
        print(f"BETA ITERATION {0.0}: target={beta_target}, current={beta_n0}")
        if self.beta_iterations_afp:
            self.parser.modify_fast_ion_pressure("fort.10", 0.1)
        else:
            self.parser.modify_at1("fort.10", 1.0)
        subprocess.call([self.executable_path])
        output_vars = self.parser.get_real_world_geometry_factors_from_f20("fort.20")
        beta_n01 = 1e2 * output_vars["BETAN"]
        print(f"BETA ITERATION {0.1}: target={beta_target}, current={beta_n01}")
        if self.beta_iterations_afp:
            apftarg = (beta_target - beta_n0) * 0.1 / (beta_n01 - beta_n0)
            self.parser.modify_fast_ion_pressure("fort.10", apftarg)
        else:
            at1_mult_targ = (beta_target - beta_n0) / (beta_n01 - beta_n0)
            self.parser.modify_at1("fort.10", at1_mult_targ)

        subprocess.call([self.executable_path])
        output_vars = self.parser.get_real_world_geometry_factors_from_f20("fort.20")
        beta_n = 1e2 * output_vars["BETAN"]
        print(f"BETA ITERATION {0.2}: target={beta_target}, current={beta_n}")
        n_beta_iteration = 0
        while (
            np.abs(beta_target - beta_n) > self.beta_tolerance * beta_target
            and n_beta_iteration < self.max_beta_iterations
        ):
            if self.beta_iterations_afp:
                apftarg = (beta_target - beta_n0) * apftarg / (beta_n - beta_n0)
                self.parser.modify_fast_ion_pressure("fort.10", apftarg)
            else:
                at1_mult_targ = (beta_target - beta_n0) / (beta_n01 - beta_n0)
                self.parser.modify_at1("fort.10", at1_mult_targ)
            subprocess.call([self.executable_path])
            output_vars = self.parser.get_real_world_geometry_factors_from_f20(
                "fort.20"
            )
            beta_n = 1e2 * output_vars["BETAN"]
            print(
                f"BETA ITERATION {n_beta_iteration}: target={beta_target}, current={beta_n}"
            )
            n_beta_iteration += 1

        print(
            f"BETA ITERATION FINISHED.\nTarget betaN: {beta_target}\n",
            f"Final betaN: {beta_n}\n",
            f"Number of beta iterations: {n_beta_iteration}",
        )

    def run_mishka_for_ntors(self, run_dir):
        mishka_dir = os.path.join(run_dir, "mishka")
        os.mkdir(mishka_dir)
        for ntor_sample in self.mishka_ntor_samples:
            mishka_run_dir = os.path.join(mishka_dir, str(ntor_sample))
            os.mkdir(mishka_run_dir)
            self.mishka_runner.single_code_run(
                params={"ntor": ntor_sample, "helena_dir": run_dir},
                run_dir=mishka_run_dir,
            )

    def pre_run_check(self):
        """
        Performs pre-run checks to ensure necessary files exist before running the simulation.

        Raises:
            FileNotFoundError: If the executable path or the namelist path is not found.

        """
        if not os.path.isfile(self.executable_path):
            raise FileNotFoundError(
                f"The executable path ({self.executable_path}) provided to the HELENA ",
                "runner is not found. Exiting.",
            )

        if not os.path.isfile(self.namelist_path):
            raise FileNotFoundError(
                f"The namelist path ({self.namelist_path}) provided to the HELENA ",
                "runner is not found. Exiting.",
            )

        return
