import os
import f90nml
import numpy as np
from .base import Parser

# import subprocess


class HELENAparser(Parser):
    """An I/O parser for HELENA

    Attributes
    ----------


    Methods
    -------
    write_input_file
        Writes the inputfile fort.10.

    read_output_file_fort20
        Not implemented.

    read_output_file_fort22
        Reads the output file fort.22.
    """

    def __init__(self):
        self.default_namelist = ""

    def write_input_file(self, params: dict, run_dir: str, namelist_path: str):
        print(run_dir, params)
        print("Writing to", run_dir)
        if os.path.exists(run_dir):
            input_fpath = os.path.join(run_dir, "fort.10")
        else:
            raise FileNotFoundError(f"Couldnt find {run_dir}")

        namelist = f90nml.read(namelist_path)

        # Update namelist with input parameters
        for key, value in params.items():
            if key == "tria":
                namelist["shape"]["tria"] = value
            elif key == "ellip":
                namelist["shape"]["ellip"] = value
            elif key == "bvac":
                namelist["phys"]["bvac"] = value
            elif key == "pedestal_delta":
                namelist["shape"]["xr2"] = 1 - value / 2.0
                namelist["shape"]["sig2"] = value

        # Europed profiles
        if "pedestal_delta" in params:
            # Update density profile parameters
            density_shift = 0.0
            an0 = (
                1.0
                if ("n_esep" not in params or "n_eped" not in params)
                else (params["n_eped"] - params["n_esep"]) / (np.tanh(1) * 2)
            )
            namelist["phys"]["ade"] = an0
            namelist["phys"]["bde"] = (
                0.725 if "n_esep" not in params else params["n_esep"]
            )
            namelist["phys"]["cde"] = 1.0 if "an1" not in params else params["an1"]
            namelist["phys"]["dde"] = (
                1.0 - 0.5 * params["pedestal_delta"] + density_shift
            )
            namelist["phys"]["ede"] = params["pedestal_delta"]
            namelist["phys"]["fde"] = 1 - params["pedestal_delta"]
            namelist["phys"]["gde"] = (
                1.1 if "alpha_n1" not in params else params["alpha_n1"]
            )
            namelist["phys"]["hde"] = (
                1.1 if "alpha_n2" not in params else params["alpha_n2"]
            )

            # Update electron temperature profile
            at0 = (
                1.0
                if ("T_esep" not in params or "T_eped" not in params)
                else (params["T_eped"] - params["T_esep"]) / (np.tanh(1) * 2)
            )
            namelist["phys"]["ate"] = at0
            namelist["phys"]["bte"] = (
                0.1 if "T_esep" not in params else params["T_esep"]
            )
            namelist["phys"]["cte"] = 0.1 if "aT1" not in params else params["aT1"]
            namelist["phys"]["dte"] = (
                1.0 - 0.5 * params["pedestal_delta"] + density_shift
            )
            namelist["phys"]["ete"] = params["pedestal_delta"]
            namelist["phys"]["fte"] = 1 - params["pedestal_delta"]
            namelist["phys"]["gte"] = (
                1.2 if "alpha_T1" not in params else params["alpha_T1"]
            )
            namelist["phys"]["hte"] = (
                1.4 if "alpha_T2" not in params else params["alpha_T2"]
            )

            # Update ion temperature profile = electron temperature profile
            namelist["phys"]["ati"] = namelist["phys"]["ate"]
            namelist["phys"]["bti"] = namelist["phys"]["bte"]
            namelist["phys"]["cti"] = namelist["phys"]["cte"]
            namelist["phys"]["ddti"] = namelist["phys"]["ddte"]
            namelist["phys"]["eti"] = namelist["phys"]["ete"]
            namelist["phys"]["fti"] = namelist["phys"]["fte"]
            namelist["phys"]["gti"] = namelist["phys"]["gte"]
            namelist["phys"]["hti"] = namelist["phys"]["hte"]

        # Update current profile guess
        namelist["profile"]["zjz"] = self.make_init_zjz_profile(
            pedestal_delta=params["pedestal_delta"], npts=namelist["profile"]["npts"]
        )

        f90nml.write(namelist, input_fpath)
        print(f"fort.10 written to: {input_fpath}")

    def read_output_file(self, run_dir: str):
        """
        The main output file fort.20.
        """
        raise NotImplementedError

    def make_init_zjz_profile(self, pedestal_delta, npts):
        """
        Makes the initial ZJZ profile based on the pressure profile
        according to Europed implementation.
        """
        alpha1, alpha2 = 1.0, 1.5
        x = np.linspace(0, 1, npts)

        base = 0.9 * (1 - x**alpha1) ** alpha2 + 0.1 * (
            1 + np.tanh((1 - pedestal_delta / 2 - x) / pedestal_delta * 2)
        )

        pzjzmultip = 0.5  # TODO: as input?
        max_pres_grad_loc = 0.97  # TODO: add calculation from europed

        pedestal_current = pzjzmultip * np.exp(
            -(((max_pres_grad_loc - x) / pedestal_delta * 1.5) ** 2)
        )

        return base + pedestal_current

    def get_europed_profiles(self, run_dir: str):
        """
        Returns
        -------
        electron density profile

        electron temperature profile
        """
        namelist = f90nml.read(os.path.join(run_dir, "fort.10"))
        psi = np.linspace(0, 1, namelist["num"]["nr"])
        n_e = []
        T_e = []
        for p in psi:
            n_e.append(
                self.europed_tanh_profile(
                    psi=p,
                    psi_mid=namelist["phys"]["dde"],
                    psi_ped=namelist["phys"]["fde"],
                    a_0=namelist["phys"]["ade"],
                    a_1=namelist["phys"]["cde"],
                    sep=namelist["phys"]["bde"],
                    alpha_1=namelist["phys"]["gde"],
                    alpha_2=namelist["phys"]["hde"],
                    delta=namelist["phys"]["ede"],
                )
            )
            T_e.append(
                self.europed_tanh_profile(
                    psi=p,
                    psi_mid=namelist["phys"]["ddte"],
                    psi_ped=namelist["phys"]["fte"],
                    a_0=namelist["phys"]["ate"],
                    a_1=namelist["phys"]["cte"],
                    sep=namelist["phys"]["bte"],
                    alpha_1=namelist["phys"]["gte"],
                    alpha_2=namelist["phys"]["hte"],
                    delta=namelist["phys"]["ete"],
                )
            )
        return n_e, T_e

    def europed_tanh_profile(
        self, psi, psi_mid, psi_ped, a_0, a_1, sep, alpha_1, alpha_2, delta
    ):
        """
        Used for temperature and density profiles
        """
        term1 = psi_mid * (
            np.tanh(2 / delta * (1 - psi_mid)) - np.tanh(2 / delta * (psi - psi_mid))
        )
        if 1 - psi / psi_ped >= 0:
            term2 = a_1 * (1 - (psi / psi_ped) ** alpha_1) ** alpha_2
        else:
            term2 = 0.0

        return sep + a_0 * term1 + term2
