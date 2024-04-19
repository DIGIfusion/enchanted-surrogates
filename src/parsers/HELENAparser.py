import os
import f90nml
import numpy as np
import yaml
from .base import Parser

# import subprocess

mu_0 = 4e-7 * np.pi
ElectronCharge = 1.6022e-19


class HELENAparser(Parser):
    """An I/O parser for HELENA

    Attributes
    ----------
    None

    Methods
    -------
    write_input_file(params: dict, run_dir: str, namelist_path: str)
        Writes the input file fort.10.

    read_output_file(run_dir: str)
        Reads the main output file fort.20.
        Returns a tuple indicating success, Mercier criterion presence, and ballooning criterion presence.

    make_init_zjz_profile(pedestal_delta, npts)
        Makes the initial ZJZ profile based on the pressure profile according to Europed implementation.

    get_europed_profiles(run_dir: str)
        Retrieves Europed electron density and temperature profiles from the output file fort.10.

    europed_tanh_profile(psi, psi_mid, psi_ped, a_0, a_1, sep, alpha_1, alpha_2, delta)
        Calculates the Europed tanh profile for temperature and density profiles.

    clean_output_files(run_dir: str)
        Removes unnecessary files except for fort.10, fort.20, and fort.12.

    write_summary(run_dir: str, params: dict)
        Generates a summary file with run directory and parameters, along with success and stability criteria.

    """

    def __init__(self):
        self.default_namelist = ""

    def write_input_file(self, params: dict, run_dir: str, namelist_path: str):
        """
        Writes input file fort.10.

        Parameters
        ----------
        params : dict
            Dictionary containing input parameters.
        run_dir : str
            Path to the run directory.
        namelist_path : str
            Path to the namelist file.

        Raises
        ------
        FileNotFoundError
            If the specified run directory is not found.
        """
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

        if "bvac" in params and "ip" in params:
            namelist["phys"]["bvac"] = value
            namelist["phys"]["xiab"] = (
                params["ip"]
                * mu_0
                / (params["bvac"] * namelist["phys"]["eps"] * namelist["phys"]["rvac"])
            )
        # Europed profiles
        if "pedestal_delta" in params:
            # Update density profile parameters
            d_ped = params["pedestal_delta"]
            density_shift = 0.0
            psi_mid = 1.0 - 0.5 * d_ped + density_shift

            nesep = 0.725 if "n_esep" not in params else params["n_esep"]
            neped = 3.0 if "n_eped" not in params else params["n_eped"]
            an0 = (neped - nesep) / (np.tanh(1) * 2)
            an1 = 1.0 if "an1" not in params else params["an1"]

            namelist["phys"]["ade"] = an0
            namelist["phys"]["bde"] = nesep
            namelist["phys"]["cde"] = an1
            namelist["phys"]["dde"] = psi_mid
            namelist["phys"]["ede"] = d_ped
            namelist["phys"]["fde"] = 1 - d_ped
            namelist["phys"]["gde"] = (
                1.1 if "alpha_n1" not in params else params["alpha_n1"]
            )
            namelist["phys"]["hde"] = (
                1.1 if "alpha_n2" not in params else params["alpha_n2"]
            )

            # Update electron temperature profile
            tesep = 0.1 if "T_esep" not in params else params["T_esep"]
            teped = 1.0 if "T_eped" not in params else params["T_eped"]
            at0 = (teped - tesep) / (np.tanh(1) * 2)
            if "core_to_ped_te" in params:
                at1 = params["core_to_ped_te"] * teped
            else:
                at1 = 0.1 if "aT1" not in params else params["aT1"]

            namelist["phys"]["ate"] = at0
            namelist["phys"]["bte"] = tesep
            namelist["phys"]["cte"] = at1
            namelist["phys"]["ddte"] = psi_mid
            namelist["phys"]["ete"] = d_ped
            namelist["phys"]["fte"] = 1 - d_ped
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

            if namelist["phys"]["ipai"] != 18:
                coret = (teped * 2 + at1 + tesep) * 1000
                coren = an0 * 2 + nesep + an1
                corep = (
                    coret
                    * (
                        1
                        + (namelist["phys"]["zimp"] + 1 - namelist["phys"]["zeff"])
                        / namelist["phys"]["zimp"]
                    )
                    * coren
                    * ElectronCharge
                    * 1e19
                )
                namelist["phys"]["corep"] = corep

        # Update current profile guess
        namelist["profile"]["zjz"] = self.make_init_zjz_profile(
            pedestal_delta=params["pedestal_delta"], npts=namelist["profile"]["npts"]
        )

        f90nml.write(namelist, input_fpath)
        print(f"fort.10 written to: {input_fpath}")

    def read_output_file(self, run_dir: str):
        """
        Reads the main output file fort.20.
        Crashed: if 'ALPHA1' is not found or 'MERCIER' is not found

        Parameters
        ----------
        run_dir : str
            Path to the run directory.

        Returns
        -------
        tuple
            A tuple indicating success (bool), Mercier criterion presence (bool), and ballooning criterion presence (bool).
        """
        success = False
        ballooning = True
        mercier = True

        try:
            file = open(run_dir + "fort.20", "r")
        except FileNotFoundError as e:
            print(str(e))
            return success, mercier, ballooning

        for line in file.readlines():
            if line.find("ALPHA1") > -1:
                success = True
            if line.find("MERCIER") > -1:
                success = True

        return success, mercier, ballooning

    def make_init_zjz_profile(self, pedestal_delta, npts):
        """
        Makes the initial ZJZ profile based on the pressure profile according to Europed implementation.

        Parameters
        ----------
        pedestal_delta : float
            Pedestal delta value.
        npts : int
            Number of points.

        Returns
        -------
        array_like
            Array representing the initial ZJZ profile.
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
        Retrieves Europed electron density and temperature profiles from the output file fort.10.

        Parameters
        ----------
        run_dir : str
            Path to the run directory.

        Returns
        -------
        tuple
            Tuple containing electron density profile and electron temperature profile.
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
        Calculates the Europed tanh profile for temperature and density profiles.

        Parameters
        ----------
        psi : float
            Psi value.
        psi_mid : float
            Mid psi value.
        psi_ped : float
            Pedestal psi value.
        a_0 : float
            Coefficient a_0.
        a_1 : float
            Coefficient a_1.
        sep : float
            Separatrix value.
        alpha_1 : float
            Alpha 1 value.
        alpha_2 : float
            Alpha 2 value.
        delta : float
            Delta value.

        Returns
        -------
        float
            Calculated tanh profile value.
        """
        term1 = np.tanh(2 / delta * (1 - psi_mid)) - np.tanh(
            2 / delta * (psi - psi_mid)
        )
        if 1 - psi / psi_ped >= 0:
            term2 = a_1 * (1 - (psi / psi_ped) ** alpha_1) ** alpha_2
        else:
            term2 = 0.0

        return sep + a_0 * term1 + term2

    def clean_output_files(self):
        """
        Removes unnecessary files except for fort.10, fort.20, and fort.12.
        (fort.30 is an input-output file which can be useful)

        Parameters
        ----------
        run_dir : str
            Path to the run directory.
        """
        files_to_remove = [
            "fort.21",
            "fort.25",
            "fort.27",
            "fort.41",
            "eliteinp",
            "density",
            "helena_bnd",
            "EQDSK",
            "PCUBEQ",
        ]
        for file_name in files_to_remove:
            if os.path.exists(file_name):
                os.remove(file_name)

        return

    def write_summary(self, run_dir: str, params: dict):
        """
        Generates a summary file with run directory and parameters, along with success and stability criteria.

        Parameters
        ----------
        run_dir : str
            Path to the run directory.
        params : dict
            Dictionary containing input parameters.

        Returns
        -------
        dict
            Summary dictionary containing run directory, parameters, success status, Mercier criterion presence, and ballooning criterion presence.
        """
        file_name = "summary.yml"
        summary = {"run_dir": run_dir, "params": params}
        success, mercier, ballooning = self.read_output_file(run_dir)
        summary["success"] = success
        summary["mercier"] = mercier
        summary["ballooning"] = ballooning
        with open(file_name, "w") as outfile:
            yaml.dump(summary, outfile, default_flow_style=False)
        return summary
