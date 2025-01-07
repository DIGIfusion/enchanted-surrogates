import os
import math
import f90nml
import numpy as np
import json
from .base import Parser
import fortranformat as ff
import scipy

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
        Returns a tuple indicating success, Mercier criterion presence,
        and ballooning criterion presence.

    make_init_zjz_profile(pedestal_delta, npts)
        Makes the initial ZJZ profile based on the pressure profile according
        to Europed implementation.

    get_europed_profiles(run_dir: str)
        Retrieves Europed electron density and temperature profiles from the output file fort.10.

    europed_tanh_profile(psi, psi_mid, psi_ped, a_0, a_1, sep, alpha_1, alpha_2, delta)
        Calculates the Europed tanh profile for temperature and density profiles.

    clean_output_files(run_dir: str)
        Removes unnecessary files except for fort.10, fort.20, and fort.12.

    write_summary(run_dir: str, params: dict)
        Generates a summary file with run directory and parameters, along with success
        and stability criteria.

    modify_fast_ion_pressure(namelistpath: str, apf: float)
        Changes the apf value in the HELENA input file to the given value.

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
        helena_shape_params = ["tria", "ellip", "quad"]
        for p in helena_shape_params:
            if p in params:
                namelist["shape"][p] = params[p]

        if "bvac" in params:
            namelist["phys"]["bvac"] = params["bvac"]
            if "ip" in params:
                namelist["phys"]["xiab"] = (
                    params["ip"]
                    * 1e6
                    * mu_0
                    / (
                        params["bvac"]
                        * namelist["phys"]["eps"]
                        * namelist["phys"]["rvac"]
                    )
                )
        # Europed profiles
        if "pedestal_delta" in params:
            d_ped = params["pedestal_delta"]
            namelist["shape"]["xr2"] = 1 - d_ped / 2.0
            namelist["shape"]["sig2"] = d_ped

            # Update density profile parameters
            density_shift = 0.0
            psi_mid = 1.0 - 0.5 * d_ped + density_shift
            # In anticipation of actually using the density_shift eventually,
            # the temperature delta is separated below. (AEJ) 2.1.2025.
            psi_mid_T = 1.0 - 0.5 * d_ped

            nesep = 0.725 if "n_esep" not in params else params["n_esep"]
            neped = 3.0 if "n_eped" not in params else params["n_eped"]
            an0 = (neped - nesep) / (np.tanh(1) * 2)
            an1 = 0.1 if "an1" not in params else params["an1"]

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
            if "teped_multip" in params and "ip" in params:
                teped = params["teped_multip"] * params["ip"] ** 2 / neped
            else:
                teped = 1.0 if "T_eped" not in params else params["T_eped"]
            at0 = (teped - tesep) / (np.tanh(1) * 2)
            if "core_to_ped_te" in params:
                at1 = params["core_to_ped_te"] * teped
            else:
                at1 = 0.1 if "aT1" not in params else params["aT1"]

            namelist["phys"]["ate"] = at0
            namelist["phys"]["bte"] = tesep
            namelist["phys"]["cte"] = at1
            namelist["phys"]["ddte"] = psi_mid_T
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

            if namelist["profile"]["ipai"] != 18:
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
        if "pedestal_delta" in params:
            namelist["profile"]["zjz"] = self.make_init_zjz_profile(
                pedestal_delta=params["pedestal_delta"],
                npts=namelist["profile"]["npts"],
            )
        else:
            namelist["profile"]["zjz"] = self.make_init_zjz_profile(
                pedestal_delta=namelist["phys"]["ede"], npts=namelist["profile"]["npts"]
            )
        # teped_multip

        f90nml.write(namelist, input_fpath)
        print(f"fort.10 written to: {input_fpath}")

    def read_output_file(self, run_dir: str):
        """
        Reads the main output file fort.20.
        Crashed: if 'ALPHA1' is not found or 'MERCIER' is not found.

        Mercier stability
        Ballooning stability
        NeoClassical Stability Condition

        Parameters
        ----------
        run_dir : str
            Path to the run directory.

        Returns
        -------
        tuple
            A tuple indicating success (bool), Mercier criterion presence (bool),
            and ballooning criterion presence (bool).
        """
        success = False
        ballooning_stable = True
        mercier_stable = True

        try:
            file = open(run_dir + "/fort.20", "r")
        except FileNotFoundError as e:
            print(str(e))
            return success, mercier_stable, ballooning_stable

        for line in file.readlines():
            if line.find("UNSTABLE AT T =") > -1:
                ballooning_stable = False
            if line.find("ALPHA1") > -1:
                success = True
            if line.find("MERCIER") > -1:
                success = True
            if line.find("Mercier Unstable") > -1:
                mercier_stable = False

        return success, mercier_stable, ballooning_stable

    def make_init_zjz_profile(
        self, pedestal_delta, npts, pzjzmultip=0.5, max_pres_grad_loc=0.97
    ):
        """
        Makes the initial ZJZ profile based on the pressure profile according
        to Europed implementation.

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

    def clean_output_files(self, run_dir: str):
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
            "fort.51",
            "eliteinp",
            "density",
            "helena_bnd",
            "EQDSK",
            "PCUBEQ",
        ]
        for file_name in files_to_remove:
            if os.path.exists(run_dir + "/" + file_name):
                os.remove(run_dir + "/" + file_name)

        return

    def write_summary(self, run_dir: str, params: dict):
        """
        Generates a summary file with run directory and parameters, along with
        success and stability criteria.

        Parameters
        ----------
        run_dir : str
            Path to the run directory.
        params : dict
            Dictionary containing input parameters.

        Returns
        -------
        dict
            Summary dictionary containing run directory, parameters,
            success status, Mercier criterion presence, and ballooning
            criterion presence.
        """
        file_name = run_dir + "/summary.yml"
        summary = {"run_dir": run_dir, "params": params}
        success, mercier_stable, ballooning_stable = self.read_output_file(run_dir)
        summary["success"] = success
        summary["mercier_stable"] = mercier_stable
        summary["ballooning_stable"] = ballooning_stable
        with open(file_name, "w") as outfile:
            json.dump(summary, outfile)
        try:
            CS, QS, _, _, _, _, _, _, P0, RBPHI, VX, VY = self.read_output_fort12(
                run_dir
            )
            np.save(run_dir + "/qs.npy", QS)
            np.save(run_dir + "/cs.npy", CS)
            np.save(run_dir + "/p0.npy", P0)
            np.save(run_dir + "/rbphi.npy", RBPHI)
            np.save(run_dir + "/vxvy.npy", (VX, VY))
        except Exception:
            print(f"error reading and summarizing fort.12 in dir {run_dir}")
        return summary

    def read_final_zjz(self, output_dir):
        """
        zjz(   1)=     1.00000,
        zjz(   2)=     1.01337,
        zjz(   3)=     0.83545,
        """

        filename = output_dir + "/final_zjz"
        file = open(filename, "r")
        lines = file.readlines()
        zjz = []
        for line in lines:
            zjz.append(float(line.split("=")[-1].replace(",", "")))

        return np.array(zjz)

    def read_multiline_list(
        self, lines, startline, n_listitems, n_columns=4, dtype=float
    ):
        n_rows = math.ceil(n_listitems / n_columns)
        endline = startline + n_rows
        lines = lines[startline:endline]
        res = []
        for line in lines:
            res = res + line.split()
        res = [float(x) for x in res]
        return res, endline

    def read_output_fort12(self, output_dir):
        """
        Read the output file fort.12 which is used as input by MISHKA.
        """
        filename = output_dir + "/" + "fort.12"
        file = open(filename, "r")
        lines = file.readlines()
        JS0 = int(lines[0].split()[0])
        CS, endline = self.read_multiline_list(lines, startline=1, n_listitems=JS0 + 1)
        QS, endline = self.read_multiline_list(
            lines, startline=endline, n_listitems=JS0 + 1
        )
        DQS_1, DQEC = float(lines[endline].split()[0]), float(lines[endline].split()[1])
        DQS, endline = self.read_multiline_list(
            lines, startline=endline + 1, n_listitems=JS0
        )
        CURJ, endline = self.read_multiline_list(
            lines, startline=endline, n_listitems=JS0 + 1
        )
        DJ0, DJE = float(lines[endline].split()[0]), float(lines[endline].split()[1])
        NCHI = int(lines[endline + 1].split()[0])
        CHI, endline = self.read_multiline_list(
            lines, startline=endline + 2, n_listitems=NCHI + 1
        )
        GEM11, endline = self.read_multiline_list(
            lines,
            startline=endline,
            n_listitems=NCHI * (JS0 + 1) - (NCHI + 1),
        )
        GEM12, endline = self.read_multiline_list(
            lines,
            startline=endline,
            n_listitems=NCHI * (JS0 + 1) - (NCHI + 1),
        )
        CPSURF, RADIUS = float(lines[endline].split()[0]), float(
            lines[endline].split()[1]
        )
        GEM33, endline = self.read_multiline_list(
            lines,
            startline=endline + 1,
            n_listitems=NCHI * (JS0 + 1) - (NCHI + 1),
        )
        RAXIS = float(lines[endline].split()[0])
        P0, endline = self.read_multiline_list(
            lines, startline=endline + 1, n_listitems=JS0 + 1
        )
        DP0, DPE = float(lines[endline].split()[0]), float(lines[endline].split()[1])
        RBPHI, endline = self.read_multiline_list(
            lines, startline=endline + 1, n_listitems=JS0 + 1
        )
        DRBPHI0, DRBPHIE = float(lines[endline].split()[0]), float(
            lines[endline].split()[1]
        )

        # Vacuum data
        VX, endline = self.read_multiline_list(
            lines, startline=endline + 1, n_listitems=NCHI
        )
        VY, endline = self.read_multiline_list(
            lines, startline=endline, n_listitems=NCHI
        )
        EPS = float(lines[endline].split()[0])
        # XOUT, endline = self.read_multiline_list(
        #     lines,
        #     startline=endline + 1,
        #     n_listitems=NCHI * (JS0 + 1) - (NCHI + 1),
        # )
        # YOUT, endline = self.read_multiline_list(
        #     lines,
        #     startline=endline,
        #     n_listitems=NCHI * (JS0 + 1) - (NCHI + 1),
        # )
        return CS, QS, DQS, CURJ, CHI, GEM11, GEM12, GEM33, P0, RBPHI, VX, VY

    def modify_fast_ion_pressure(self, namelist_path: str, apf: float):
        """
        Changes the fast ion pressure (apf) value in the input file fort.10.

        Parameters
        ----------
        namelist_path : str
            Path to the namelist file.
        apf: float
            Value for the fast ion pressure parameter.

        Raises
        ------
        FileNotFoundError
            If the specified run directory is not found.
        """
        namelist = f90nml.read(namelist_path)
        namelist["profile"]["apf"] = apf
        f90nml.write(namelist, namelist_path, force=True)

    def get_real_world_geometry_factors_from_f20(self, f20_fort: str):
        """
        Function copy paste from tokamak_sampler by A. Kit
        Get Geometry factors from fort.20

        Parameters
        ----------
        f20_fort : str
            Path to the fort.20 file

        Returns
        ----------
        Dictionary with parameter names below:
            Read: BVAC, RVAC, EPS, XAXIS, CPSURF, ALFA, BETAP, BETAT, BETAN, XIAB
            Derived: RADIUS (minor radius), RMAGAXIS, CURRENT

            - XIAB: normalized total current: XIAB = mu_0 I / (a*B0))
                -> CURRENT = (XIAB / MU_0) * (a*B0)
        """
        BVAC, RVAC, EPS, XAXIS, CPSURF, ALFA, BETAP, BETAT, BETAN, XIAB = (
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )
        output_vars = {}
        with open(f20_fort, "r") as f:
            lines = f.readlines()

            """ 
            Looking for
            $PHYS     EPS  =  0.320, ALFA =  2.438, B =  0.006, C =  1.000,
                    XIAB =  1.350, BETAP =  -1.0000,COREP =   43259.3989
            """
            for i, line in enumerate(lines):
                if "$PHYS" in line:
                    break
            vars_line1 = lines[i].split("PHYS")[1].strip().split(",")
            vars_line2 = lines[i + 1].strip().split(",")
            for _vars in vars_line1:
                if "EPS" in _vars:
                    EPS = float(_vars.split("=")[1])
                    # print(f'EPS={EPS}')
            for _vars in vars_line2:
                if "XIAB" in _vars:
                    XIAB = float(_vars.split("=")[1])
                    # print(f'XIAB={XIAB}')

            """ Looking for
            MAJOR RADIUS    :    2.9031 [m]
            MAGNETIC FIELD  :    1.9891 [T]
            TOTAL CURRENT   :   1982.873 [kA]
            RADIUS (a)      :    0.9277 [m]
            PSI ON BOUNDARY :    0.7022 [Wb/rad]
            CENTRAL DENSITY :    1.0000 10^19 [m^-3]
            ZEFF            :    1.1372
            TE/(TE+TI)      :    0.5000
            """
            for i, line in enumerate(lines):

                if "MAJOR RADIUS" in line:
                    RVAC = float(line.split(":")[1].split("[")[0].strip())
                    # print(f'RVAC={RVAC}')
                if "MAGNETIC FIELD" in line:
                    BVAC = float(line.split(":")[1].split("[")[0].strip())
                    # print(f'BVAC={BVAC}')
                if "PSI ON BOUNDARY" in line:
                    CPSURF = float(line.split(":")[1].split("[")[0].strip())
                    # print(f'CPSURF={CPSURF}')
                if "POL. FLUX" in line:
                    # ALFA     = RADIUS**2 * BVAC / CPSURF
                    ALFA = float(line.split(":")[1].split("[")[0].strip())
                    # print(f'ALFA={ALFA}')
                if "MAGNETIC AXIS" in line:
                    data = line.split(":")[1]
                    XAXIS, YAXIS = [float(x) for x in data.split()]

                if "POLOIDAL BETA" in line:
                    BETAP = float(line.split(":")[1].split("[")[0].strip())
                    # print(f'BETAP={BETAP}')
                if "TOROIDAL BETA" in line:
                    BETAT = float(line.split(":")[1].split("[")[0].strip())
                    # print(f'BETAT={BETAT}')
                if "NORM. BETA" in line:
                    BETAN = float(line.split(":")[1].split("[")[0].strip())
                    # print(f'BETAN={BETAN}')

        RMAGAXIS = RVAC * (1 + EPS * XAXIS)
        RADIUS = RVAC * EPS  # Minor radius
        CURRENT = (XIAB / mu_0) * (RADIUS * BVAC)

        # print(f'RMAGAXIS={RMAGAXIS}')
        # print(f'RADIUS={RADIUS}')

        output_vars["BVAC"] = BVAC
        output_vars["RVAC"] = RVAC
        output_vars["EPS"] = EPS
        output_vars["XAXIS"] = XAXIS
        output_vars["CPSURF"] = CPSURF
        output_vars["ALFA"] = ALFA
        output_vars["BETAP"] = BETAP
        output_vars["BETAT"] = BETAT
        output_vars["BETAN"] = BETAN
        output_vars["XIAB"] = XIAB
        output_vars["RMAGAXIS"] = RMAGAXIS  # NOTE: magnetic axis, different than RVAC
        output_vars["RADIUS"] = RADIUS  # NOTE: this is small a in TGLF inputs
        output_vars["CURRENT"] = CURRENT
        return output_vars

    def hel2eqdsk(
        self, helena_output_dir: str, eqdsk_output_path: str, NR: int, NZ: int
    ) -> None:
        fpath_elite = os.path.join(helena_output_dir, "eliteinp")
        fpath_f12 = os.path.join(helena_output_dir, "fort.12")
        fpath_f20 = os.path.join(helena_output_dir, "fort.20")

        elite_data = self.read_eliteinput(fpath_elite)
        geometry_vars = self.get_real_world_geometry_factors_from_f20(fpath_f20)
        helena_fort12_outputs = self.read_helena_fort12(
            fpath_f12,
            B0=geometry_vars["BVAC"],
            RVAC=geometry_vars["RVAC"],
            CPSURF=geometry_vars["CPSURF"],
            RADIUS=geometry_vars["RADIUS"],
        )

        psi_arr_og = elite_data["Psi"]
        fpol_arr_og = elite_data["fpol"]
        pres_arr_og = helena_fort12_outputs["P0_SCALED"]
        ffprim_arr_og = elite_data["ffp"]
        qpsi_arr_og = elite_data["q"]
        r_arr_og = elite_data["R"]
        z_arr_og = elite_data["z"]

        RDIM = r_arr_og.max() - r_arr_og.min()
        ZDIM = z_arr_og.max() - z_arr_og.min()
        ZMID = 0.0  # ZDIM / 2.0
        RLEFT = r_arr_og.min()

        SIMAG = psi_arr_og.min()
        SIBRY = psi_arr_og.max()
        RMAXIS = geometry_vars["RMAGAXIS"]
        RCENTR = geometry_vars["RVAC"]
        ZMAXIS = 0.0  # NOTE: Assumption
        BCENTR = geometry_vars["BVAC"]
        CURRENT = geometry_vars["CURRENT"]

        RBBBS = r_arr_og[:, -1]
        ZBBBS = z_arr_og[:, -1]
        NBBBS = RBBBS.shape[0]
        limitr = NBBBS
        RLIM = np.zeros_like(RBBBS)
        ZLIM = np.zeros_like(ZBBBS)

        """ Remap to uniform psin grid """
        psi_arr = np.linspace(SIMAG, SIBRY, NR)

        def remap_func_to_new_domain(
            x_old, y_old, x_new
        ) -> tuple[np.ndarray, np.ndarray]:
            spl_f = scipy.interpolate.CubicSpline(x_old, y_old)
            return spl_f(x_new), spl_f(x_new, 1)

        fpol_new, _ = remap_func_to_new_domain(psi_arr_og, fpol_arr_og, psi_arr)
        pres_new, pprime_new = remap_func_to_new_domain(
            psi_arr_og, pres_arr_og, psi_arr
        )
        ffpr_new, _ = remap_func_to_new_domain(psi_arr_og, ffprim_arr_og, psi_arr)
        qpsi_new, _ = remap_func_to_new_domain(psi_arr_og, qpsi_arr_og, psi_arr)

        """ Remap psi to R, Z on uniform R, Z grid"""

        r_1d = np.linspace(RLEFT, RLEFT + RDIM, NR)
        z_1d = np.linspace(ZMID - ZDIM / 2.0, ZMID + ZDIM / 2.0, NZ)

        R_new, Z_new = np.meshgrid(r_1d, z_1d)
        psi_old_2d = np.repeat(psi_arr_og[:, np.newaxis], z_arr_og.shape[0], axis=1).T
        fill_value = (
            psi_old_2d.max() + 0.2
        )  # NOTE The plus value is to ensure that the boundary contour is found,
        # or that there is a "boundary" value.

        grid_lin = scipy.interpolate.griddata(
            np.array([r_arr_og.flatten(), z_arr_og.flatten()]).T,
            psi_old_2d.flatten(),
            (R_new, Z_new),
            method="linear",
            fill_value=fill_value,
        )
        grid_lin = grid_lin.T

        format_2000 = ff.FortranRecordWriter("(6a8, 3i4)")
        format_2020 = ff.FortranRecordWriter("(5e16.9)")
        format_2022 = ff.FortranRecordWriter("2i5")

        XDUM = -1.0
        idnum = 3
        case_strings = ["HELENA", "PRODUCED", "EQDSK", "ARBT", "FILE", idnum, NR, NZ]

        header_str = format_2000.write(case_strings)

        with open(eqdsk_output_path, "w") as file:
            file.write(header_str + "\n")
            # (2020) rdim,zdim,rcentr,rleft,zmid
            file.write(format_2020.write([RDIM, ZDIM, RCENTR, RLEFT, ZMID]) + "\n")
            # (2020) rmaxis,zmaxis,simag,sibry,bcentr
            file.write(format_2020.write([RMAXIS, ZMAXIS, SIMAG, SIBRY, BCENTR]) + "\n")
            # (2020) current,simag,xdum,rmaxis,xdum
            file.write(format_2020.write([CURRENT, SIMAG, XDUM, RMAXIS, XDUM]) + "\n")
            # (2020) zmaxis,xdum,sibry,xdum,xdum
            file.write(format_2020.write([ZMAXIS, XDUM, SIBRY, XDUM, XDUM]) + "\n")
            # (2020) (fpol(i),i=1,nw)
            file.write(format_2020.write(fpol_new) + "\n")
            # (2020) (pres(i),i=1,nw)
            file.write(format_2020.write(pres_new) + "\n")
            # (2020) (ffprim(i),i=1,nw)
            file.write(format_2020.write(ffpr_new) + "\n")
            # (2020) (pprime(i),i=1,nw)
            file.write(format_2020.write(pprime_new) + "\n")
            # (2020) ((psirz(i,j),i=1,nw),j=1,nh)
            towrite = np.array(grid_lin, dtype=float).flatten(order="F")
            file.write(format_2020.write(towrite) + "\n")
            # (2020) (qpsi(i),i=1,nw)
            file.write(format_2020.write(qpsi_new) + "\n")
            # (2022) nbbbs,limitr
            file.write(format_2022.write([NBBBS, limitr]) + "\n")
            # (2020) (rbbbs(i),zbbbs(i),i=1,nbbbs)
            # See: https://github.com/Fusion-Power-Plant-Framework/eqdsk/blob/main/eqdsk/file.py#L733
            towrite = np.array([RBBBS, ZBBBS]).flatten(order="F")
            file.write(format_2020.write(towrite) + "\n")
            # (2020) (rlim(i),zlim(i),i=1,limitr)
            towrite = np.array([RLIM, ZLIM]).flatten(order="F")
            file.write(format_2020.write(towrite) + "\n")

    def read_eliteinput(self, filepath) -> dict[str, np.ndarray]:
        """
        Reads the following parameters:
        1D: Psi, dp/dpsi, d2p/dpsi, fpol, ffp, dffp, q, ne, dne/dpsi, Te, dTe/dpsi, Ti, dTi/dpsi,
            nMainIon, nZ
        2D: R, z, Bp
        Calculates PsiN from Psi
        returns dictionary with above keys
        """
        with open(filepath, "r") as file:
            file.readline()
            N, M = file.readline().split()
        N, M = int(N), int(M)

        keywords_Nshape = [
            "Psi",
            "dp/dpsi",
            "d2p/dpsi",
            "fpol",
            "ffp",
            "dffp",
            "q",
            "ne",
            "dne/dpsi",
            "Te",
            "dTe/dpsi",
            "Ti",
            "dTi/dpsi",
            "nMainIon",
            "nZ",
        ]
        keywords_Mshape = ["R", "z", "Bp"]

        data_N = self.read_fortran_ascii(filepath, keywords_Nshape, N)
        data_M = self.read_fortran_repeated_arrays(filepath, keywords_Mshape, N, M)

        data_N["PsiN"] = (data_N["Psi"] - data_N["Psi"].min()) / (
            data_N["Psi"].max() - data_N["Psi"].min()
        )
        return {**data_M, **data_N}

    def read_fortran_ascii(self, file_path, keywords, N) -> dict[str, np.ndarray]:
        """
        Reads data from a Fortran ASCII file and extracts arrays following specified keywords.

        Parameters:
            file_path (str): Path to the ASCII file.
            keywords (list of str): List of strings to search for in the file.
            N (int): Number of floats to extract for each keyword.

        Returns:
            dict: A dictionary where keys are keywords and values are 1D NumPy arrays of floats.
        """
        extracted_data = {key: None for key in keywords}

        with open(file_path, "r") as file:
            lines = file.readlines()

        for keyword in keywords:
            # Find the keyword in the lines
            for i, line in enumerate(lines):
                if line.strip().startswith(keyword):
                    # Collect the floats following the keyword
                    floats = []
                    j = i + 1  # Start reading the lines after the keyword
                    while j < len(lines) and len(floats) < N:
                        floats.extend(map(float, lines[j].split()))
                        j += 1
                    # Truncate or reshape if more/less than expected N values are found
                    extracted_data[keyword] = np.array(floats[:N])
                    break  # Stop looking for this keyword

        return extracted_data

    def read_helena_fort12(
        self,
        fort_12_file,
        B0=1.0,
        RVAC=2.9043,
        CPSURF=0.7022,
        RADIUS=0.9269,
    ):
        """
        Reads the mapping file for MISHKA produced by HELENA.

        - B0 is the on axis toroidal field strength (B_m below)
        - Rvac is vaccum geometric radius [Rvac below]
        - CPSURF is the poloidal flux on the surface of the plasma
        - RADIUS is the minor radius of the plasma

        NOTE: the P0_SCALED matches that of fort.20 real world output

        From toon's PB3D
        !! The variales in the HELENA mapping file are globalized in two ways:
        !!  - X and  Y are normalized w.r.t. vacuum geometric  axis \c R_vac and
        !!  toroidal field at the geometric axis \c B_vac.
        !!  - <tt> R[m] = R_vac[m] (1 + eps X[]) </tt>,
        !!  - <tt> Z[m] = R_vac[m] eps Y[] </tt>.
        !!
        !! The covariant  toroidal field  \c F_H,  \c pres_H  and poloidal  flux are
        !! normalized  w.r.t  magnetic axis  \c  R_m  and  total toroidal  field  at
        !! magnetic axis \c B_m:
        !!  - <tt> RBphi[Tm]     = F_H[] R_m[m] B_m[T] </tt>,
        !!  - <tt> pres[N/m^2]   = pres_H[] (B_m[T])^2/mu_0[N/A^2] </tt>,
        !!  - <tt> flux_p[Tm^2]  = 2pi (s[])^2 cpsurf[] B_m[T] (R_m[m])^2 </tt>.

        The result of the mapping is a dictionary with the following
        keys:
        - CS: 1D array of floats, the flux coordinate
        - QS: 1D array of floats, the safety factor
        - DQS: 1D array of floats, the safety factor gradient
        - DQEC: float, the safety factor gradient at the edge
        - CURJ: 1D array of floats, the current density
        - DJ0: float, the current density at the edge
        - DJE: float, the current density at the edge
        - CHI: 1D array of floats, the chi values
        - GEM11: 2D array of floats, the mapping matrix
        - GEM12: 2D array of floats, the mapping matrix
        - GEM33: 2D array of floats, the mapping matrix
        - CPSURF: float, the poloidal flux on the surface of the plasma
        - RADIUS: float, the minor radius
        - RAXIS: float, the major radius of the plasma
        - P0: 1D array of floats, the pressure
        - DP0: float, the pressure gradient
        - DPE: float, the pressure gradient at the edge
        - RBPHI: 1D array of floats, the toroidal field flux function (F)
        - DRBPHI0: float, the toroidal field flux function gradient at the axis
        - DRBPHIE: float, the toroidal field flux function gradient at the edge
        - VX: 1D array of floats, the r on the surface
        - VY: 1D array of floats, the z on the surface
        - EPS: float, the inverse aspect ratio
        - XOUT: 2D array of floats, the r mapping
        - YOUT: 2D array of floats, the z mapping
        - XOUT_SCALED: 2D array of floats, the scaled r mapping
        - YOUT_SCALED: 2D array of floats, the scaled z mapping
        - CS_SCALED: 1D array of floats, the scaled flux coordinate
        - P0_SCALED: 1D array of floats, the scaled pressure
            -> NB: If the file contains B0, then the pressure scaled with that instead of
                    what is passed above

        """

        def remap_2d_f12_output(_data, _NCHI, _JS0) -> np.ndarray:
            OUT = np.zeros((_NCHI, _JS0 + 1))
            for _id in range(_NCHI + 1, (_JS0 + 1) * _NCHI + 1):
                row = (_id - 1) % _NCHI
                col = (_id - 1) // _NCHI
                OUT[row, col] = _data[_id - (_NCHI + 1)]
            return OUT

        helena_fort12_outputs = {}
        with open(fort_12_file, "r") as file:
            lines = file.readlines()
            JS0 = int(lines[0])
            # JS0 += 1

            curr_indx = 1
            # Read CS
            floats = []
            for i, line in enumerate(lines[curr_indx:]):
                if len(floats) == JS0 + 1:
                    break
                floats.extend([float(i) for i in line.split()])
            helena_fort12_outputs["CS"] = np.array(floats) ** 2
            curr_indx += i
            # Read QS
            floats = []
            for i, line in enumerate(lines[curr_indx:]):
                if len(floats) == JS0 + 1:
                    break
                floats.extend([float(i) for i in line.split()])
            helena_fort12_outputs["QS"] = np.array(floats)
            curr_indx += i

            # Read DQS, DQEC
            DQS, DQEC = [float(i) for i in lines[curr_indx].split()]
            curr_indx += 1
            # Read DQS
            floats = []
            for i, line in enumerate(lines[curr_indx:]):
                if len(floats) == JS0:
                    break
                floats.extend([float(i) for i in line.split()])
            floats[0] = DQS
            helena_fort12_outputs["DQS"] = np.array(floats)
            curr_indx += i

            # Read CURJ (CURJ(JS), JS=1, JS0+1)
            floats = []
            for i, line in enumerate(lines[curr_indx:]):
                if len(floats) == JS0 + 1:
                    break
                floats.extend([float(i) for i in line.split()])
            helena_fort12_outputs["CURJ"] = np.array(floats)
            curr_indx += i
            # Read DJ0, DJE
            DJ0, DJE = [float(i) for i in lines[curr_indx].split()]
            curr_indx += 1
            # Read NCHI
            NCHI = int(lines[curr_indx])
            curr_indx += 1
            # Read CHI, (CHI(JS), JS=1, NCHI)
            floats = []
            for i, line in enumerate(lines[curr_indx:]):
                if len(floats) == NCHI:
                    break
                floats.extend([float(i) for i in line.split()])
            helena_fort12_outputs["CHI"] = np.array(floats)
            curr_indx += i
            # Read GEM11 (GEM11(JS), JS=NCHI+1, (JS0+1)*NCHI)
            floats = []
            for i, line in enumerate(lines[curr_indx:]):
                if len(floats) == (JS0 + 1) * NCHI - NCHI:
                    break
                floats.extend([float(i) for i in line.split()])

            helena_fort12_outputs["GEM11"] = remap_2d_f12_output(
                np.array(floats), NCHI, JS0
            )
            helena_fort12_outputs["GEM11"][:, 0] = 1e-8
            curr_indx += i
            # should be of length (JS0+1)*NCHI - NCHI
            # Read GEM12 (GEM12(JS), JS=NCHI+1, (JS0+1)*NCHI)
            floats = []
            for i, line in enumerate(lines[curr_indx:]):
                if len(floats) == (JS0 + 1) * NCHI - NCHI:
                    break
                floats.extend([float(i) for i in line.split()])
            helena_fort12_outputs["GEM12"] = remap_2d_f12_output(
                np.array(floats), NCHI, JS0
            )

            curr_indx += i
            # Read CPSURF, RADIUS, poloidal flux on the surface of the plasma and minor radius
            _CPSURF, _RADIUS = [float(i) for i in lines[curr_indx].split()]
            # helena_fort12_outputs['CPSURF'] = CPSURF
            # helena_fort12_outputs['RADIUS'] = RADIUS
            curr_indx += 1
            # Read GEM33 (GEM33(JS), JS=NCHI+1, (JS0+1)*NCHI)
            floats = []
            for i, line in enumerate(lines[curr_indx:]):
                if len(floats) == (JS0 + 1) * NCHI - NCHI:
                    break
                floats.extend([float(i) for i in line.split()])
            helena_fort12_outputs["GEM33"] = remap_2d_f12_output(
                np.array(floats), NCHI, JS0
            )
            helena_fort12_outputs["GEM33"][:, 0] = RADIUS**2
            helena_fort12_outputs["GEM33"] = 1.0 / helena_fort12_outputs["GEM33"]
            # helena_fort12_outputs['GEM33'] = np.array(floats).reshape(NCHI, JS0)
            curr_indx += i

            # Read RAXIS, Major radius of the plasma
            if len(lines[curr_indx].split()) == 2:
                # Some versions of HELENA output RAXIS and B0
                RAXIS, _B0 = [float(i) for i in lines[curr_indx].split()]
            else:
                RAXIS = float(lines[curr_indx])
            curr_indx += 1
            helena_fort12_outputs["RAXIS"] = RAXIS
            helena_fort12_outputs["B0"] = B0
            helena_fort12_outputs["RVAC"] = RVAC

            # Read P0 (P0(JS), JS=1, JS0+1)
            floats = []
            for i, line in enumerate(lines[curr_indx:]):
                if len(floats) == JS0 + 1:
                    break
                floats.extend([float(i) for i in line.split()])
            helena_fort12_outputs["P0"] = np.array(floats)
            curr_indx += i
            # Read DP0, DPE
            DP0, DPE = [float(i) for i in lines[curr_indx].split()]
            curr_indx += 1
            # Read RBPHI (RBPHI(JS), JS=1, JS0+1)
            floats = []
            for i, line in enumerate(lines[curr_indx:]):
                if len(floats) == JS0 + 1:
                    break
                floats.extend([float(i) for i in line.split()])
            helena_fort12_outputs["RBPHI"] = np.array(floats)
            curr_indx += i
            # Read DRBPHI0, DRBPHIE, (derivatives of RBPHI on axis and surface)
            DRBPHI0, DRBPHIE = [float(i) for i in lines[curr_indx].split()]
            curr_indx += 1

            # Read VX (VX(JS), JS=1, NCHI) (r on surface)
            floats = []
            for i, line in enumerate(lines[curr_indx:]):
                if len(floats) == NCHI:
                    break
                floats.extend([float(i) for i in line.split()])
            helena_fort12_outputs["VX"] = np.array(floats)
            curr_indx += i
            # Read VY (VY(JS), JS=1, NCHI) (z on surface)
            floats = []
            for i, line in enumerate(lines[curr_indx:]):
                if len(floats) == NCHI:
                    break
                floats.extend([float(i) for i in line.split()])
            helena_fort12_outputs["VY"] = np.array(floats)
            curr_indx += i
            # Read EPS (epsilon, inverse aspect ratio)
            EPS = float(lines[curr_indx])
            helena_fort12_outputs["EPS"] = EPS
            curr_indx += 1
            # Read XOUT (XOUT(JS), JS=NCHI+1, (JS0+1)*NCHI) (r)
            floats = []
            for i, line in enumerate(lines[curr_indx:]):
                if len(floats) == (JS0 + 1) * (NCHI) - (NCHI):
                    break
                floats.extend([float(i) for i in line.split()])

            data = np.array(floats)
            XOUT = np.zeros((NCHI, JS0 + 1))
            for _id in range(NCHI + 1, (JS0 + 1) * NCHI + 1):
                row = (_id - 1) % NCHI
                col = (_id - 1) // NCHI
                XOUT[row, col] = data[_id - (NCHI + 1)]
            helena_fort12_outputs["XOUT"] = XOUT

            # helena_fort12_outputs['XOUT'] = np.array(floats).reshape(NCHI, JS0)
            curr_indx += i

            # Read YOUT (YOUT(JS), JS=NCHI+1, (JS0+1)*NCHI) (z)
            floats = []
            for i, line in enumerate(lines[curr_indx:]):
                if len(floats) == (JS0 + 1) * NCHI - NCHI:
                    break
                floats.extend([float(i) for i in line.split()])

            data = np.array(floats)
            YOUT = np.zeros((NCHI, JS0 + 1))
            for _id in range(NCHI + 1, (JS0 + 1) * NCHI + 1):
                row = (_id - 1) % NCHI
                col = (_id - 1) // NCHI
                YOUT[row, col] = data[_id - (NCHI + 1)]
            helena_fort12_outputs["YOUT"] = YOUT
            curr_indx += i

            # Scale XOUT, YOUT with RADIUS*RAXIS
            helena_fort12_outputs["XOUT_SCALED"] = (RVAC) * (
                1.0 + EPS * helena_fort12_outputs["XOUT"]
            )
            helena_fort12_outputs["YOUT_SCALED"] = (RVAC * EPS) * helena_fort12_outputs[
                "YOUT"
            ]
            helena_fort12_outputs["XOUT_SCALED"][:, 0] = RAXIS
            helena_fort12_outputs["YOUT_SCALED"][:, 0] = 0.0

            # SCALE Flux coordinate flux_p[Tm^2 / rad]  = (s[])^2 cpsurf[] B_m[T] (R_m[m])^2
            helena_fort12_outputs["CS_SCALED"] = (
                (helena_fort12_outputs["CS"] ** 2) * _CPSURF * B0 * (RVAC**2)
            )
            helena_fort12_outputs["CS_SCALED"] = (
                helena_fort12_outputs["CS"] ** 2
            ) * CPSURF  # * B0 * (RVAC**2)
            # NOTE: I am not sure the above is correct or wrong, below gives better results?

            # SCALE pressure, p*(B_m[T])^2/mu_0[N/A^2] </tt>,
            helena_fort12_outputs["P0_SCALED"] = (helena_fort12_outputs["P0"]) * (
                B0**2 / (mu_0)
            )

        return helena_fort12_outputs

    def read_fortran_repeated_arrays(
        self, file_path, keywords, N, M
    ) -> dict[str, np.ndarray]:
        """
        Reads data from a Fortran ASCII file and extracts repeated arrays of length N following
        a keyword.

        Parameters:
            file_path (str): Path to the ASCII file.
            keyword (str): String to search for in the file to start reading data.
            N (int): Number of floats in each array.
            M (int): Number of arrays of length N to extract.

        Returns:
            np.ndarray: A 2D NumPy array of shape (M, N) containing the extracted data.
        """
        with open(file_path, "r") as file:
            lines = file.readlines()

        extracted_data = {key: None for key in keywords}

        for keyword in keywords:
            data = []
            reading = False  # Flag to start reading after the keyword
            floats_collected = 0  # Track the number of floats collected
            for i, line in enumerate(lines):
                if not reading:
                    if line.strip().startswith(keyword):  # Find the keyword
                        reading = True
                        continue
                if reading:
                    # Extract floats from the line
                    floats = list(map(float, line.split()))
                    data.extend(floats)
                    floats_collected += len(floats)
                    # Stop reading if we've collected M * N floats
                    if floats_collected >= M * N:
                        break

            # Reshape the collected data into a 2D array of shape (M, N)
            data = np.array(data[: M * N]).reshape(M, N)
            extracted_data[keyword] = data
        return extracted_data

    def write_iterbd_profiles(
        self, helena_output_dir: str, iterdb_output_path: str
    ) -> None:
        fpath_elite = os.path.join(helena_output_dir, "eliteinp")
        elite_data = self.read_eliteinput(fpath_elite)
        f_out = open(iterdb_output_path, "w")
        format_iterdb = ff.FortranRecordWriter("(6e13.6)")

        def write_header(label, unit, nx, ny=1):
            f_out.write(
                "  00000DUM "
                + str(ny + 1)
                + " 0 6              ;-SHOT #- F(X,Y) DATA -UFILELIB- 00Jan0000\n"
            )
            f_out.write(
                "                               ;-SHOT DATE- UFILES ASCII FILE SYSTEM\n"
            )
            f_out.write(
                "   0                           ;-NUMBER of ASSOCIATED SCALAR QUANTITIES-\n"
            )
            f_out.write(
                " RHOTOR              -         ;-INDEPENDENT VARIABLE LABEL: X-\n"
            )
            f_out.write(
                " TIME                SECONDS   ;-INDEPENDENT VARIABLE LABEL: Y-\n"
            )
            f_out.write(
                " "
                + str(label)
                + "               "
                + str(unit)
                + "     ;-DEPENDENT VARIABLE LABEL-\n"
            )
            f_out.write(
                " 0                             ;-PROC CODE- 0:RAW 1:AVG 2:SM.  3:AVG+SM\n"
            )
            f_out.write("        " + str(nx) + "                    ;-# OF X PTS-\n")
            f_out.write(
                "          "
                + str(ny)
                + "                    ;-# OF Y PTS-   X,Y,F(X,Y) DATA FOLLOW:\n"
            )

        def write_closure():
            f_out.write(";----END-OF-DATA-----------------------COMMENTS:----------\n")
            f_out.write(
                "*******************************************************************************\n"
            )
            f_out.write(
                "*******************************************************************************\n"
            )

        def write_quantity(vector):
            number = math.ceil(len(vector) / 6)
            for i in range(number):
                f_out.write(" " + format_iterdb.write(vector[6 * i : 6 * i + 6]) + "\n")
            f_out.write(" " + format_iterdb.write([0.0]) + "\n")

        psi = elite_data["Psi"]
        nx = len(psi)
        te = elite_data["Te"]
        ti = elite_data["Ti"]
        ne = elite_data["ne"]
        q = elite_data["q"]
        q_cd = 0.5 * (q[:-1] + q[1:])
        dpsi = np.diff(psi)
        dphit = q_cd * dpsi
        phit = np.cumsum(dphit)
        phitt = np.concatenate(([0], phit))
        rhot = np.sqrt(phitt / np.max(phitt))
        write_header("TE   ", "eV   ", nx)
        write_quantity(rhot)
        write_quantity(te)
        write_closure()
        write_header("TI   ", "eV   ", nx)
        write_quantity(rhot)
        write_quantity(ti)
        write_closure()
        write_header("NE   ", "m^-3 ", nx)
        write_quantity(rhot)
        write_quantity(ne)
        write_closure()
        write_header("NM1  ", "m^-3 ", nx)
        write_quantity(rhot)
        write_quantity(ne)
        write_closure()
        write_header("VROT ", "rad/s", nx)
        write_quantity(rhot)
        write_quantity(ne * 0)
        write_closure()
        write_header("ZEFFR", "     ", nx)
        write_quantity(rhot)
        write_quantity(np.ones(len(ne)))
        write_closure()
        f_out.close()
