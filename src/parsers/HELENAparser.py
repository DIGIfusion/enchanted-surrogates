import os
import math
import f90nml
import numpy as np
import json
from .base import Parser
try:
    from dask.distributed import print
except:
    pass
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
        Generates a summary file with run directory and parameters, along with success and
        stability criteria.

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
        print(f"Writing to {run_dir}. Params: {params}")
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

        helena_phys_params = [
            "eps", "b", "betap", "alfa", "xiab", "q95", "rvac", "bvac", "zn0", "zeff",  # fmt: skip
            "idete", "iscalejz", "neosig", "xiab", "scalemix", "zjzsmooth",     # fmt: skip
            "bsmultip", "zimp", "corep", "bkeep", "bsmodel", "efitsig",         # fmt: skip
            "ifastsca", "safemode", "coreneo", "amain", "zmain", "idete",       # fmt: skip
            "ade", "bde", "cde", "dde", "ede", "fde", "gde", "hde", "ide",      # fmt: skip
            "ate", "bte", "cte", "ddte", "ete", "fte", "gte", "hte", "ite",     # fmt: skip
            "ati", "bti", "cti", "ddti", "eti", "fti", "gti", "hti", "iti"]     # fmt: skip
        for p in helena_phys_params:
            if p in params:
                namelist["phys"][p] = params[p]

        f90nml.write(namelist, input_fpath)
        print(f"fort.10 written to: {input_fpath}")
        return

    def write_input_file_europed(self, params: dict, run_dir: str, namelist_path: str):
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
        print(f"Writing to {run_dir}. Params (europed): {params}")
        if os.path.exists(run_dir):
            input_fpath = os.path.join(run_dir, "fort.10")
        else:
            raise FileNotFoundError(f"Couldnt find {run_dir}")

        namelist = f90nml.read(namelist_path)

        # Update namelist with input parameters
        if "bvac" in params:
            namelist["phys"]["bvac"] = params["bvac"]
        if "ip" in params and "bvac" in params:
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
        if "tria" in params:
            namelist["shape"]["tria"] = params["tria"]
        # Europed profiles
        if "pedestal_delta" in params:
            d_ped = params["pedestal_delta"]
            namelist["shape"]["xr2"] = 1 - d_ped / 2.0
            namelist["shape"]["sig2"] = d_ped

            # Update density profile parameters
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
            namelist["phys"]["ddte"] = psi_mid
            namelist["phys"]["ete"] = d_ped
            namelist["phys"]["fte"] = 1 - d_ped
            namelist["phys"]["gte"] = (
                1.2 if "alpha_T1" not in params else params["alpha_T1"]
            )
            namelist["phys"]["hte"] = (
                1.4 if "alpha_T2" not in params else params["alpha_T2"]
            )

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
        namelist["profile"]["zjz"] = self.make_init_zjz_profile(
            pedestal_delta=params["pedestal_delta"], npts=namelist["profile"]["npts"]
        )

        # teped_multip

        f90nml.write(namelist, input_fpath)
        print(f"fort.10 written to: {input_fpath}")
        return

    def write_input_file_scaling(self, params: dict, run_dir: str, namelist_path: str):
        """
        Writes input file fort.10.
        2: using a scaling law for changing ATE and CTE, requires input parameter "scaling_factor"

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
        print(f"Writing to {run_dir}. Params (scaling): {params}")
        if os.path.exists(run_dir):
            input_fpath = os.path.join(run_dir, "fort.10")
        else:
            raise FileNotFoundError(f"Couldnt find {run_dir}")

        namelist = f90nml.read(namelist_path)

        # Update namelist with input parameters
        sf = params["scaling_factor"]
        namelist["phys"]["ate"] = namelist["phys"]["ate"] * sf**(-6)
        namelist["phys"]["cte"] = namelist["phys"]["ate"] * sf**(8)

        f90nml.write(namelist, input_fpath)
        return

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
            file = open(os.path.join(run_dir, "fort.20"), "r")
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
            term2 = (1 - (psi / psi_ped) ** alpha_1) ** alpha_2
        else:
            term2 = 0.0

        return sep + a_0 * term1 + a_1 * term2

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
            if os.path.exists(os.path.join(run_dir, file_name)):
                os.remove(os.path.join(run_dir, file_name))
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
        summary = {"run_dir": run_dir, "params": params}
        success, mercier_stable, ballooning_stable = self.read_output_file(run_dir)
        summary["h_id"] = run_dir.split("/")[-1]
        summary["success"] = success
        summary["mercier_stable"] = mercier_stable
        summary["ballooning_stable"] = ballooning_stable
        with open(os.path.join(run_dir, "summary.json"), "w") as outfile:
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
        try:
            psi, S, J, vol, _, area = self.read_fort20_s_j_vol_area(run_dir)
            np.save(run_dir + "/psi.npy", psi)
            np.save(run_dir + "/s.npy", S)
            np.save(run_dir + "/j.npy", J)
            np.save(run_dir + "/volume.npy", vol)
            np.save(run_dir + "/area.npy", area)
        except Exception as e:
            print(f"Something went wrong when trying to read/write read_fort20_s_j_vol_area, {e}")
        
        try:
            realworld_S, realworld_P, realworld_Ne, realworld_Te, realworld_Ti = self.read_fort20_realworld(run_dir)
            np.save(run_dir + "/realworld_S.npy", realworld_S)
            np.save(run_dir + "/realworld_P.npy", realworld_P)
            np.save(run_dir + "/realworld_Ne.npy", realworld_Ne)
            np.save(run_dir + "/realworld_Te.npy", realworld_Te)
            np.save(run_dir + "/realworld_Ti.npy", realworld_Ti)
        except Exception as e:
            print(f"Something went wrong when trying to read/write read_fort20_realworld, {e}")
        
        return success

    def read_final_zjz(self, output_dir):
        """
        zjz(   1)=     1.00000,
        zjz(   2)=     1.01337,
        zjz(   3)=     0.83545,
        """
        file = open(os.path.join(output_dir, "final_zjz"), "r")
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

        JS0: NRMAP - 1
        CS: sqrt(psi)
        QS:
        DQS_1, DQEC
        DQS: derivative of QS
        CURJ
        DJ0, DJE
        NCHI: number of poloidal grid points written to output file
        CHI
        GEM11
        GEM12
        CPSURF, RADIUS
        GEM33
        RAXIS
        P0
        DP0, DPE
        RBPHI
        DRBPHI0, DRBPHIE
        VX, VY
        EPS
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

    def read_fort20_NUM(self, output_dir):
        """
        Find the numerical parameters

        Returns

        NR: number of radial grid points
        NP: number of poloidal grid points
        NRMAP: number of radial grid points in mapping to flux surface grid (should be 2*NR)
        NPMAP: number of poloidal grid points in mapping (should be 2*NP)
        NCHI: number of poloidal grid points written to output file
        (should be 2*NMAP, NCHI must be 2^n + 1)
        NRCUR: number of radial grid points for iteration current profile (should be NR)
        NPCUR: number of poloidal grid points (should be NP)
        NITER: (ICUR≠0) or 1 (ICUR=0) maximum number of iterations over current profile
        """
        filename = os.path.join(output_dir, "fort.20")
        file = open(filename).readlines()

        start_line = file.index(
            "*  HELENA : GRAD-SHAFRANOV EQUILIBRIUM  *\n"
        )

        for line in file[start_line:start_line+30]:
            index = line.find("NR =")
            if index != -1:
                NR = int(line[index + 4:index + 7])
            index = line.find("NP =")
            if index != -1:
                NP = int(line[index + 4:index + 7])
            index = line.find("NRMAP =")
            if index != -1:
                NRMAP = int(line[index + 7:index + 10])
            index = line.find("NPMAP =")
            if index != -1:
                NPMAP = int(line[index + 7:index + 10])
            index = line.find("NCHI =")
            if index != -1:
                NCHI = int(line[index + 6:index + 10])
            index = line.find("NITER =")
            if index != -1:
                NITER = int(line[index + 7:index + 11])

        return NR, NP, NRMAP, NPMAP, NCHI, NITER

    def read_fort20_beta_section(self, output_dir):
        """
         ***************************************
        MAGNETIC AXIS :   0.01908  0.00000
        POLOIDAL BETA :   0.1198E+00
        TOROIDAL BETA :   0.3802E-02
        BETA STAR     :   0.4250E-02
        NORM. BETA    :   0.00335
        TOTAL CURRENT :   0.1428E+01
        TOTAL AREA    :   0.5115E+01
        TOTAL VOLUME  :   0.3110E+02
        INT. INDUCTANCE :  0.685990E+00
        POL. FLUX     :   0.2130E+01
        A,B,C         :   0.4176E+01  0.1522E-01  0.1000E+01
        ***************************************
        """

        file = open(os.path.join(output_dir, "fort.20"), "r")
        while True:
            line = file.readline()
            if line.find("NORM. BETA") > -1:
                spl = line.split(":")
                betan = float(spl[1]) * 100
            if line.find("POLOIDAL BETA") > -1:
                spl = line.split(":")
                betap = float(spl[1])
            if line.find("TOTAL CURRENT") > -1:
                spl = line.split(":")
                total_current = float(spl[1])
            if line.find("TOTAL AREA") > -1:
                spl = line.split(":")
                total_area = float(spl[1])
            if line.find("TOTAL VOLUME") > -1:
                spl = line.split(":")
                total_volume = float(spl[1])
            if line.find("TOROIDAL BETA") > -1:
                spl = line.split(":")
                beta_tor = float(spl[1])
            if line.find("BETA STAR") > -1:
                spl = line.split(":")
                beta_star = float(spl[1])
            if line.find("PED. BETAPOL") > -1:
                spl = line.split(":")
                helenaBetap = float(spl[1])
            if line.find("A,B,C") > -1:
                spl = line.split(":")
                sp2 = spl[1].split()
                b_last_round = float(sp2[1])
            if line.find("RADIUS") > -1:
                spl = line.split(":")
                radius = float(spl[1])
            if line.find("B0") > -1:
                spl = line.split(":")
                B0 = float(spl[1])
                break
        file.close()
        return (
            betan, betap, total_current, total_area, total_volume, beta_tor,
            beta_star, helenaBetap, b_last_round, radius, B0
        )

    def read_fort20_inputprofiles(self, output_dir, NRMAP):
        """
        ***********************************************
        *        INPUT PROFILES :                     *
        ***********************************************
        *  PSI,       dP/dPSI,    FdF/dPSI,     J_phi *
        ***********************************************
        0.0000E+00  0.0000E+00  0.1000E+01  0.1000E+01
        0.1000E-01  0.1940E+01  0.1000E+01  0.9789E+00
        0.2000E-01  0.2193E+01  0.1000E+01  0.9573E+00
        ...

        """
        filename = os.path.join(output_dir, "fort.20")
        file = open(filename).readlines()
        i = 0
        i_table_start = -1
        with open(os.path.join(output_dir, "fort.20"), "r") as file:
            lines = file.readlines()
        for line in lines:
            if ("INPUT PROFILES" in line):
                i_table_start = i
                break
            i += 1
        if i_table_start == -1:
            print("Input profiles not found.")
            return

        columns = np.loadtxt(
            filename,
            dtype=str,
            comments=None,
            # delimiter=' ',
            converters=None,
            skiprows=i_table_start + 2,
            # usecols=None, unpack=False, ndmin=0, encoding='bytes',
            max_rows=1,
        )
        columns = columns[1:-1]
        columns = [c.replace(",", "") for c in columns]
        print(columns)

        inputprofiles = np.loadtxt(
            filename,
            # dtype=<class 'float'>,
            comments="*",
            # delimiter=' ',
            converters=None,
            skiprows=i_table_start + 3,
            # usecols=None, unpack=False, ndmin=0, encoding='bytes',
            max_rows=NRMAP,
        )

        i_col = 0
        for c in columns:
            if c == "PSI":
                psi = inputprofiles[:, i_col]
            elif c == "dP/dPSI":
                dP_dPSI = inputprofiles[:, i_col]
            elif c == "FdF/dPSI":
                FdF_dPSI = inputprofiles[:, i_col]
            elif c == "J_phi":
                J_phi = inputprofiles[:, i_col]
            i_col += 1

        return psi, dP_dPSI, FdF_dPSI, J_phi

    def read_fort20_s_j_vol_area(self, output_dir):
        """
        
        *********************************************************************
        * I   PSI     S      <J>     ERROR   LENGTH    BUSSAC   VOL    VOLP   AREA *
        *********************************************************************
        300  0.000159  0.012600    1.0029  3.95E-03    0.0809    0.0035    0.0032   19.9232    0.0005
        299  0.000635  0.025200    1.0115  1.58E-02    0.1614    0.0137    0.0126   19.7961    0.0020

        """
        NRMAP = self.get_namelist(output_dir=output_dir)["NUM"]["NRMAP"]
        npts = NRMAP - 1
        i = 0
        i_table_start = -1
        with open(os.path.join(output_dir, "fort.20"), "r") as file:
            lines = file.readlines()
        for line in lines:
            if (" LENGTH    BUSSAC   VOL    VOLP   AREA" in line ):
                i_table_start = i
                break
            i += 1
        numerical_lines = lines[i_table_start + 2:i_table_start + 1 + npts]

        # Convert the data to numpy arrays
        data_array = np.array(
            [list(map(float, line.split())) for line in numerical_lines]
        )

        # Split into columns
        psi = data_array[:, 1]
        S = data_array[:, 2]
        J = data_array[:, 3]
        vol = data_array[:, 7]
        volp = data_array[:, 8]
        area = data_array[:, 9]
        return psi, S, J, vol, volp, area

    def get_namelist(self, output_dir):
        namelist_path = os.path.join(output_dir, "fort.10")
        return f90nml.read(namelist_path)

    def read_fort20_realworld(self, output_dir):
        """
        **************************************************
            S,   P [Pa], Ne [10^19m^-3], Te [eV],  Ti [eV],
        **************************************************
        """
        NRMAP = self.get_namelist(output_dir=output_dir)["NUM"]["NRMAP"]
        npts = NRMAP - 1
        i = 0
        i_table_start = -1
        with open(os.path.join(output_dir, "fort.20"), "r") as file:
            lines = file.readlines()
        for line in lines:
            if "S,   P [Pa], Ne [10^19m^-3], Te [eV],  Ti [eV]" in line:
                i_table_start = i
                break
            i += 1
        numerical_lines = lines[i_table_start + 2:i_table_start + 1 + npts]

        # Convert the data to numpy arrays
        data_array = np.array(
            [list(map(float, line.split())) for line in numerical_lines]
        )

        # Split into columns
        S = data_array[:, 0]
        P = data_array[:, 1]
        Ne = data_array[:, 2]
        Te = data_array[:, 3]
        Ti = data_array[:, 4]
        return S, P, Ne, Te, Ti

    def get_te_params(self, output_dir: str):
        namelist_path = os.path.join(output_dir, "fort.10")
        namelist = f90nml.read(namelist_path)
        return {
            "ate": namelist["phys"]["ate"],
            "bte": namelist["phys"]["bte"],
            "cte": namelist["phys"]["cte"],
            "ddte": namelist["phys"]["ddte"],
            "ete": namelist["phys"]["ete"],
            "fte": namelist["phys"]["fte"],
            "gte": namelist["phys"]["gte"],
            "hte": namelist["phys"]["hte"],
            # "ite": namelist["phys"]["ite"],
        }

    def get_ne_params(self, output_dir: str):
        namelist_path = os.path.join(output_dir, "fort.10")
        namelist = f90nml.read(namelist_path)
        return {
            "ade": namelist["phys"]["ade"],
            "bde": namelist["phys"]["bde"],
            "cde": namelist["phys"]["cde"],
            "dde": namelist["phys"]["dde"],
            "ede": namelist["phys"]["ede"],
            "fde": namelist["phys"]["fde"],
            "gde": namelist["phys"]["gde"],
            "hde": namelist["phys"]["hde"],
            # "ide": namelist["phys"]["ite"],
        }
    
