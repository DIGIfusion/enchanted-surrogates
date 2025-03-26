from .base import Parser

# import subprocess
import os
import f90nml
import numpy as np
import math
import json


class CASTORparser(Parser):
    """An I/O parser for CASTOR

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

    def __init__(self, namelist_path: str):
        self.namelist_path = namelist_path

    def write_input_file(self, params: dict, run_dir: str):
        print("parser params", params)
        print("Writing to", run_dir)

        if os.path.exists(run_dir):
            input_fpath = os.path.join(run_dir, "fort.10")
        else:
            raise FileNotFoundError(f"Couldnt find {run_dir}")

        # Update toroidal mode number
        namelist = f90nml.read(self.namelist_path)
        namelist["newrun"][0]["ntor"] = -int(params["ntor"])

        f90nml.write(namelist, input_fpath)
        print(input_fpath)

    def read_output_fort20(self, run_dir: str):
        """
        The main output file.
        Looking for line:
         INSTABILITY FOUND :   -10    9  0.3076E+00  0.2167E-08

        If number of iterations reached the maximum limit (usually 20),
        then the growthrate is assumed to be 0. (Europed)
        """
        success = False
        growthrate = (None, None)
        filename = run_dir + "/fort.20"
        try:
            file = open(filename, "r")
            lines = file.readlines()
            for line in lines:
                if "INSTABILITY" in line:
                    print(line)
                    spl = line.split()
                    iteration = int(spl[4])
                    growthrate = (
                        (0.0, 0.0)
                        if iteration >= 20
                        else (float(spl[5]), float(spl[6]))
                    )
                    success = True
        except FileNotFoundError as e:
            print(f"(read_output_fort20) FileNotFoundError: {e}")
        return success, growthrate

    def read_output_fort22(self, run_dir: str):
        """
        Read the output file fort.22 which contains a list of
        eigenvalues (real, complex). The eigenvalues are listed
        as [Re(ev1), Re(ev1)+Im(ev1), Re(ev2), Re(ev2)+Im(ev2), ...]

        gamma: growth rate eigenvalue
        ng: number of grid points
        ngl: 2
        nbg: the number of eigenvalues per harmonic (2 * ngl * manz)
        manz: number of poloidal harmonics
        sgrid: flux system coordinate s = sqrt(psi/psi_boundary)
        rfour: "RFOUR(1) - LOWEST POLOIDAL MODE NUMBER" ?

        ev: array of eigenvalues
        [complex(Re(ev1), Im(ev1)), complex(Re(ev2), Im(ev2)), ...]

        """

        filename = run_dir + "/fort.22"
        file = open(filename, "r")
        lines = file.readlines()
        ew, ewc = lines[0].split()
        gamma = complex(ew, ewc)
        ng, manz, ngl = lines[1].split()
        ng, manz, ngl = int(ng), int(manz), int(ngl)
        nbg = 2 * ngl * manz
        print(f"manz={manz}, ng={ng}, ngl={ngl}, nbg={nbg}")

        rfour, endline = self.read_multiline_list(lines, startline=2, n_listitems=manz)
        sgrid, endline = self.read_multiline_list(
            lines, startline=endline, n_listitems=ng
        )
        ev_orig, endline = self.read_multiline_list(
            lines, startline=endline, n_listitems=int(ng * nbg * 2)
        )

        ev = np.array(
            [
                complex(ev_orig[i], ev_orig[i + 1] - ev_orig[i])
                for i in range(0, len(ev_orig), 2)
            ]
        )

        return (gamma, ng, manz, ngl, nbg, rfour, sgrid, ev)

    def read_input_fort12(self, run_dir: str):
        """
        Read the HELENA output file fort.12 which is used as input by MISHKA.
        """
        filename = run_dir + "/fort.12"
        lines = open(filename, "r").readlines()
        JS0 = int(lines[0].split()[0])
        CS, endline = self.read_multiline_list(lines, startline=1, n_listitems=JS0 + 1)
        QS, endline = self.read_multiline_list(
            lines, startline=endline, n_listitems=JS0 + 1
        )
        DQS_1, self.DQEC = float(lines[endline].split()[0]), float(
            lines[endline].split()[1]
        )
        DQS, endline = self.read_multiline_list(
            lines, startline=endline + 1, n_listitems=JS0
        )
        CURJ, endline = self.read_multiline_list(
            lines, startline=endline, n_listitems=JS0 + 1
        )
        DJ0, self.DJE = float(lines[endline].split()[0]), float(
            lines[endline].split()[1]
        )
        NCHI = int(lines[endline + 1].split()[0])
        CHI, endline = self.read_multiline_list(
            lines, startline=endline + 2, n_listitems=NCHI + 1
        )
        GEM11, endline = self.read_multiline_list(
            lines, startline=endline, n_listitems=NCHI * (JS0 + 1) - (NCHI + 1)
        )
        GEM12, endline = self.read_multiline_list(
            lines, startline=endline, n_listitems=NCHI * (JS0 + 1) - (NCHI + 1)
        )
        CPSURF, RADIUS = float(lines[endline].split()[0]), float(
            lines[endline].split()[1]
        )
        GEM33, endline = self.read_multiline_list(
            lines, startline=endline + 1, n_listitems=NCHI * (JS0 + 1) - (NCHI + 1)
        )
        RAXIS = float(lines[endline].split()[0])
        P0, endline = self.read_multiline_list(
            lines, startline=endline + 1, n_listitems=JS0 + 1
        )
        DP0, self.DPE = float(lines[endline].split()[0]), float(
            lines[endline].split()[1]
        )
        RBPHI, endline = self.read_multiline_list(
            lines, startline=endline + 1, n_listitems=JS0 + 1
        )
        DRBPHI0, self.DRBPHIE = float(lines[endline].split()[0]), float(
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
        XOUT, endline = self.read_multiline_list(
            lines, startline=endline + 1, n_listitems=NCHI * (JS0 + 1) - (NCHI + 1)
        )
        YOUT, endline = self.read_multiline_list(
            lines, startline=endline, n_listitems=NCHI * (JS0 + 1) - (NCHI + 1)
        )

        return (
            JS0,
            CS,
            QS,
            DQS_1,
            DQS,
            CURJ,
            DJ0,
            NCHI,
            CHI,
            GEM11,
            GEM12,
            CPSURF,
            RADIUS,
            GEM33,
            RAXIS,
            P0,
            DP0,
            RBPHI,
            DRBPHI0,
            VX,
            VY,
            EPS,
            XOUT,
            YOUT,
        )

    def read_input_density(self, run_dir: str):
        """
        The file called "density" is generated by HELENA and
        can be used as an input (in some version??) to MISHKA
        as fort.17.


        sgrid: sqrt(psi/psi_boundary)
        ne: electron density

        """
        filename = run_dir + "/fort.17"
        if not os.path.exists(filename):
            print(f"File {filename} does not exist.")
            return

        NRMAP = np.loadtxt(
            filename,
            dtype=int,
            comments="*",
            # delimiter=' ',
            converters=None,
            skiprows=0,
            # usecols=None, unpack=False, ndmin=0, encoding='bytes',
            max_rows=1,
        )

        lines = open(filename).readlines()

        sgrid, endline = self.read_multiline_list(
            lines, startline=1, n_listitems=NRMAP, n_columns=5
        )
        ne, endline = self.read_multiline_list(
            lines, startline=endline + 1, n_listitems=NRMAP, n_columns=5
        )

        return NRMAP, sgrid, ne

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

    def clean_output_files(self, run_dir):
        """
        Removes unnecessary files.

        Parameters
        ----------
        run_dir : str
            Path to the run directory.
        """
        files_to_remove = [
            # "fort.21",
            "fort.24",
            "fort.26",
            # "fort.41",
            "CASPLOT",
        ]
        for file_name in files_to_remove:
            if os.path.exists(run_dir + "/" + file_name):
                os.remove(run_dir + "/" + file_name)

        return

    def write_summary(self, run_dir: str, mpol: int, params: dict):
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
            Summary dictionary containing run directory, parameters, success
            status, Mercier criterion presence, and ballooning criterion
            presence.
        """
        file_name = os.path.join(run_dir, "summary.json")
        summary = {"run_dir": run_dir, "params": params}
        success, growthrate = self.read_output_fort20(run_dir)
        summary["success"] = success
        summary["mpol"] = mpol
        summary["ntor"] = params["ntor"]
        summary["growthrate"] = growthrate

        if "helena_dir" in params:
            summary["helena_dir"] = params["helena_dir"]
            summary["h_id"] = params["helena_dir"].split("/")[-1]

        with open(file_name, "w") as outfile:
            json.dump(summary, outfile)

        np.save(
            os.path.join(run_dir, "ntor_growthrate.npy"),
            np.array([params["ntor"], growthrate[0], growthrate[1]]),
        )
        return success, growthrate

    def extract_resistivity(
        self, filepath: str, outputpath: str, resistivity_type: str = "spitzer"
    ):
        """
        __author__ = "Hampus Nyström"

        filepath: str
            Path to the HELENA output file (fort.20).
        outputpath: str
            Path to the output file. Filename should generally be "fort.14".
        resistivity_type: str
            Type of resistivity to extract. Options are "spitzer" and "neoclassical".

        Script for writing resistivity data from helena output file to an
        outputfile that can be read by CASTOR.

        """
        # Initializing arrays for storing data
        s = []
        spitzer = []
        neo = []

        # Opening HELENA output file
        with open(filepath, "r") as f:
            # Finding and storing major radius and magnetic field
            line = f.readline()
            while "MAJOR RADIUS" not in line:
                line = f.readline()
            R = float(line.split()[-2])
            line = f.readline()
            B0 = float(line.split()[-2])

            # Finding start of conductivity data
            while "SIG(Spitz)" not in line:
                line = f.readline()
            line = f.readline()

            # Storing conductivity data and density on axis
            spl = f.readline().split()
            rho0 = float(spl[2]) * 1e19
            while len(spl) == 7:
                s.append(float(spl[0]))
                spitzer.append(float(spl[5]))
                neo.append(float(spl[6]))
                spl = f.readline().split()

        # Calculating resistivity data and normalizing to CASTOR standard
        if resistivity_type == "neoclassical":
            eta = np.array(neo)
        else:
            if resistivity_type != "spitzer":
                print("Resistivity not recognized. Reverting to spitzer.")
            eta = np.array(spitzer)

        mu0 = 4e-7 * np.pi
        amu = 1.672623e-27
        mdeut = 2.01400 * amu
        rho0 = mdeut * rho0
        norm_constant = np.sqrt(rho0 / mu0) / (R * B0)
        eta = norm_constant / eta
        deta_e = (eta[-1] - eta[-2]) / (s[-1] - s[-2])

        # adding point in s = 1
        s = np.append(s, 1.0)
        eta = np.append(eta, eta[-1] + deta_e * (s[-1] - s[-2]))

        # Writing resistivity data to output
        with open(outputpath, "w") as f:
            f.write("  %s" % (len(s) - 1))
            for i, val in enumerate(s):
                if i % 4 == 0:
                    f.write("\n")
                f.write("  %.8e" % (val))
            for i, val in enumerate(eta):
                if i % 4 == 0:
                    f.write("\n")
                f.write("  %.8e" % (val))
            f.write("\n")
            f.write("  %.8e  %.8e" % (0, deta_e))
        return eta
