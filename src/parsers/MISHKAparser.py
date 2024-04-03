from .base import Parser

# import subprocess
import os
import f90nml
import numpy as np
import math


class MISHKAparser(Parser):
    """An I/O parser for MISHKA

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

    def __init__(self, default_namelist: str):
        self.default_namelist = default_namelist

    def write_input_file(self, params: dict, run_dir: str):
        print("parser params", params)
        print("Writing to", run_dir)

        if os.path.exists(run_dir):
            input_fpath = os.path.join(run_dir, "fort.10")
        else:
            raise FileNotFoundError(f"Couldnt find {run_dir}")

        # TODO: params should be dict, not list
        namelist = f90nml.read(self.default_namelist)
        namelist["newrun"][0]["ntor"] = -int(params[0])

        f90nml.write(namelist, input_fpath)
        print(input_fpath)

    def read_output_file_fort20(self, run_dir: str):
        """
        The main output file.
        """
        raise NotImplementedError

    def read_output_file_fort22(self, run_dir: str):
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
