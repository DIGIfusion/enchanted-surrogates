import os
import ast
from .base import Parser
import numpy as np
import scipy.stats as stats
import h5py
try: 
    import DREAM
except ImportError as e:
    print(
        "Cannot find DREAM.",
        "Please add the $DREAMPATH/py to your PYTHONPATH environment variable before running.",
        str(e),
    )
    

class DREAMparser(Parser):
    """
    An I/O parser for DREAM.

    Methods:
        __init__()
            Initializes the DREAMparser object.
        write_input_file(params: dict, run_dir: str) -> None
            Writes a sample input file.
        read_output_file(params: dict, run_dir: str) -> dict
            Reads the output file containing the input parameters.

    """

    def __init__(self):
        """
        Initializes the DREAMparser object.

        """
        pass

    def write_input_file(self, params: dict, run_dir: str, base_input_file_path: str):
        """
        Writes a sample input file.

        Args:
            params (dict): Dictionary containing input parameters.
            run_dir (str): Directory where the input file is written.
            base_input_file_path (str): Path to a basecase input file .h5.

        Returns:
            None

        """
        # Initialize the DREAM settings and load the base input settings.
        ds = DREAM.DREAMSettings()
        try:
            ds.load(base_input_file_path)
        except:
            print("Give the base case input file as DREAM input hdf5")

        # Initialize the input and output files as run_dir/input.h5 and 
        # run_dir/output.h5.
        if os.path.exists(run_dir):
            input_path = os.path.join(run_dir, 'input.h5')
        else:
            raise FileNotFoundError(f"Couldnt find {run_dir}")

        ds.output.setFilename('output.h5')

        # Anything that the user aims to sample, should be defined
        # here through options.           
        if 'V_loop_wall' in params and 'tau_V_loop' in params:
            tvec = np.linspace(0, ds.timestep.tmax, ds.timestep.nt)
            Vl = params['V_loop_wall']*np.exp(-tvec/params['tau_V_loop'])
            inverse_wall_time = ds.eqsys.E_field.inverse_wall_time
            R = ds.eqsys.E_field.R0
            ds.eqsys.E_field.setBoundaryCondition(
                bctype = DREAM.Settings.Equations.ElectricField.BC_TYPE_TRANSFORMER,
                inverse_wall_time = inverse_wall_time,
                R0=R,
                V_loop_wall_R0 = Vl,
                times = tvec,
                )
        if 'dBB' in params and 'alpha' in params and 'beta' in params:
            tgrid = np.array([0, ds.timestep.tmax])
            r = np.linspace(0, ds.radialgrid.a/100, int(ds.radialgrid.nr))
            dBr = stats.gamma.pdf(r, a=params['alpha'], scale=1.0/params['beta'])
            sumdb = np.sum(dBr)
            dBr = params['dBB']*dBr/sumdb
            dBr = np.array([[dBr], [dBr]]).reshape(2, 30)
            ds.eqsys.f_re.transport.setMagneticPerturbation(dBr, t=tgrid, r=r)
            ds.eqsys.f_re.transport.setBoundaryCondition(
                DREAM.Settings.Equations.DistributionFunction.BC_F_0,
                )

        # Save the input file in run_dir.
        ds.save(input_path)
        print(f"DREAM input written to: {input_path}.")
          
        

    def read_output_file(self, run_dir: str):
        """
        Reads the input and output files from the run directory

        Args:
            run_dir (str): Directory where the output file is located.

        Returns:
            dict: Dictionary containing the settings and the output dictionaries.

        Raises:
            FileNotFoundError: If the output file does not exist.

        """
        file_name_input = os.path.join(run_dir, "input.h5")
        file_name_output = os.path.join(run_dir, "output.h5")
        if not os.path.exists(file_name_input):
            raise FileNotFoundError(f"{file_name_input}")
        if not os.path.exists(file_name_output):
            raise FileNotFoundError(f"{file_name_output}")
        ds = DREAM.DREAMSettings()
        do = DREAM.DREAMOutput()
        ds.load(file_name_input)
        do.load(file_name_output)
        outputdict = {'input':ds, 'output':do}
        return outputdict


