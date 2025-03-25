"""
# runners/GENErunner.py

Defines the GENErunner class for running the GENE code.

"""

# import numpy as np
from .base import Runner
from parsers import GENEparser
import subprocess

import warnings


class TGLFrunner(Runner):
    """
    Class for running TGLF codes.

    Methods:
        __init__(*args, **kwargs)
            Initializes the TGLFrunner object.
        single_code_run(params: dict, run_dir: str) -> dict
            Runs a single TGLF code simulation.

    """

    def __init__(self, pre_run_commands:list=None, *args, **kwargs):
        """
        Initializes the GENErunner object.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        """
        self.base_parameters_file_path = kwargs.get('base_parameters_file_path', None)
        if type(pre_run_commands) == type(None):
            warnings.warn('''
                             GENE needs some pre run commands to function.
                             The default commands will be used
                             To remove this warning paste the defaults into your config file:
                             runner:
                                 type: GENErunner
                                 pre_run_commands:
                                 - export MEMORY_PER_CORE=1800
                                 - export OMP_NUM_THREADS=1
                                 - export HDF5_USE_FILE_LOCKING=FALSE
                             ''')
            self.pre_run_commands = ['export MEMORY_PER_CORE=1800', 'export OMP_NUM_THREADS=1','export HDF5_USE_FILE_LOCKING=FALSE']
        else:
            self.pre_run_commands = pre_run_commands
        self.parser = GENEparser()
        
        

    def single_code_run(self, run_dir: str, out_dir:str, params:dict=None):
        """
        Runs a single GENE code simulation.

        Args:
            params (dict): Dictionary containing parameters for the code run.
            run_dir (str): Directory path where the run command must be called from.
            out_dir (str): Directory path where the run command must be called from.

        Returns:
            (str): Containing comma seperated values of interest parsed from the GENE output 
        """
        # Edit the parameters file with the passed sample params
        self.parser.write_input_file(params, run_dir, out_dir, self.base_params_file_path)
        
        #Performing prerun commands
        subprocess.run(self.pre_run_commands)
        
        #Running GENE
        run_commands = ['set -x','srun -l -K -n $SLURM_NTASKS ./gene_lumi_csc','set +x']
        subprocess.run(run_commands) 

        # read relevant output values
        output = self.parser.read_output(out_dir)
        output = ','.join(output)
        return output
