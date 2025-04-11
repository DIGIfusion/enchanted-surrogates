"""
# runners/GENErunner.py

Defines the GENErunner class for running the GENE code.

"""

# import numpy as np
from .base import Runner
from parsers.GENEparser import GENEparser
import subprocess
import sys, os
import warnings
from dask.distributed import print
from time import sleep


class GENErunner(Runner):
    """
    Class for running the GENE code.

    Methods:
        __init__(*args, **kwargs)
            Initializes the GENErunner object.
        single_code_run(params: dict, run_dir: str) -> dict
            Runs a single GENE code simulation.

    """

    def __init__(self, executable_path:str, return_mode='deeplasma', *args, **kwargs):
        """
        Initializes the GENErunner object.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
            return_mode (str): Either 'deeplasma' or 'growthrate'. This changes what will be returned.
        """
        self.executable_path = executable_path
        self.base_parameters_file_path = kwargs.get('base_parameters_file_path', None)
        self.parser = GENEparser()
        self.return_mode = return_mode
        
        

    def single_code_run(self, run_dir: str, params:dict=None):
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
        self.parser.write_input_file(params, run_dir, self.base_parameters_file_path)
        
        # Move to run directory for executing the commands
        print('RUNNING GENE IN RUN DIR:',run_dir)
        print('slurm n tasks',os.getenv('SLURM_NTASKS'))
        print('mem _per_core',os.getenv('MEMORY_PER_CORE'))
        print('omp_num_threads',os.getenv('OMP_NUM_THREADS'))
        print('mem_per_node', os.getenv('SLURM_MEM_PER_NODE'))
        
        print('env OMPI_COMM_WORLD_SIZE:',os.getenv('OMPI_COMM_WORLD_SIZE'))
        print('env SLURM_NTASKS:',os.getenv('SLURM_NTASKS'))
        print('env MPI_NUM_PROCS:',os.getenv('MPI_NUM_PROCS'))
        
        
        
        run_command = f"cd {run_dir} && export MEMORY_PER_CORE=1800 && export OMP_NUM_THREADS=1 && export HDF5_USE_FILE_LOCKING=FALSE && set -x && srun --output={os.path.join(run_dir,'std_out.txt')} --error={os.path.join(run_dir,'err_out.txt')} -l -K -n $SLURM_NTASKS {self.executable_path} && set +x"#"./scanscript --np $SLURM_NTASKS --ppn $SLURM_NTASKS_PER_NODE --mps 4 --syscall='srun -l -K -n $SLURM_NTASKS ./gene_lumi_csc'"
        # && export MEMORY_PER_CORE=1800
        result = subprocess.run(run_command, shell=True, capture_output=False, text=True)
        # out = result.stdout
        # err = result.stderr
        # print('OUT', out)
        # print('ERR', err)
        
        finished = False
        while not finished:
            files = os.listdir(run_dir)
            if 'GENE.finished' in files:
                finished = True
            sleep(5)
        
        # read relevant output values
        if self.return_mode=='deeplasma':
            output = self.parser.read_output(run_dir)
            output = [str(v) for v in output]
        elif self.return_mode=='growthrate':
            ky, growthrate, frequency = self.parser.read_omega(run_dir)
            output = [str(growthrate)]
            
        params_list = [str(v) for k,v in params.items()]
        return_list = params_list + output
        return ','.join(return_list)