import os, sys
import traceback
from dask.distributed import print

# when submitting functions to dask workers to be ran, they cannot be functions from a class. They must be defined in another file. This file

def run_simulation_task(runner, run_dir:str, params_from_sampler: dict=None) -> dict:
        """
        Runs a single simulation task using the specified runner and parameters.
        Args:
        Returns:
        Raises:
        """
        try:
            if type(params_from_sampler) == type(None):
            #we are not given sampled parameters so we assume the run directories have already been set up
            # and we can simply run the code. This will happen if the executor calling this function is second (or third etc) in a pipeline
                print('ASSUMING RUN DIRECTORIES HAVE BEEN SET UP')
                runner_output = runner.single_code_run(run_dir)
            else:
                print('MAKING RUN DIRECTORY', run_dir)
                os.mkdir(run_dir)
                runner_output = runner.single_code_run(run_dir, params_from_sampler) 
            
        except Exception as exc:
            print("="*100,f"\nThere was a Python ERROR on a DASK worker when running a simulation task:\n{exc}\n",traceback.format_exc(), flush=True)
            #print the whole traceback and not just the last error
            runner_output = None
        return runner_output