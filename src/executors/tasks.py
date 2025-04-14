import os, sys
import traceback
from dask.distributed import print
import importlib

# when submitting functions to dask workers to be ran, they cannot be functions from a class. They must be defined in another file. This file

def run_simulation_task(runner_args:dict, run_dir:str, params: dict=None, future=None) -> dict:
        """
        Runs a single simulation task using the specified runner and parameters.
        Args:
            Future: A dask future to be used as a dependancy. Dask will not allow a future to run on a worker untill all dependant futures have finished and returned a value.
        Returns:
        Raises:
        """
        runner_type = runner_args['type']
        runner = getattr(importlib.import_module(f'runners.{runner_type}'),runner_type)(**runner_args)
        try:            
            if type(params) == type(None):
            #we are not given sampled parameters so we assume the run directories have already been set up
            # and we can simply run the code. This will happen if the executor calling this function is second (or third etc) in a pipeline
                runner_output = runner.single_code_run(run_dir)
            else:
                if run_dir != None:
                    if not os.path.exists(run_dir):
                        os.makedirs(run_dir)        
                runner_output = runner.single_code_run(run_dir=run_dir, params=params) 
            
        except Exception as exc:
            print("="*100,f"\nThere was a Python ERROR on a DASK worker when running a simulation task:\n{exc}\n",traceback.format_exc(), flush=True)
            #print the whole traceback and not just the last error
            runner_output = None
        return runner_output
    
def print_error_wrapper(function, *args, **kwargs):
    try:
        output = function(*args, **kwargs)
    except Exception as exc:
        print("="*100,f"\nThere was a Python ERROR on a DASK worker when running {function.__name__}:\n{exc}\n",traceback.format_exc(), flush=True)
        #print the whole traceback and not just the last error
        output = None
    return output