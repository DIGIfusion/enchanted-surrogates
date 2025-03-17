import os, sys
import traceback
from dask.distributed import print

# when submitting functions to dask workers to be ran, they cannot be functions from a class. They must be defined in another file. This file

def run_simulation_task(runner, run_dir:str, out_dir:str, params_from_sampler: dict=None, future=None) -> dict:
        """
        Runs a single simulation task using the specified runner and parameters.
        Args:
            Future: A dask future to be used as a dependancy. Dask will not allow a future to run on a worker untill all dependant futures have finished and returned a value.
        Returns:
        Raises:
        """
        try:
            print('RUNNING SIMULATION TASK:')
            if future!=None:
                #Calling future.result() is needed to establish dependancy on the future.
                print('RESULT OF INPUTTED FUTURE:', future.result())
            
            if out_dir == None:
                out_dir = run_dir
            if type(params_from_sampler) == type(None):
            #we are not given sampled parameters so we assume the run directories have already been set up
            # and we can simply run the code. This will happen if the executor calling this function is second (or third etc) in a pipeline
                print('ASSUMING RUN DIRECTORIES HAVE BEEN SET UP')
                print('run_dir:', run_dir,'out_dir:',out_dir)
                print(os.system(f'ls {run_dir}'))
                runner_output = runner.single_code_run(run_dir, out_dir)
            else:
                print('MAKING RUN DIRECTORY', run_dir)
                if not os.path.exists(run_dir):
                    os.mkdir(run_dir)
                runner_output = runner.single_code_run(run_dir, out_dir, params_from_sampler) 
            
        except Exception as exc:
            print("="*100,f"\nThere was a Python ERROR on a DASK worker when running a simulation task:\n{exc}\n",traceback.format_exc(), flush=True)
            #print the whole traceback and not just the last error
            runner_output = None
        return runner_output