import os, sys
import traceback
from dask.distributed import print
import importlib
import uuid
from run import load_configuration

# when submitting functions to dask workers to be ran, they cannot be functions from a class. They must be defined in another file. This file

def run_simulation_task(params: dict, base_run_dir:str, runner_args:dict, future=None, *args, **kwargs) -> dict:
        """
        Runs a single simulation task using the specified runner and parameters.
        Args:
            Future: A dask future to be used as a dependancy. Dask will not allow a future to run on a worker untill all dependant futures have finished and returned a value.
        Returns:
        Raises:
        """
        print('RUNNING SIMULATION TASK')
        print("MAKING RUN DIRECTORY")
        print('debug 1', base_run_dir)
        try:
            # This is set in the run.py and is inherited by the workers
            config_path = os.environ.get('ENCHANTED_CONFIG_PATH')
            # The downside of this is that the full config file path needs to be passed for the worker to find it. 
            config = load_configuration(config_path) 
            naming_convention = config.general['run_dir_naming_convention']
        except:
            naming_convention = 'uuid'
        
        run_dir=None # needed to get past pylint it thinks run_dir is referenced before assignment
        if naming_convention == 'uuid':
            run_dir = os.path.join(base_run_dir, str(uuid.uuid4()))
        elif naming_convention == 'params':
            run_dir = os.path.join(base_run_dir, '-'.join([str(k)+'-'+str(v) for k,v in params.items()]))
        else:
            run_dir = os.path.join(base_run_dir, str(uuid.uuid4()))
        
        print('debug A 1 exists', run_dir, os.path.exists(run_dir))
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)
        print('debug A 2 exists', run_dir, os.path.exists(run_dir))

        runner_type = runner_args['type']
        runner = getattr(importlib.import_module(f'runners.{runner_type}'),runner_type)(**runner_args)
        try:            
            print('debug B',run_dir)
            runner_output = runner.single_code_run(run_dir=run_dir, params=params, *args, **kwargs)     
        except Exception as exc:
            print("="*100,f"\nThere was a Python ERROR on a DASK worker when running a simulation task:\n{exc}\n",traceback.format_exc(), flush=True)
            #print the whole traceback and not just the last error
            runner_output = None
        if run_dir == None:
            raise ValueError('run_simulation_task must have a run_dir')
        return runner_output, run_dir
    
def print_error_wrapper(function, *args, **kwargs):
    try:
        output = function(*args, **kwargs)
    except Exception as exc:
        print("="*100,f"\nThere was a Python ERROR on a DASK worker when running {function.__name__}:\n{exc}\n",traceback.format_exc(), flush=True)
        #print the whole traceback and not just the last error
        output = None
    return output