import importlib 
import traceback 
import os 

def run_simulation_task(runner_args:dict, run_dir:str, sample_params: dict=None, future=None) -> dict:
    """
    Runs a single simulation task using the specified runner and parameters.
    Args:
        Future: A dask future to be used as a dependancy. Dask will not allow a future to run on a worker untill all dependant futures have finished and returned a value.
    Returns:
    Raises:
    """
    runner_type = runner_args['type']
    runner = getattr(importlib.import_module(f'enchanted_surrogates.runners.{runner_type}'), runner_type)(**runner_args)
    try:            
        runner_output: dict = runner.single_code_run(run_dir=run_dir, params=sample_params)
        
    except Exception as exc:
        print("="*100,f"\nThere was a Python ERROR on a DASK worker when running a simulation task:\n{exc}\n",traceback.format_exc(), flush=True)
        #print the whole traceback and not just the last error
        runner_output = {"success": False} 
    runner_output.update(sample_params)
    runner_output['run_dir'] = run_dir
    return runner_output