import importlib 
import traceback
from enchanted_surrogates.utils.precise_imports import import_runner
import os 

def run_simulation_task(runner_kwargs:dict, run_dir:str, params: dict=None, future=None) -> dict:
    """
    Runs a single simulation task using the specified runner and parameters.
    Args:
        Future: A dask future to be used as a dependancy. Dask will not allow a future to run on a worker untill all dependant futures have finished and returned a value.
    Returns:
    Raises:
    """
    os.makedirs(run_dir, exist_ok=True)
    runner_type = runner_kwargs['type']
    runner = import_runner(type=runner_type, runner_kwargs=runner_kwargs)
    try:            
        runner_output: dict = runner.single_code_run(run_dir=run_dir, params=params)
        
    except Exception as exc:
        print("="*100,f"\nThere was a Python ERROR on when running a simulation task:\n{exc}\n",
              "PARAMS:", params, "\n",
              "RUN_DIR:", run_dir, "\n",
              traceback.format_exc(), flush=True)
        #print the whole traceback and not just the last error
        runner_output = {"success": False} 
    if not 'success' in runner_output:
        raise KeyError("THE RUNNERS SINGLE CODE RUN MUST RETURN A DICT THAT ATLEAST CONTAINS THE KEY VALUE PAIR 'success':bool")
    elif not isinstance(runner_output.get('success'), bool):
        raise TypeError("THE RUNNERS SINGLE CODE RUN MUST RETURN A DICT THAT CONTAINS THE KEY 'success' WITH VALUE OF *TYPE* bool")
    # handle the error or raise an exception

    runner_output.update(params)
    runner_output['run_dir'] = run_dir
    return runner_output