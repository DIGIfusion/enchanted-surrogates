import traceback
from enchanted_surrogates.utils.precise_imports import import_runner
import os
import pandas as pd
import uuid
import sys
from pprint import pformat
            

def run_simulation_task(
        runner_config: dict, run_dir: str, params: dict = None, future=None, return_errors=False) -> dict:
    """
    Runs a single simulation task using the specified runner and parameters.
    Args:
        Future: A dask future to be used as a dependancy. Dask will not allow a future
        to run on a worker untill all dependant futures have finished and returned a value.
    Returns:
    Raises:
    """
    os.makedirs(run_dir, exist_ok=True)
    runner_type = runner_config['type']
    runner = import_runner(type=runner_type, runner_config=runner_config)
    error_info = None
    try:
        runner_output: dict = runner.single_code_run(run_dir=run_dir, params=params)
        if 'success' not in runner_output or not isinstance(runner_output.get('success'), bool):
            raise KeyError(
                "THE RUNNER'S single_code_run MUST RETURN A DICT THAT ATLEAST CONTAINS THE KEY"
                + " VALUE PAIR 'success': bool")
        
    except Exception as exc:
        msg = ["=" * 100,
                f"\nThere was a Python ERROR on when running a simulation task:", exc,
                "params:", params,
                "run_dir:", run_dir,
                traceback.format_exc()]
        # print the whole traceback and not just the last error
        error_id = f"error_id_{uuid.uuid1()}"
        error_info = {
            "success": False,
            "error_id": error_id,
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "traceback": traceback.format_exc(),
            "params": params,
            "run_dir": run_dir,
            "python_version": sys.version,
            "module": exc.__class__.__module__,
        }
        runner_output = {"success": False, "error_id": error_id}
        try:   
            print('\n'.join(msg), flush=True) # This can return a broken pipe error.
        except Exception as exc2:
            home = os.path.expanduser("~")
            extra_errors_path = os.path.join(home,f'enchanted_extra_{error_id}.txt')            
            with open(extra_errors_path, "w", encoding="utf-8") as f:
                f.write('\n'.join(msg))
                f.write(pformat(error_info))
        
    runner_output.update(params)
    runner_output['run_dir'] = run_dir
    
    if os.path.exists(run_dir):
        df_point = pd.DataFrame({r:[v] for r,v in runner_output.items()})
        df_point.to_csv(os.path.join(run_dir, 'enchanted_datapoint.csv'), header=True, index=False)
    
    if return_errors:
        return runner_output, error_info
    else:
        return runner_output
