import traceback
from enchanted_surrogates.utils.logger import get_logger
from enchanted_surrogates.utils.precise_imports import import_runner
import os
import pandas as pd

log = get_logger(__name__)

def run_simulation_task(
        runner_config: dict, run_dir: str, params: dict = None, future=None) -> dict:
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
    try:
        runner_output: dict = runner.single_code_run(run_dir=run_dir, params=params)

    except Exception as exc:
        log.error("=" * 100)
        log.error("There was a Python ERROR on when running a simulation task:")
        log.error(exc)
        log.error(f"params: {params}")
        log.error(f"run_dir: {run_dir}")
        log.error(traceback.format_exc())
        # print the whole traceback and not just the last error
        runner_output = {"success": False} 
    if 'success' not in runner_output or not isinstance(runner_output.get('success'), bool):
        raise KeyError(
            "THE RUNNER'S single_code_run MUST RETURN A DICT THAT ATLEAST CONTAINS THE KEY"
            + " VALUE PAIR 'success': bool")
    runner_output.update(params)
    runner_output['run_dir'] = run_dir
    df_point = pd.DataFrame({r:[v] for r,v in runner_output.items()})
    df_point.to_csv(os.path.join(run_dir, 'enchanted_datapoint.csv'), header=True, index=False)
    return runner_output
