import os
import uuid
import joblib
from .base_executor import Executor
from .simulation_task import run_simulation_task
from enchanted_surrogates.utils.logger import get_logger
from enchanted_surrogates.utils.precise_imports import import_sampler

log = get_logger(__name__)


class JoblibExecutor(Executor):
    """
    ---

    ## Overview

    An executor that runs simulations in parallel using `joblib.Parallel`.
    This executor integrates with an Enchanted Surrogates sampler to generate
    parameter configurations and execute tasks concurrently on the local machine.

    ---

    ## Features

    - Uses `joblib` to parallelize simulation tasks across available CPU cores.
    - Integrates with Enchanted Surrogates sampler for parameter exploration.
    - Generates a unique run directory for each sample.
    - Automatically registers completed futures with the sampler.

    ---

    !!! notes
        - This executor does not manage clusters; it runs everything locally.
        - No dynamic scaling or distributed execution is supported.
        - Cleanup is minimal since `joblib` runs in-process and does not leave persistent resources.

    """

    def execute(self, input: list[(str, dict)], sampler):
        """
        Execute simulation tasks in parallel using joblib.

        Params:
            input (list[(str, dict)]): A list of simulation tasks to execute. Each element is a tuple consisting of path to the directory where the simulation run should be executed and dictionary of simulation parameters.
            sampler (object): Sampler instance responsible for tracking submitted simulation tasks
        """
        joblib.Parallel(n_jobs=-1, verbose=10)(
            joblib.delayed(run_simulation_task)(
                self.runner_config, sample_run_dir, params=sample
            )
            for sample_run_dir, sample in input
        )

    def clean(self):
        log.warning("Joblib runner doesn't clean up any resources")
        return
