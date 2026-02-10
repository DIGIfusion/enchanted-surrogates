import os
import uuid
from enchanted_surrogates.utils.logger import get_logger
from enchanted_surrogates.executors.base_executor import Executor
from enchanted_surrogates.executors.simulation_task import run_simulation_task

log = get_logger(__name__)

class LocalExecutor(Executor):
    """
    ---

    ## Overview

    An executor that runs simulation tasks sequentially on the local machine.
    This executor is intended for simple workflows, debugging, and environments
    where parallel or distributed execution is not required.
    Example configuration: /configs/example_local.yaml

    ---

    ## Features

    - Executes simulation tasks sequentially in the local Python process.
    - Integrates with an Enchanted Surrogates sampler for parameter exploration.
    - Generates a unique run directory for each sample.
    - Registers completed runs directly with the sampler.

    ---

    !!! notes
        - No parallelism or distributed execution is used.
        - This executor is best suited for small workloads or debugging.
        - Cleanup is minimal since no external resources are allocated.
    """

    def execute(self, input: list[(str, dict)], sampler):
        for run_dir, sample in input:
            new_future = run_simulation_task(
                self.runner_config, run_dir, params=sample)
            sampler.register_future(new_future)

    def clean(self):
        log.warning('Local runner doesn\'t clean up any resources')
        return
