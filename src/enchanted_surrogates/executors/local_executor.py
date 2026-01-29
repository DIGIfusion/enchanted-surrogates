import os
import uuid
from enchanted_surrogates.executors.base_executor import Executor
from enchanted_surrogates.executors.simulation_task import run_simulation_task
from enchanted_surrogates.utils.precise_imports import import_sampler


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

    def start_runs(self):
        """
        Start execution of simulation runs.

        ## Workflow

        1. Initialize the sampler using the provided `sampler_config`.
        2. While the sampler has remaining budget:
            - Generate the next batch of samples.
            - For each sample:
                - Create a unique run directory.
                - Execute the simulation task locally.
                - Register the result with the sampler.

        !!! notes
            - Each simulation is executed synchronously.
            - Run directories are created using UUIDs to avoid name collisions.
            - This method blocks until all simulations are completed.
        """
        self.sampler = import_sampler(
            type=self.sampler_config.pop("type"), sampler_config=self.sampler_config)
        while self.sampler.has_budget:
            samples: list[dict] = self.sampler.get_next_samples()

            for sample in samples:
                sample_run_dir = os.path.join(self.base_run_dir, str(uuid.uuid4()))  # TODO. uuid.uuid should probably have a random seed ? 
                new_future = run_simulation_task(
                    self.runner_config, sample_run_dir, params=sample)
                self.sampler.register_future(new_future)

    def clean(self):
        """

        LocalExecutor does not manage any external resources or worker processes. This method prints a message but does not perform any cleanup actions.
        
        """
        print('Local runner doesn\'t clean up any resources')
        return
