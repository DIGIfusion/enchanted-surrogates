import os
import uuid
import joblib
from .base_executor import Executor
from .simulation_task import run_simulation_task
from enchanted_surrogates.utils.precise_imports import import_sampler


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

    ---

    """
    def __init__(self, *args, **kwargs):
        """
        Initialize the JoblibExecutor.

        Args:
            - *args: Additional positional arguments passed to the base Executor.
            - **kwargs: Additional keyword arguments passed to the base Executor.

        """
        super().__init__(*args, **kwargs)
        self.sampler = import_sampler(
            type=self.sampler_config.pop("type"), sampler_config=self.sampler_config)

    def start_runs(self):
        """
        Start execution of simulation runs.

        ## Workflow

        1. While the sampler has remaining budget:
            - Generate the next batch of samples using the sampler.
            - Create a unique directory for each sample run.
            - Execute the simulation tasks in parallel using `joblib.Parallel`.
            - Register the returned futures with the sampler.

        """
        while self.sampler.has_budget:
            samples: list[dict] = self.sampler.get_next_samples()
            sample_run_dirs = [os.path.join(self.base_run_dir, str(uuid.uuid4())) for _ in samples]
            new_futures = joblib.Parallel(n_jobs=-1, verbose=10)(
                joblib.delayed(run_simulation_task)(
                    self.runner_config, sample_run_dir, params=sample)
                for sample, sample_run_dir in zip(samples, sample_run_dirs)
            )
            self.sampler.register_futures(new_futures)

    def clean(self):
        """
        JoblibExecutor does not maintain any external resources or clusters. This method prints a message but does not perform any actual cleanup.
        """

        print('Joblib runner doesn\'t clean up any resources')
        return
