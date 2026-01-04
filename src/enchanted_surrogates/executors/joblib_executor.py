
import os
import uuid
import joblib
from .base_executor import Executor
from .simulation_task import run_simulation_task
from enchanted_surrogates.utils.precise_imports import import_sampler


class JoblibExecutor(Executor):
    """
    Docstring for JoblibExecutor TODO
    """

    def execute(self, input: list[(str, dict)], sampler):
        new_futures = joblib.Parallel(n_jobs=-1, verbose=10)(
            joblib.delayed(run_simulation_task)(
                self.runner_config, sample_run_dir, params=sample)
            for sample_run_dir, sample in input
        )
        sampler.register_futures(new_futures)

    def clean(self):
        print('Joblib runner doesn\'t clean up any resources')
        return
