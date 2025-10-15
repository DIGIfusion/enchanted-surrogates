import os
import uuid
from enchanted_surrogates.executors.base_executor import Executor
from enchanted_surrogates.executors.simulation_task import run_simulation_task
from enchanted_surrogates.utils.precise_imports import import_sampler


class LocalExecutor(Executor):

    def start_runs(self):
        self.sampler = import_sampler(
            type=self.sampler_kwargs.pop("type"), sampler_kwargs=self.sampler_kwargs)
        while self.sampler.has_budget:
            samples: list[dict] = self.sampler.get_next_samples()

            for sample in samples:
                sample_run_dir = os.path.join(self.base_run_dir, str(uuid.uuid4()))  # TODO. uuid.uuid should probably have a random seed ? 
                new_future = run_simulation_task(
                    self.runner_kwargs, sample_run_dir, params=sample)
                self.sampler.register_future(new_future)

    def clean(self):
        print('Local runner doesn\'t clean up any resources')
        return
