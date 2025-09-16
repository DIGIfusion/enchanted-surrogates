
import os
import uuid
import joblib
from .base_executor import Executor
from .simulation_task import run_simulation_task


class JoblibExecutor(Executor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def start_runs(self):
        while self.sampler.has_budget:
            samples: list[dict] = self.sampler.get_next_samples()
            sample_run_dirs = [os.path.join(self.base_run_dir, str(uuid.uuid4())) for _ in samples]
            new_futures = joblib.Parallel(n_jobs=-1, verbose=10)(
                joblib.delayed(run_simulation_task)(
                    self.runner_kwargs, sample_run_dir, params=sample)
                for sample, sample_run_dir in zip(samples, sample_run_dirs)
            )
            self.sampler.register_futures(new_futures)

    def clean(self):
        print('Joblib runner doesn\'t clean up any resources')
        return
