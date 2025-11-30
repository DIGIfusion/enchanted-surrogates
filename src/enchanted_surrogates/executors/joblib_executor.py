
import os
import uuid
import joblib
from .base_executor import Executor
from .simulation_task import run_simulation_task
from enchanted_surrogates.utils.logger import get_logger
from enchanted_surrogates.utils.precise_imports import import_sampler

log = get_logger(__name__)

class JoblibExecutor(Executor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sampler = import_sampler(
            sampler_type=self.sampler_config.pop("type"), sampler_config=self.sampler_config)

    def start_runs(self):
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
        log.warning('Joblib runner doesn\'t clean up any resources')
        return
