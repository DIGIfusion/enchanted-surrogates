from enchanted_surrogates.executors.base_executor import Executor
from enchanted_surrogates.executors.simulation_task import run_simulation_task


class LocalExecutor(Executor):

<<<<<<< HEAD
    def start_runs(self):
        self.sampler = import_sampler(
            sampler_type=self.sampler_config.pop("type"), sampler_config=self.sampler_config)
        while self.sampler.has_budget:
            samples: list[dict] = self.sampler.get_next_samples()
=======
    """
    Docstring for LocalExecutor
    """
    def start_runs(self, input: list[(str, dict)]):
>>>>>>> 71fc318 (change: moved functionalities to supervisor)

        for run_dir, sample in input:
            new_future = run_simulation_task(
                self.runner_config, run_dir, params=sample)
            self.sampler.register_future(new_future)

    def clean(self):
        print('Local runner doesn\'t clean up any resources')
        return
