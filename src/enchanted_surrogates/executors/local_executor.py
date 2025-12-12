from enchanted_surrogates.executors.base_executor import Executor
from enchanted_surrogates.executors.simulation_task import run_simulation_task


class LocalExecutor(Executor):

    """
    Docstring for LocalExecutor
    """
    def execute(self, input: list[(str, dict)], sampler):

        for run_dir, sample in input:
            new_future = run_simulation_task(
                self.runner_config, run_dir, params=sample)
            sampler.register_future(new_future)

    def clean(self):
        print('Local runner doesn\'t clean up any resources')
        return
