# executors/base.py
import os
import shutil
from abc import ABC
import uuid
import runners





class Executor(ABC):
    def __init__(
        self, sampler, runner_args, base_run_dir, config_filepath, *args, **kwargs
    ):
        print("Starting Setup")
        self.sampler = sampler  # kwargs.get('sampler')
        self.runner_args = runner_args  # kwargs.get('runner_args')
        self.base_run_dir = base_run_dir  # , kwargs.get('base_run_dir')
        self.max_samples = self.sampler.total_budget
        self.config_filepath = config_filepath  # kwargs.get('config_filepath')
        self.client = None
        print(config_filepath)
        print(f"Making directory of simulations at: {self.base_run_dir}")
        os.makedirs(self.base_run_dir, exist_ok=True)

        print("Base Executor Initialization")

        shutil.copyfile(config_filepath, os.path.join(self.base_run_dir, "CONFIG.yaml"))

    def run_simulation_task(runner_args, params_from_sampler, base_run_dir):
        print("Making Run dir")
        run_dir = os.path.join(base_run_dir, str(uuid.uuid4()))
        os.mkdir(run_dir)
        runner = getattr(runners, runner_args["type"])(**runner_args)
        result = runner.single_code_run(params_from_sampler, run_dir)
        return result, params_from_sampler

    