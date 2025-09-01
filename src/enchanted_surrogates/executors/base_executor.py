from abc import ABC, abstractmethod

import os
import shutil
from enchanted_surrogates.samplers.base_sampler import Sampler


class Executor(ABC):
    def __init__(
        self, sampler: Sampler, runner_args, base_run_dir, output_dir=None, *args, **kwargs
    ):
        
        print("Starting Setup")
        self.sampler: Sampler = sampler
        self.runner_args      = runner_args
        self.base_run_dir     = base_run_dir
        self.output_dir       = output_dir # TODO rename 
        
    def create_run_dir(self, base_run_dir, config_filepath):
        print(
            f"Making directory of simulations at: {base_run_dir}, and copying {config_filepath} to CONFIG.yaml"
        )

        os.makedirs(base_run_dir, exist_ok=True)
        shutil.copyfile(config_filepath, os.path.join(base_run_dir, "CONFIG.yaml"))

    @abstractmethod
    def start_runs(self):
        raise NotImplementedError("start_runs method not implemented.")

    @abstractmethod
    def clean(self):
        raise NotImplementedError("clean method not implemented.")