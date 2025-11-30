
import os
import shutil
from abc import ABC, abstractmethod
from enchanted_surrogates.utils.logger import get_logger

log = get_logger(__name__)

class Executor(ABC):
    def __init__(
        self, sampler_config, runner_config, base_run_dir, output_dir=None, *args, **kwargs
    ):
        self.sampler_config = sampler_config
        self.runner_config = runner_config
        self.base_run_dir = base_run_dir
        self.output_dir = output_dir  # TODO rename

    def create_base_run_dir(self, base_run_dir, config_filepath):
        log.info(
            f"Making directory of simulations at: {base_run_dir}.",
            f"Copying {config_filepath} to CONFIG.yaml."
        )

        os.makedirs(base_run_dir, exist_ok=True)
        shutil.copyfile(config_filepath, os.path.join(base_run_dir, "CONFIG.yaml"))

    @abstractmethod
    def start_runs(self):
        raise NotImplementedError("start_runs method not implemented.")

    @abstractmethod
    def clean(self):
        raise NotImplementedError("clean method not implemented.")
