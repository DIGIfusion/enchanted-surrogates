from abc import ABC, abstractmethod
from enchanted_surrogates.utils.logger import get_logger

log = get_logger(__name__)

class Executor(ABC):
    def __init__(
        self, runner_config=None, output_dir=None, *args, **kwargs
    ):
        self.runner_config = runner_config
        self.output_dir = output_dir  # TODO rename

    @abstractmethod
    def execute(self, samples, sampler):
        raise NotImplementedError("execute method not implemented.")

    @abstractmethod
    def clean(self):
        raise NotImplementedError("clean method not implemented.")
