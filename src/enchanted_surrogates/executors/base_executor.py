from abc import ABC, abstractmethod
from enchanted_surrogates.utils.logger import get_logger

log = get_logger(__name__)


class Executor(ABC):
    def __init__(self, output_dir=None, *args, **kwargs):
        self.output_dir = output_dir  # TODO rename

    @abstractmethod
    def execute(self, samples, runner_config):
        raise NotImplementedError("execute method not implemented.")

    @abstractmethod
    def clean(self):
        raise NotImplementedError("clean method not implemented.")
