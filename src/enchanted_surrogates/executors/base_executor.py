from abc import ABC, abstractmethod


class Executor(ABC):
    def __init__(
        self, output_dir=None, *args, **kwargs
    ):
        #self.sampler_config = sampler_config
        #self.runner_config = runner_config
        self.output_dir = output_dir  # TODO rename

    @abstractmethod
    def execute(self, samples, sampler):
        raise NotImplementedError("execute method not implemented.")

    @abstractmethod
    def clean(self):
        raise NotImplementedError("clean method not implemented.")

    def set_runner_config(self, config):
        self.runner_config = config
