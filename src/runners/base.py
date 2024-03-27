# runners/base.py

from abc import ABC, abstractmethod


class Runner(ABC):
    @abstractmethod
    def single_code_run(self, params: dict, run_dir: str):
        """ Logic to run a single code """
        pass
