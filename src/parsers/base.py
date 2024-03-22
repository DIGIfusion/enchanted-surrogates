# parsers/base.py

from abc import ABC, abstractmethod

class Parser(ABC):
    @abstractmethod
    def write_input_file(self, params: dict, run_dir: str):
        pass