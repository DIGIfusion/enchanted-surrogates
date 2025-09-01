"""
# runners/base.py

Defines the abstract base class Runner for running codes.

Attributes:
    ABC: Abstract Base Class module from the abc module.
    abstractmethod: Decorator for abstract methods from the abc module.

"""

from abc import ABC, abstractmethod


class Runner(ABC):
    """
    Abstract base class for running codes.

    Methods:
        single_code_run(params: dict, run_dir: str)
            Abstract method for running a single code.

    """

    @abstractmethod
    def single_code_run(self, params: dict, run_dir: str):
        """
        Abstract method for running a single code.

        Args:
            params (dict): Dictionary containing parameters for the code run.
            run_dir (str): Directory path for storing the run output.

        """
        pass
