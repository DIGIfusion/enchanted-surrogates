"""
# runners/STATICPOOL.py

Defines the STATICPOOLrunner class, a dummy class as for the moment the sampler handles everything.

Attributes:
    Runner: Abstract base class for running codes.

"""

from .base import Runner


class STATICPOOLrunner(Runner):
    """
    Dummy class for handling static pool sampling.

    Methods:
        __init__(*args, **kwargs)
            Initializes the STATICPOOLrunner object.
        single_code_run(params: dict, run_dir: str) -> None
            Runs a single code run with static pool sampling.

    """

    def __init__(self, *args, **kwargs):
        """This is a dummy class as for the moment the sampler handles everything"""
        pass

    def single_code_run(self, params: dict, run_dir: str):
        """
        (Dummy) Runs a single code run with static pool sampling.

        Args:
            params (dict): Dictionary containing parameters for the simulation.
            run_dir (str): Directory where the code is run.

        Returns:
            None

        """
        pass
        return None
