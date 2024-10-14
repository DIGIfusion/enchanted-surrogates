"""
executors/base.py

Any executor should handle the generation of data leveraging parallelization in some way.
To do so, the executor needs a way of sampling new points (sampler), which are then ran with a simulator (runner).

Any executor should have the following attributes:
    - A sampler
    - The arguments needed to initialize a runner

As per run.py, any executor must have the following methods:

- clean() -> post database generation, how to clean up everything
- start_runs() -> start database generation

An executor could be, for example, enabled by Dask, or Dakota, or some clever MPI.
"""

import os
import shutil
from abc import ABC, abstractmethod
import uuid
import runners


# TODO: move to seperate file, tasks.py?
def run_simulation_task(
    runner_args: dict, params_from_sampler: dict, base_run_dir: str
) -> dict:
    """
    Runs a single simulation task using the specified runner and parameters.

    Args:
        runner_args (dict): A dictionary of arguments required to initialize the runner.
        params_from_sampler (dict): Parameters generated by the sampler for the simulation run.
        base_run_dir (str): The base directory where the simulation results will be stored.

    Returns:
        dict: A dictionary containing the input parameters and the output from the simulation run.

    Raises:
        AttributeError: If the runner type specified in runner_args does not exist.
        FileNotFoundError: If the base_run_dir does not exist and cannot be created.
        Exception: For other exceptions that may arise during the simulation run.
    """
    run_dir = os.path.join(base_run_dir, str(uuid.uuid4()))
    runner = getattr(runners, runner_args["type"])(**runner_args)
    runner_output = runner.single_code_run(params_from_sampler, run_dir)
    result = {"input": params_from_sampler, "output": runner_output}
    return result


class Executor(ABC):
    """
    Abstract base class for executing tasks with a given sampler and configuration.

    Attributes:
        sampler: The sampler object providing samples for execution.
        runner_args: Arguments required for running the tasks.
        base_run_dir: The base directory where runs will be executed and stored.
        config_filepath: The file path to the configuration file.
        clients: A list to store client connections.

    Methods:
        start_runs(): Abstract method to start execution of runs. Must be implemented by subclasses.
        clean(): Closes all client connections.
    """

    def __init__(
        self, sampler, runner_args, base_run_dir, config_filepath, *args, **kwargs
    ):
        """
        Initializes the Executor with the given parameters and prepares the environment for execution.

        Args:
            sampler: The sampler object providing samples for execution.
            runner_args: Arguments required for running the tasks.
            base_run_dir: The base directory where runs will be executed and stored.
            config_filepath: The file path to the configuration file.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        print("Starting Setup")
        self.sampler = sampler  # kwargs.get('sampler')
        self.runner_args = runner_args  # kwargs.get('runner_args')
        self.base_run_dir = base_run_dir  # , kwargs.get('base_run_dir')
        # self.max_samples = self.sampler.total_budget
        self.clients = []
        self.create_run_dir(self.base_run_dir, config_filepath)

    def create_run_dir(self, base_run_dir, config_filepath):
        print(
            f"Making directory of simulations at: {base_run_dir}, and copying {config_filepath} to CONFIG.yaml"
        )

        os.makedirs(base_run_dir, exist_ok=True)
        shutil.copyfile(config_filepath, os.path.join(self.sampler.save_dir, "CONFIG.yaml"))

    @abstractmethod
    def start_runs(self):
        """
        Abstract method to start execution of runs. This method must be implemented by subclasses.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError()

    @abstractmethod
    def clean(self):
        """Abstrat method to clean up the processes created by the executor"""
        raise NotImplementedError()
