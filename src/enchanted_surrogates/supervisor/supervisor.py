"""
Supervisor module.

Provides the Supervisor class, which coordinates configuration,
execution, sampling, and result aggregation for simulation runs.
"""

import os
import sys
import warnings
import shutil
from time import sleep
import pandas as pd
from enchanted_surrogates.utils.precise_imports import import_sampler, import_executor


class Supervisor:
    """
    Creates supervisor which handles configuration, running and file output of the
    program.

    Attributes:
        args (argparse.Namespace):  Namespace containing the configuration parameters
        executor (Executor): Executor for this run
        sampler (Sampler): Sampler for this run
        base_run_dir (str): Path where runner saves result files

    Methods:
        start: Starts the simulation process. Main function of supervisor.
        create_base_run_dir: Creates base directory for simulation run results.
        all_processes_done: Returns true when all simulations are done.
        wait_all_processes: Waits in while loop until all simulations are done.
        create_dataset: Creates pandas DataFrame that includes all the
            "enchanted_datapoints.csv" files of running directories.
    """

    def __init__(self, args, config_path=None):
        """
        Initializes supervisor and sets class attributes.

        Arguments:
            args (argparse.Namespace): Namespace containing the configuration parameters.
            config_path (str or None): Optional path for configuration file where
                configuration is fetched from.
        """

        self.args = args
        self.executor = import_executor(
            type=args.executor.pop("type"), executor_config=args.executor
        )
        self.sampler = import_sampler(
            type=args.sampler.pop("type"), sampler_config=args.sampler
        )
        self.base_run_dir = args.supervisor.get("base_run_dir")
        self.create_base_run_dir(self.base_run_dir, config_path)

    def start(self):
        """
        Main function of the supervisor. Starts the simulation process. Currently
        is the only function, that is accessed outside of supervisor.py.
        Gathers samples and paths, and gives them to executor. After all processes
        are finished, creates summary file.
        """

        print("Starting runs...")

        batch_number = 0
        while self.sampler.has_budget:
            # Get samples
            samples: list[dict] = self.sampler.get_next_samples()
            # Create run_dirs with order number as name
            run_dirs = [
                os.path.join(self.base_run_dir, f"{batch_number}_{i}")
                for i in range(len(samples))
            ]
            # Call executor with folder path and samples in tuple
            self.executor.execute(zip(run_dirs, samples), self.sampler)

        self.wait_all_processes()
        enchanted_dataset = self.create_dataset()

        # Create summary csv or parquet file
        if (
            self.args.supervisor
            and self.args.supervisor.get("summary_datatype") == "parquet"
        ):
            enchanted_dataset.to_parquet(
                os.path.join(self.base_run_dir, "enchanted_dataset.parquet"),
                engine="pyarrow",
                index=True,
            )
        else:
            enchanted_dataset.to_csv(
                os.path.join(self.base_run_dir, "enchanted_dataset.csv")
            )

        # Clean run_dirs
        print("Shutting down scheduler and workers...")
        self.executor.clean()

        # TODO: Create HDF5 file

    def create_base_run_dir(self, base_run_dir, config_path):
        """
        Creates base directory for simulation run results. Checks if base_run_dir
        is empty. Prompts user option to delete existing data in base_run_dir.
        Execution is stopped if user chooses to not delete files. Copies config_file
        to base_run_dir if config_file was provided.


        Attributes:
            base_run_dir (str): Path where runner saves result files
            config_path (str or None): Optional path for configuration file where
                configuration is fetched from.
        """

        # Make sure that there is nothing in base_run_dir
        if os.path.exists(base_run_dir):
            if next(os.scandir(base_run_dir), None):
                if sys.stdout.isatty():
                    value = input(
                        str(os.path.abspath(base_run_dir))
                        + "\nFolders have content. Do you want to delete data in existing folders? y/N "
                    )
                else:
                    print(
                        str(os.path.abspath(base_run_dir))
                        + "\nFolders have content. If you wish to continue. Go delete them"
                    )
                    value = "n"

                if value.lower() in ("y", "yes"):
                    shutil.rmtree(base_run_dir)
                    print("base_run_dir was deleted")
                else:
                    print("No content was deleted. Enchanted surrogates is exited.")
                    sys.exit(1)

        # Create base run dir
        os.makedirs(base_run_dir, exist_ok=True)

        # Move config path to base_run_dir if config path is given
        if config_path is not None:
            new_config_path = os.path.join(base_run_dir, os.path.basename(config_path))
            print(f"Moving config file... from {config_path} to {new_config_path}")
            try:
                shutil.copy(config_path, new_config_path)
            except OSError as exe:
                warnings.warn(
                    "Failed to copy configuration file to base run directory.\n"
                    "Try using the full path to the config file. \n"
                    f"Source: '{config_path}'\n"
                    f"Target: '{new_config_path}'\n"
                    f"Error type: {type(exe).__name__}\n"
                    f"Error message: {exe}"
                )

    def all_processes_done(self):
        """
        Monitors simulation processes and returns boolean describing state.
        Helper function for wait_all_processes.

        Return:
            True when all simulations are done. Helper function for
                wait_all_processes. Checks inside base_run_dir if folders inside it
                contain "enchanted_datapoint.csv" files.
            False If any runner has not yet created the csv file
        """

        for name in os.listdir(self.base_run_dir):
            folder_path = os.path.join(self.base_run_dir, name)
            if os.path.isdir(folder_path):
                datapoint_file = os.path.join(folder_path, "enchanted_datapoint.csv")
                if not os.path.isfile(datapoint_file):
                    return False

        return True

    def wait_all_processes(self):
        """
        Waits in while loop until all simulations are done. Loop is broken
        when all_processes_done returns true. Checks condition once in
        second.
        """

        while True:
            if self.all_processes_done():
                break
            sleep(1)

    def create_dataset(self):
        """
        Creates pandas DataFrame that includes all the "enchanted_datapoints.csv"
        files of running directories inside base_run_dir.

        Return:
            pandas.DataFrame containing all the enchanted_datapoint.csv files
            created by runners.

        """
        enchanted_dataset = pd.DataFrame()
        for name in os.listdir(self.base_run_dir):
            folder_path = os.path.join(self.base_run_dir, name)
            if os.path.isdir(folder_path):
                datapoint_file = os.path.join(folder_path, "enchanted_datapoint.csv")
                if os.path.isfile(datapoint_file):
                    enchanted_datapoint = pd.read_csv(datapoint_file)
                    enchanted_dataset = pd.concat(
                        [enchanted_dataset, enchanted_datapoint]
                    )
        return enchanted_dataset
