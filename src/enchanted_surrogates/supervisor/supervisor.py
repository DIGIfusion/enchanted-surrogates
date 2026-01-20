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
import h5py
import numpy as np
import pandas as pd
from enchanted_surrogates.supervisor.run_group import RunGroup
from enchanted_surrogates.samplers.nested_sampler import NestedSampler
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
        create_hdf5: Creates hdf5 structured file that includes numeric data of
            enchanted_dataset and metadata
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
        self.executors = self.import_executors(args)
        self.samplers = self.import_samplers(args)
        groups = self.import_run_groups(args)

        self.groups: list[RunGroup] = []
        for group in groups:
            run_group = RunGroup(
                self.executors[group["executor"]],
                self.samplers[group["sampler"]],
                args.runners[group["runner"]]
            )
            run_group.executor.set_runner_config(run_group.runner)
            self.groups.append(run_group)

        self.base_run_dir = args.supervisor.get("base_run_dir")

        if self.base_run_dir is None:
            raise ValueError("base_run_dir is not set in the provided configuration")

        self.create_base_run_dir(self.base_run_dir, config_path)

    def start(self):
        """
        Main function of the supervisor. Starts the simulation process. Currently
        is the only function, that is accessed outside of supervisor.py.
        Gathers samples and paths, and gives them to executor. After all processes
        are finished, creates summary file.
        """

        print("Starting runs...")

        rows = [{}]
        for depth, group in enumerate(self.groups):
            next_rows = []
            batch_number = 0

            while group.sampler.has_budget:
                samples = group.sampler.get_next_samples()

                expanded = []
                for parent in rows:
                    for sample in samples:
                        merged = {**parent, **sample}
                        expanded.append(merged)

                run_dirs = [
                    os.path.join(self.base_run_dir, f"d{depth}_b{batch_number}_r{i}")
                    for i in range(len(expanded))
                ]

                group.executor.execute(zip(run_dirs, expanded), group.sampler)

                self.wait_all_processes(f"d{depth}")

                for run_dir, row in zip(run_dirs, expanded):
                    datapoint_file = os.path.join(run_dir, "enchanted_datapoint.csv")
                    if os.path.isfile(datapoint_file):
                        result = pd.read_csv(datapoint_file).iloc[0].to_dict()
                        combined = {**row, **result}
                        next_rows.append(combined)

                batch_number += 1

            rows = next_rows

        enchanted_dataset = pd.DataFrame(rows)

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

        # Create HDF5 file by default
        if not hasattr(self.args, "storage") or self.args.storage.get("type") != "None":
            self.create_hdf5(enchanted_dataset)

        # Clean run_dirs
        print("Shutting down scheduler and workers...")
        for group in self.groups:
            group.executor.clean()

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

    def all_processes_done(self, filter=None):
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
            if not filter or str(filter) in str(name):
                folder_path = os.path.join(self.base_run_dir, name)
                if os.path.isdir(folder_path):
                    datapoint_file = os.path.join(folder_path, "enchanted_datapoint.csv")
                    if not os.path.isfile(datapoint_file):
                        return False

        return True

    def wait_all_processes(self, filter=None):
        """
        Waits in while loop until all simulations are done. Loop is broken
        when all_processes_done returns true. Checks condition once in
        second.
        """

        while True:
            if self.all_processes_done(filter):
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

    def create_hdf5(self, enchanted_dataset: pd.DataFrame):
        """
        Creates hdf5 and saves storage file in base_run_dir with name runs.h5
        Includes aggregated data from enchanted_dataset and run specific data
        in structured format. Dataset has only numeric values, column headers
        are saved separately in in same location. Metadata includes types for
        sampler, executor and runner.

        Attributes:
            - enchanted_dataset (pd.DataFrame): Dataframe containing all run results

        """
        h5_path = os.path.join(self.base_run_dir, "runs.h5")

        with h5py.File(h5_path, "w") as f:
            # Aggregated dataset
            agg_group = f.create_group("data/aggregated")

            agg_group.create_dataset(
                "values",
                data=enchanted_dataset.select_dtypes(include=[np.number]).to_numpy(),
            )

            agg_group.create_dataset(
                "columns",
                data=np.array(
                    enchanted_dataset.select_dtypes(include=[np.number]).columns,
                    dtype=h5py.string_dtype(encoding="utf-8"),
                ),
            )

            # Run directory datasets
            runs_group = f.create_group("data/runs")

            for name in os.listdir(self.base_run_dir):
                folder_path = os.path.join(self.base_run_dir, name)
                csv_path = os.path.join(folder_path, "enchanted_datapoint.csv")

                if not os.path.isfile(csv_path):
                    continue

                df = pd.read_csv(csv_path)
                run_group = runs_group.create_group(name)

                # Select only numeric values
                numeric_df = df.select_dtypes(include=[np.number])

                run_group.create_dataset("values", data=numeric_df.to_numpy())

                run_group.create_dataset(
                    "columns",
                    data=np.array(numeric_df.columns, dtype=h5py.string_dtype("utf-8")),
                )

            # Metadata
            meta_group = f.create_group("metadata")
            meta_group.attrs["executor"] = str(self.args.executor.get("type"))
            meta_group.attrs["sampler"] = str(self.args.sampler.get("type"))
            meta_group.attrs["runner"] = str(self.args.runner.get("type"))
