"""
Supervisor module.

Provides the Supervisor class, which coordinates configuration,
execution, sampling, and result aggregation for simulation runs.
"""

import os
import sys
import warnings
import shutil
import glob
from time import sleep
import h5py
import numpy as np
import pandas as pd
from enchanted_surrogates.utils.logger import get_logger
from enchanted_surrogates.supervisor.run_data import RunData
from enchanted_surrogates.supervisor.nested_imports import (
    RunGroup,
    parse_all_run_groups,
    import_saved_files_list,
)

log = get_logger(__name__)


class Supervisor:
    """
    Creates supervisor which handles configuration, running and file output of the
    program.

    Attributes:
        args (argparse.Namespace):  Namespace containing the configuration parameters

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

        self.nested_groups: list[RunGroup] = parse_all_run_groups(args)
        self.base_run_dir = args.supervisor.get("base_run_dir")
        self.run_mode = args.supervisor.get("run_mode", "fresh")
        self.save_files_arg = args.supervisor.get("save_files", "all")

        if self.base_run_dir is None:
            if sys.stdout.isatty():
                self.base_run_dir = "base_run_dir"
                log.warning(
                    "No config for base_run_dir was found, "
                    "created base_run_dir folder to working directory"
                )
            else:
                raise ValueError(
                    "base_run_dir is not set in the provided configuration"
                )

        self.local_storage = args.supervisor.get("local_storage")

        if self.local_storage and not os.path.exists(self.local_storage):
            env = self.local_storage
            self.local_storage = os.environ.get(env)

            if not self.local_storage:
                log.warning(
                    f"Local storage environment variable {env} not set, ignoring..."
                )

        self.data_dir = os.path.join(self.base_run_dir, "data")
        self.previous_run_file = os.path.join(self.base_run_dir, "enchanted_run.yaml")
        self.previous_run_data = None

        if self.run_mode in ("resume", "extend"):
            self.previous_run_data = RunData.load(self.previous_run_file)
            if self.previous_run_data:
                if len(self.nested_groups) > self.previous_run_data.depth:
                    # Extending should generate budget worth of new samples so add
                    # already submitted amount to the current budget
                    if self.run_mode == "extend":
                        self.nested_groups[
                            self.previous_run_data.depth
                        ].sampler.budget += self.previous_run_data.submitted_samples

                    self.nested_groups[self.previous_run_data.depth].sampler.skip(
                        self.previous_run_data.batch_number + 1
                    )
            else:
                raise RuntimeError(
                    "Tried to continue from previous sampling but no "
                    " enchanted_run.yaml was found"
                )

            self.continue_with_base_run_dir(config_path)
        else:
            self.create_base_run_dir(self.base_run_dir, config_path)

    def start(self):
        """
        Main function of the supervisor. Starts the simulation process. Currently
        is the only function, that is accessed outside of supervisor.py.
        Gathers samples and paths, and gives them to executor. After all processes
        are finished, creates summary file.
        """

        log.info("Starting runs...")

        if self.local_storage:
            real_run_dir = self.local_storage
        else:
            real_run_dir = self.base_run_dir

        last_complete_dataset = pd.DataFrame()

        for depth, group in enumerate(self.nested_groups):
            batch_number = 0
            batch_dataset = pd.DataFrame()

            # Restore run state from previous data, if needed and in correct position of the loops
            if self.previous_run_data:
                if depth < self.previous_run_data.depth:
                    continue

                if depth == self.previous_run_data.depth:
                    batch_number = self.previous_run_data.batch_number + 1

                    batch_dataset = self.read_summary()
                    last_complete_dataset = self.read_summary(
                        "last_complete_enchanted_dataset"
                    )
                    group.sampler.register_future(last_complete_dataset)

            while group.sampler.has_budget:
                samples = group.sampler.get_next_samples()

                # Merge parameter names for nesting. On first depth run, expanded=samples
                expanded = []
                if not last_complete_dataset.empty:
                    for parent in last_complete_dataset.to_dict(orient="records"):
                        for sample in samples:
                            expanded.append({**parent, **sample})
                else:
                    expanded = samples

                # Run for each sequential runner/executor combination
                df_batch = pd.DataFrame()
                for i, (executor, runner) in enumerate(zip(group.executors, group.runners)):
                    run_dirs = [
                        os.path.join(
                            real_run_dir, "data", f"d{depth}_b{batch_number}_r{j}_s{i}"
                        )
                        for j in range(len(expanded))
                    ]
                    executor.execute(zip(run_dirs, expanded), runner, group.sampler)

                    # Wait processes of current batch to complete
                    self.wait_batch_dirs(run_dirs)

                    # Then the files in this batch should be saved into summary files
                    df_batch = self.load_batch_to_df(run_dirs)
                    expanded = df_batch.to_dict(orient="records")

                batch_dataset = pd.concat([batch_dataset, df_batch])
                if batch_number == 0:
                    self.write_summary(df_batch, write_mode='w')
                else:
                    self.write_summary(df_batch, write_mode='a')
                group.sampler.register_future(batch_dataset)

                run_data = RunData(
                    batch_number=batch_number,
                    depth=depth,
                    submitted_samples=group.sampler.submitted,
                )
                run_data.save(self.previous_run_file)

                self.fetch_from_local_storage()

                # Clean unwanted files
                self.delete_unwanted_files(self.save_files_arg, real_run_dir)

                batch_number += 1

            # Update data rows for next nesting level
            last_complete_dataset = batch_dataset.copy()

            # Create a summary file with last_complete_dataset for nesting
            if depth < len(self.nested_groups) - 1:
                self.write_summary(
                    dataset=last_complete_dataset,
                    filename="last_complete_enchanted_dataset",
                    write_mode="w"
                )

            self.fetch_from_local_storage()

        # Convert summary now after batches if configured
        self.finalize_summary()

        # Create HDF5 file by default
        if not hasattr(self.args, "storage") or self.args.storage.get("type") != "None":
            self.create_hdf5(last_complete_dataset)

        # Clean unwanted files
        self.delete_unwanted_files(self.save_files_arg, real_run_dir)

        # Clean run_dirs
        print("Shutting down scheduler and workers...")
        for group in self.nested_groups:
            for executor in group.executors:
                executor.clean()

    def write_summary(self, dataset: pd.DataFrame, filename: str = "enchanted_dataset", write_mode: str = "a"):
        """
        Writes a summary of dataset to base_run_dir/filename
        This functionality is used within the start function to
        enable seamless sampling. It appends each dataset on top of the
        previous dataset by default.

        Attributes:
            dataset (pd.DataFrame): batch to be written
            filename (str): base filename without extension for the written file
            write_mode (str): style of writing summary. appending ("a") is default, write ("w")
                is used for overwriting summary
        """

        csv_path = os.path.join(self.base_run_dir, f"{filename}.csv")
        write_header = write_mode != "a"
        dataset.to_csv(csv_path, mode=write_mode, header=write_header, index=False)



    def finalize_summary(self, filename: str = "enchanted_dataset"):
        """
        Finalizes summary after all the batches have been processed. Currently
        creates parquet summary file if configured in the configuration file.

        Attributes:
            filename (str): base filename without extension for summarized file
        """

        if (
            self.args.supervisor
            and self.args.supervisor.get("summary_datatype") == "parquet"
        ):
            csv_path = os.path.join(self.base_run_dir, f"{filename}.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                df.to_parquet(
                    os.path.join(self.base_run_dir, f"{filename}.parquet"),
                    engine="pyarrow",
                    index=False,
                )

    def read_summary(self, filename: str = "enchanted_dataset") -> pd.DataFrame:
        """
        Reads the summary written by write_summary.

        Attributes:
            filename (str): base filename without extension for the file to be read

        Returns:
            pd.Dataframe: dataset from the disk or an empty DataFrame if not found
        """

        csv_path = os.path.join(self.base_run_dir, f"{filename}.csv")
        if os.path.exists(csv_path):
            return pd.read_csv(csv_path)

        return pd.DataFrame()

    def continue_with_base_run_dir(self, config_path):
        """
        Deletes old unfinished bathes prompting the user if they want to keep them
        Creates a base_run_dir if one does not exist

        Attributes:
            config_path (str or None): Optional path for configuration file where
                configuration is fetched from
        """

        if not os.path.exists(self.base_run_dir):
            self.create_base_run_dir(self.base_run_dir, config_path)
            return

        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir, exist_ok=True)

        dirs = glob.glob(
            f"{self.data_dir}/"
            + f"d{self.previous_run_data.depth}_b{self.previous_run_data.batch_number + 1}*"
        )

        if not dirs:
            return

        for path in dirs:
            shutil.rmtree(path)

        # Create also data and logs folders if they don't exist

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
                        + "\nFolders have content. "
                        + "Do you want to delete data in existing folders? y/N "
                    )
                else:
                    print(
                        str(os.path.abspath(base_run_dir))
                        + "\nFolders have content. If you wish to continue, go delete them"
                    )
                    value = "n"

                if value.lower() in ("y", "yes"):
                    shutil.rmtree(base_run_dir)
                    print("base_run_dir was deleted")
                else:
                    print("No content was deleted. Enchanted surrogates is exited.")
                    sys.exit(1)

        # Create base run dir and data dir inside it
        os.makedirs(base_run_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)

        # Move config path to base_run_dir if config path is given
        if config_path is not None:
            os.makedirs(os.path.join(base_run_dir, "config"), exist_ok=True)
            config_dir = os.path.join(base_run_dir, "config")

            new_config_path = os.path.join(config_dir, os.path.basename(config_path))
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

    def all_processes_done(self, name_filter=None):
        """
        Monitors simulation processes and returns boolean describing state.
        Helper function for wait_all_processes.

        Args:
            filter (str or None): Optional filter used to limit checking to run directories
                containing this text. If None (default), all run directories are checked.
        Returns:
            True when all simulations are done. Helper function for
                wait_all_processes. Checks inside base_run_dir if folders inside it
                contain "enchanted_datapoint.csv" files.
            False If any runner has not yet created the csv file
        """

        for name in os.listdir(self.data_dir):
            if not name_filter or str(name_filter) in str(name):
                folder_path = os.path.join(self.data_dir, name)
                if os.path.isdir(folder_path):
                    datapoint_file = os.path.join(
                        folder_path, "enchanted_datapoint.csv"
                    )
                    if not os.path.isfile(datapoint_file):
                        return False

        return True

    def wait_all_processes(self, name_filter=None):
        """
        Waits in while loop until all simulations are done. Loop is broken
        when all_processes_done returns true. Checks condition once in
        second.

        Args:
            filter (str or None): Optional filter used to limit waiting to run directories
                containing this text. If None (default), all run directories are waited.
        """

        while True:
            if self.all_processes_done(name_filter):
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

        for name in os.listdir(self.data_dir):
            folder_path = os.path.join(self.data_dir, name)
            if os.path.isdir(folder_path):
                datapoint_file = os.path.join(folder_path, "enchanted_datapoint.csv")
                if os.path.isfile(datapoint_file):
                    enchanted_datapoint = pd.read_csv(datapoint_file)
                    enchanted_dataset = pd.concat(
                        [enchanted_dataset, enchanted_datapoint]
                    )
        return enchanted_dataset

    def batch_dirs_done(self, run_dirs: list[str]) -> bool:
        """
        Checks if enchanted_datapoint.csv files exist in the directories list given

        Attributes:
            run_dirs (list[str]): List of running directories within the batch

        Return:
            False if any of the datapoint files in the run_dirs is missing
            True if all datapoint files are found
        """
        for d in run_dirs:
            if not os.path.isfile(os.path.join(d, "enchanted_datapoint.csv")):
                return False
        return True

    def wait_batch_dirs(self, run_dirs: list[str]):
        """
        Waits for batch_dirs_done function to return True

        Attributes:
            run_dirs (list[str]): List of running directories within the batch
        """
        while not self.batch_dirs_done(run_dirs):
            sleep(1)

    def load_batch_to_df(self, run_dirs: list[str]) -> pd.DataFrame:
        """
        Creates pd.DataFrame combining enchanted_datapoint.csv files in given path list folders

        Attributes:
            run_dirs (list[str]): List of running directories within the batch

        Returns:
            pd.DataFrame containing batch datapoints combined
        """
        dfs = []
        for d in run_dirs:
            file = os.path.join(d, "enchanted_datapoint.csv")
            dfs.append(pd.read_csv(file))
        return pd.concat(dfs)

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

            for name in os.listdir(self.data_dir):
                folder_path = os.path.join(self.data_dir, name)
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
            run_groups = meta_group.create_group("run_groups")
            for i, run_group in enumerate(self.nested_groups):
                meta_run_group = run_groups.create_group(str(i))

                meta_run_group.attrs["sampler"] = str(
                    run_group.sampler.__class__.__name__
                )
                meta_run_group.attrs["executors"] = []
                meta_run_group.attrs["runners"] = []
                for j in range(len(run_group.executors)):
                    np.append(
                        meta_run_group.attrs["executors"],
                        str(run_group.executors[j].__class__.__name__),
                    )
                    np.append(
                        meta_run_group.attrs["runners"],
                        str(run_group.runners[j].get("type")),
                    )

    def delete_unwanted_files(self, argument: str, base_dir: str | None = None):
        """
        Deletes files according to command given.
        """
        default_list = ["enchanted_dataset.csv", "runs.h5"]
        if argument == "all":
            return

        if argument == "custom":
            saved_list = import_saved_files_list(self.args)
            allowed_files = set(default_list) | set(saved_list)
        elif argument == "none":
            allowed_files = set(default_list)
        else:
            return

        if not base_dir:
            base_dir = self.base_run_dir

        for root, dirs, files in os.walk(base_dir, topdown=False):
            # Remove files
            for file in files:
                if file not in allowed_files:
                    file_path = os.path.join(root, file)
                    os.remove(file_path)
            # Remove dirs
            for dir_ in dirs:
                dir_path = os.path.join(root, dir_)
                if not os.listdir(dir_path):
                    os.rmdir(dir_path)

    def fetch_from_local_storage(self):
        """
        Moves all files from local_storage to base_run_dir, if local_storage is defined.
        """
        if self.local_storage:
            for item in os.listdir(self.local_storage):
                src = os.path.join(self.local_storage, item)
                dst = os.path.join(self.base_run_dir, item)
                shutil.move(src, dst)
