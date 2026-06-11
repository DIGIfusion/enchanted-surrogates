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
from datetime import datetime
import numpy as np
import pandas as pd
from enchanted_surrogates.utils.logger import get_logger
from enchanted_surrogates.supervisor.run_data import RunData
from enchanted_surrogates.supervisor.nested_imports import (
    RunGroup,
    parse_all_run_groups,
    import_saved_files_list,
)
from enchanted_surrogates.utils.ascii_loading_bar import ascii_loading_bar
from enchanted_surrogates.utils.time_format import time_format
from enchanted_surrogates.utils.ascii_art import enchanted_wizard_version_3
import time
log = get_logger(__name__)

LOG_DIR = "logs"


class Supervisor:
    """
    Creates supervisor which handles configuration, running and file output of the
    program.

    Attributes:
        args (argparse.Namespace):  Namespace containing the configuration parameters

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

        # start the progress for each runner at 0
        self.runner_progress = {}
        self.group_budget = {}
        budget = 1
        for i, group in enumerate(self.nested_groups):
            budget = group.sampler.budget * budget
            self.group_budget[f'G{i}'] = budget
            self.runner_progress[f'G{i}'] = {}
            for runner in group.runners:
                self.runner_progress[f'G{i}'][runner["__runner_name"]] = {'submitted':0, 'completed':0,'num_successes':0}
        
        self.base_run_dir = args.supervisor.get("base_run_dir")
        self.run_mode = args.supervisor.get("run_mode", "fresh")
        self.save_files_arg = args.supervisor.get("save_files", "all")
        self.log_failures = args.supervisor.get("log_failures", False)
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
        self.current_progress_info_file = os.path.join(self.base_run_dir, LOG_DIR, 'current_progress.txt')
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

    # def reset_progress(self):
    #     for group in self.nested_groups:
    #         for runner in group.runners:
    #             self.runner_progress[f'G{i}'][runner["__runner_name"]] = {'submitted':0, 'completed':0,'num_successes':0}

    def start(self):
        """
        Main function of the supervisor. Starts the simulation process. Currently
        is the only function, that is accessed outside of supervisor.py.
        Gathers samples and paths, and gives them to executor. After all processes
        are finished, creates summary file.
        """
        start_runs_time = time.time()
        log.info("Starting runs...")
        if self.local_storage:
            real_run_dir = self.local_storage
        else:
            real_run_dir = self.base_run_dir
                
        last_complete_dataset = pd.DataFrame()
        
        log.info(f"\n\nProgress report will be saved to: {self.current_progress_info_file}\n\n")
        
        for nested_depth, group in enumerate(self.nested_groups):
            # self.reset_progress()
            log.debug(f'At depth {depth} with sampler {group.sampler} and executors {group.executors} and runners {group.runners}')
            batch_number = 0
            batch_dataset = pd.DataFrame()
            # Restore run state from previous data, if needed and in correct position of the loops
            if self.previous_run_data:
                if nested_depth < self.previous_run_data.depth:
                    continue

                if nested_depth == self.previous_run_data.depth:
                    batch_number = self.previous_run_data.batch_number + 1

                    batch_dataset = self.read_summary()
                    last_complete_dataset = self.read_summary(
                        "last_complete_enchanted_dataset"
                    )
                    group.sampler.register_future(last_complete_dataset)

            # initalise the current progress file
            group_start_time = time.time()
            
            log.info(f"Starting nested group {nested_depth} with sampler {group.sampler.__class__.__name__}")
            self.write_current_progress_string(current_runner_name="N/A", nested_depth=nested_depth, sequential_depth=0, batch_number=batch_number, group_start_time=group_start_time)
            
            while group.sampler.has_budget:
                samples = group.sampler.get_next_samples()
                
                if samples is None:
                    log.debug('Sampler returned None.')
                    break
                
                if group.sampler.submitted > group.sampler.budget:
                    log.debug('Budget Exceeded')
                    break

                # Merge parameter names for nesting. On first depth run, expanded=samples
                expanded = self.get_cartesian_product(samples, last_complete_dataset)

                # Run for each sequential runner/executor combination
                df_batch = pd.DataFrame()
                for sequential_depth, (executor, runner) in enumerate(
                    zip(group.executors, group.runners)
                ):
                    run_dirs = [
                        os.path.join(
                            real_run_dir, "data", f"dn{nested_depth}_ds{sequential_depth}_b{batch_number}_s{j}"
                        )
                        for j in range(len(expanded))
                    ]
                    executor.execute(list(zip(run_dirs, expanded)), runner)
                    self.update_runner_progress(f'G{nested_depth}', runner, submitted=len(expanded))
                    self.write_current_progress_string(runner["__runner_name"], nested_depth, sequential_depth, batch_number, group_start_time)
                    # monitor runs for failures and update progress file
                    self.monitor_runs(f'G{nested_depth}', runner, run_dirs, nested_depth = nested_depth, sequential_depth = sequential_depth, batch_number = batch_number, group_start_time=group_start_time)
                    # Wait processes of current batch to complete
                    self.wait_batch_dirs(run_dirs)

                    # Load runner output of this batch, used as input for next sequential run
                    df_batch = self.load_batch_to_df(run_dirs)
                    expanded = df_batch.to_dict(orient="records")

                # Save batch results into summary files
                batch_dataset = pd.concat([batch_dataset, df_batch])
                if batch_number == 0:
                    self.write_summary(df_batch, write_mode="w")
                else:
                    self.write_summary(df_batch, write_mode="a")
                
                log.debug('Registering data with future...')
                group.sampler.register_future(df_batch)

                log.debug('Saving run data...')
                run_data = RunData(
                    batch_number=batch_number,
                    depth=nested_depth,
                    submitted_samples=group.sampler.submitted,
                )
                run_data.save(self.previous_run_file)

                # Appends hdf5 file with new datapoints
                # The final dataset is written later
                if not hasattr(self.args, "storage") or self.args.storage.get("type") != "None":
                    log.debug('Appending to hdf5 file...')
                    self.hdf5_append_datapoints(run_dirs)

                self.fetch_from_local_storage()
                
                # Clean unwanted files
                self.delete_unwanted_files(self.save_files_arg, self.data_dir)

                batch_number += 1
            
            log.debug(f"Completed batch {batch_number} at depth {depth}")

            # Update data rows for next nesting level
            last_complete_dataset = batch_dataset.copy()

            # Create a summary file with last_complete_dataset for nesting
            if nested_depth < len(self.nested_groups) - 1:
                self.write_summary(
                    dataset=last_complete_dataset,
                    filename="last_complete_enchanted_dataset",
                    write_mode="w",
                )

            self.fetch_from_local_storage()


        # Convert summary now after batches if configured
        self.finalize_summary()

        # Create HDF5 file by default
        if not hasattr(self.args, "storage") or self.args.storage.get("type") != "None":
            self.hdf5_write_aggregate_dataset_and_metadata(last_complete_dataset)

        # Clean unwanted files
        self.delete_unwanted_files(self.save_files_arg, self.data_dir)
        
        end_runs_time = time.time()
        log.info(f"All Runs completed in {time_format(int(end_runs_time - start_runs_time))} (days - hours:minutes:seconds)")
        # Clean run_dirs
        log.info("Shutting down scheduler and workers...")
        for group in self.nested_groups:
            for executor in group.executors:
                executor.clean()

    def update_runner_progress(self, group_name: str, runner_config: dict, submitted: int = 0, completed: int = 0, num_successes: int = 0):
        self.runner_progress[group_name][runner_config["__runner_name"]]['completed'] += completed
        self.runner_progress[group_name][runner_config["__runner_name"]]['submitted'] += submitted
        self.runner_progress[group_name][runner_config["__runner_name"]]['num_successes'] += num_successes
    
    def write_summary(
        self,
        dataset: pd.DataFrame,
        filename: str = "enchanted_dataset",
        write_mode: str = "a",
    ):
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
        log.debug(f'Writing summary to {filename} with write mode {write_mode}...')
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
        log.debug('Finalizing summary...')
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
    
    def get_cartesian_product(self, samples: list[dict], last_dataset: pd.DataFrame) -> list[dict]:
        """
        Creates cartesian product of the new samples and the previous dataset.
        Used for nested sampling.

        Arguments:
            samples (list[dict]): Sample batch from get_next_samples
            last_dataset (pd.DataFrame): The complete dataset (summary file) from
                previous nesting level.

        Returns:
            out (list[dict]): Cartesian product samples x last_dataset. If last_dataset is empty,
                only the unaltered samples are returned.
        """
        if last_dataset.empty:
            return samples

        expanded = []
        for parent in last_dataset.to_dict(orient="records"):
            for sample in samples:
                expanded.append({**parent, **sample})
        return expanded

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
        log.debug('Waiting for runs to finnish...')
        """
        Waits for batch_dirs_done function to return True

        Attributes:
            run_dirs (list[str]): List of running directories within the batch
        """
        while not self.batch_dirs_done(run_dirs):
            sleep(1)
    
    def monitor_runs(self, group_name, runner_config, run_dirs: list[str], nested_depth, sequential_depth, batch_number, group_start_time):
        log.debug('Monitoring runs...')
        """
        Keeps checking all the run_dirs for failures and logs the failures it finds
        
        Attributes:
            run_dirs (list[str]): List of running directories to monitor
        """
        
        run_dirs = set(run_dirs)   # if it isn't already a set
        while run_dirs:
            for run_dir in list(run_dirs):   # iterate over a snapshot
                result = self.load_run_result(run_dir)
                if result is not None:
                    # remove so it is not rechecked and we are closer to while loop stopping
                    run_dirs.remove(run_dir)
                    self.delete_unwanted_files(self.save_files_arg, run_dir, extra_keep_files=['enchanted_datapoint.csv'])
                    self.update_runner_progress(group_name, runner_config,completed=1)
                    if result['success']:
                        self.update_runner_progress(group_name, runner_config,num_successes=1)
                    else:
                        if self.log_failures:
                            details = "\n".join(f"  {k}: {v}" for k, v in result.items())
                            log_message = f"""\n\n
                            
==   FAILURE  ========================================
Run directory: {run_dir}

Result:
{details}
======================================================

                            \n""".strip()

                            log.error(log_message)                            
                    self.write_current_progress_string(runner_config["__runner_name"], nested_depth, sequential_depth, batch_number, group_start_time)
                sleep(0.1)
        
    def write_current_progress_string(self, current_runner_name, nested_depth, sequential_depth, batch_number, group_start_time):
        log.debug('Writing progress string')
        def format_runner_progress(runner_name, stats, budget):
            submitted = stats.get("submitted", 0)
            completed = stats.get("completed", 0)
            successes = stats.get("num_successes", 0)
            failures = completed - successes
            success_rate = (successes * 100 / completed) if completed else 0

            bar_completed = ascii_loading_bar(budget, completed)
            bar_submitted = ascii_loading_bar(submitted if submitted else 1, completed)

            return f"""
--------------------   RUNNER: {runner_name}   --------------------

  STATUS
--------------------------------------------------------
Submitted:          {submitted}
Completed:          {completed}
Successes:          {successes}
Failures:           {failures}
Success Rate:       {success_rate:5.1f}%

  COMPLETED vs SUBMITTED
--------------------------------------------------------
{completed} / {submitted if submitted else 0}
{bar_submitted}

  COMPLETED vs BUDGET
--------------------------------------------------------
{completed} / {budget}
{bar_completed}



""".rstrip()
        
        def format_all_runners_progress(runner_progress):
            blocks = []
            for group, runners in runner_progress.items():
                blocks.append(f'====================   GROUP: {group}   ====================')
                for runner_name, stats in runners.items():
                    blocks.append(format_runner_progress(runner_name, stats, self.group_budget[group]))
            
            return "\n".join(blocks)

        runner_progress_string = format_all_runners_progress(self.runner_progress) 
        progress_string=f"""

{enchanted_wizard_version_3}

===   PROGRESS REPORT   ====================================

Group Start Time:     {datetime.fromtimestamp(group_start_time).strftime("%Y-%m-%d %H:%M:%S")}
Last Update:          {datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d %H:%M:%S")}

Active Runner:        {current_runner_name}
Nested Depth:         {nested_depth}
Sequential Depth:     {sequential_depth}
Current Batch:        {batch_number}

 
{runner_progress_string}

==============================================================

"""

        
        os.makedirs(os.path.dirname(self.current_progress_info_file), exist_ok=True)
        with open(self.current_progress_info_file, 'w') as file:
            file.write(progress_string)
        
        return progress_string
        
        
    def load_run_result(self, run_dir):
        result_path = os.path.join(run_dir, 'enchanted_datapoint.csv')

        if not os.path.exists(result_path):
            return None

        # Protect against empty CSV
        if os.path.getsize(result_path) == 0:
            return None

        try:
            df = pd.read_csv(result_path)
            if df.empty:
                return None
            return df.iloc[0].to_dict()

        except pd.errors.EmptyDataError:
            return None

    def load_batch_to_df(self, run_dirs: list[str]) -> pd.DataFrame:
        """
        Creates pd.DataFrame combining enchanted_datapoint.csv files in given path list folders

        Attributes:
            run_dirs (list[str]): List of running directories within the batch

        Returns:
            pd.DataFrame containing batch datapoints combined
        """
        log.debug('Loading batch data to df...')
        dfs = []
        for d in run_dirs:
            file = os.path.join(d, "enchanted_datapoint.csv")
            dfs.append(pd.read_csv(file))
        return pd.concat(dfs)
    
    def hdf5_append_datapoints(self, new_dirs: list[str]):
        """
        Appends new datapoints to the hdf5 storage file. This allows removing
        the intermediate files and directories after each batch run.

        Args:
            new_dirs (list[str]): List of new datapoint directories created
                during a single batch

        """
        h5_path = os.path.join(self.base_run_dir, "runs.h5")

        with h5py.File(h5_path, "a") as f:
            runs_group = f.require_group("data/runs")

            for dir in new_dirs:
                csv_path = os.path.join(dir, "enchanted_datapoint.csv")

                if not os.path.isfile(csv_path):
                    continue

                df = pd.read_csv(csv_path)

                # Get just the final path component of dir 
                dir_name = os.path.basename(dir)

                run_group = runs_group.require_group(dir_name)

                # Select only numeric values
                numeric_df = df.select_dtypes(include=[np.number])

                run_group.create_dataset("values", data=numeric_df.to_numpy())

                run_group.create_dataset(
                    "columns",
                    data=np.array(numeric_df.columns, dtype=h5py.string_dtype("utf-8")),
                )

    def hdf5_write_aggregate_dataset_and_metadata(self, enchanted_dataset: pd.DataFrame):
        """
        Writes the completed dataset and run metadata to the hdf5 storage file.
        Dataset has only numeric values, column headers are saved separately 
        in the same location. Metadata includes types for sampler, executor and
        runner.

        Args:
            enchanted_dataset (pd.DataFrame): Dataframe containing all run
                results

        """
        h5_path = os.path.join(self.base_run_dir, "runs.h5")

        with h5py.File(h5_path, "a") as f:
            # Aggregated dataset
            # Remove old dataset if continuing a previous run
            if f.get("data/aggregated"):
                del f["data/aggregated"]

            agg_group = f.require_group("data/aggregated")

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

            # Metadata
            # Remove old metadata if continuing a previous run
            if f.get("metadata"):
                del f["metadata"]

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

    def delete_unwanted_files(self, argument: str, base_dir: str | None = None, extra_keep_files=[]):
        """
        Deletes files according to command given.
        """
        log.debug('Deleting unwanted files...')
        default_list = ["enchanted_dataset.csv", "runs.h5"]
        if argument == "all":
            return

        if argument == "custom":
            saved_list = import_saved_files_list(self.args)
            allowed_files = set(default_list) | set(saved_list) | set(extra_keep_files)
        elif argument == "none":
            allowed_files = set(default_list) | set(extra_keep_files)
        else:
            return

        if not base_dir:
            base_dir = self.base_run_dir

        for root, dirs, files in os.walk(base_dir, topdown=False):
            # Remove files
            for file in files:
                # Do nothing for log files
                parent = os.path.basename(os.path.dirname(os.path.join(root, file)))
                if parent == LOG_DIR:
                    continue

                if file not in allowed_files:
                    file_path = os.path.join(root, file)
                    os.remove(file_path)
            # Remove dirs
            for dir_ in dirs:
                # Do nothing for log dir
                if dir_ == LOG_DIR:
                    continue
                dir_path = os.path.join(root, dir_)
                if not os.listdir(dir_path):
                    os.rmdir(dir_path)

    def fetch_from_local_storage(self):
        """
        Moves all files from local_storage to base_run_dir, if local_storage is defined.
        """
        log.debug('Fetching from local storage...')
        if self.local_storage:
            for item in os.listdir(self.local_storage):
                src = os.path.join(self.local_storage, item)
                dst = os.path.join(self.base_run_dir, item)
                shutil.move(src, dst)
