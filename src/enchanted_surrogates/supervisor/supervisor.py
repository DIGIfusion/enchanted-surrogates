import os
import warnings
import shutil
from time import sleep
import pandas as pd
from enchanted_surrogates.utils.precise_imports import import_sampler, import_executor

# Tasks
# 1. Execution of program is going through this module
# 2. Supervisor creates samples with configured sampler
# 3. Creates folders for runners
# 4. Supervisor gives samples to executor
# 5. Collects all the files created to folders
# 6. Forms HDF5 dataset and enchanted_dataset.csv or parquet summary file to base_dir


class Supervisor():

    def __init__(self, args, config_path=None):
        self.args = args
        self.executor = import_executor(
            type=args.executor.pop("type"),
            executor_config=args.executor)
        self.sampler = import_sampler(
            type=args.sampler_config.pop("type"),
            sampler_config=args.sampler_config)
        self.base_run_dir = args.executor.base_run_dir  # TODO modify config parameter to be under supervisor

        self.create_base_run_dir(self.base_run_dir,config_path)


    def start(self):
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

            # Call executor with folder path and samples
            self.executor.start_runs(zip (run_dirs, samples))
            # Collect files in folders


            for run_dir in run_dirs:
                filename = os.path.join(run_dir, "enchanted_datapoint.csv")


        self.wait_all_processes()

        enchanted_dataset = self.create_dataset()

        # Create summary csv or parquet file
        if self.args.supervisor.summary_datatype == "parquet":
            enchanted_dataset.to_parquet(
                os.path.join(self.base_run_dir, "enchanted_dataset.parquet"),
                engine="pyarrow",
                index=True
            )
        else:
            enchanted_dataset.to_csv(os.path.join(self.base_run_dir, "enchanted_dataset.csv"))

        # Clean run_dirs
        print("Shutting down scheduler and workers...")
        self.executor.clean()

        # TODO Create HDF5 file


    def create_base_run_dir(self, base_run_dir, config_path):
        # Create base run dir if it does not exist
        if not os.path.exists(base_run_dir):
            os.makedirs(base_run_dir)

        # Move config path to base_run_dir if config path is given
        if config_path is not None:
            new_config_path = os.path.join(base_run_dir, os.path.basename(config_path))
            print(f"Moving config file... from {config_path} to {new_config_path}")
            try:
                shutil.copy(config_path, new_config_path)
            except Exception as exe:
                warnings.warn(
                    f"Copying the config file to the base run dir failed.\n \
                    Try using the full path to the config file.\n \
                    Here is the exception raised:\n {exe}"
                )


    def all_processes_done(self):
        # Check all the run_dirs that they have "enchanted_datapoint.csv"
        for name in os.listdir(self.base_run_dir):
            folder_path = os.path.join(self.base_run_dir,name)
            if os.path.isdir(folder_path):
                datapoint_file = os.path.join(folder_path, "enchanted_datapoint.csv")
                if not os.path.isfile(datapoint_file):
                    return False

        return True

    def wait_all_processes(self):
        while True:
            if self.all_processes_done():
                break
            sleep(1)

    def create_dataset(self):

        enchanted_dataset = pd.DataFrame()
        for name in os.listdir(self.base_run_dir):
            folder_path = os.path.join(self.base_run_dir, name)
            if os.path.isdir(folder_path):
                datapoint_file = os.path.join(folder_path, "enchanted_datapoint.csv")
                if os.path.isfile(datapoint_file):
                    enchanted_datapoint = pd.read_csv(datapoint_file)
                    enchanted_dataset.append(enchanted_datapoint)
        return enchanted_dataset
