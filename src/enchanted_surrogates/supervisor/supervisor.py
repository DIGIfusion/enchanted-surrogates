import os
import warnings
import shutil
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
        executor_type = args.executor.pop("type")
        self.executor = import_executor(
            type=executor_type, 
            executor_config=args.executor)
        self.sampler = import_sampler(
            type=self.sampler_config.pop("type"), 
            sampler_config=self.sampler_config)        
        self.base_run_dir = args.executor.base_run_dir

        # Create base run dir if it does not exist
        if not os.path.exists(self.base_run_dir):
            os.makedirs(self.base_run_dir)

        # Move config path to base_run_dir if config path is given
        if config_path is not None:
            new_config_path = os.path.join(self.base_run_dir, os.path.basename(config_path))
            print(f"Moving config file... from {config_path} to {new_config_path}")
            try:
                shutil.copy(config_path, new_config_path)
            except Exception as exe:
                warnings.warn(
                    f"Copying the config file to the base run dir failed.\n \
                    Try using the full path to the config file.\n \
                    Here is the exception raised:\n {exe}"
                )

        
    def start(self):
        enchanted_dataset = pd.DataFrame()
        sampler = self.init_sampler()
        print("Starting runs...")

        while sampler.has_budget:
            # Get samples
            samples: list[dict] = sampler.get_next_samples()
            # Create folders with order number as name
            folders = [os.path.join(self.base_run_dir, str(i)) for i in range(len(samples))]
            # Call executor with folder path and samples
            self.executor.start_runs(zip (folders, samples))
            # Collect files in folders
            for folder in folders:
                filename = os.path.join(folder, "enchanted_datapoint.csv")
                enchanted_datapoint = pd.read_csv(filename)
                enchanted_dataset.append(enchanted_datapoint)

        # Create summary csv TODO or parquet file
        if self.args.supervisor.summary_datatype == "parquet":
            enchanted_dataset.to_parquet(
                os.path.join(self.base_run_dir, "enchanted_dataset.parquet"),
                engine="pyarrow",
                index=True
            )
        else:
            enchanted_dataset.to_csv(os.path.join(self.base_run_dir, "enchanted_dataset.csv"))
            
        # Clean folders
        print("Shutting down scheduler and workers...")
        self.executor.clean()

        # TODO Create HDF5 file



        
    # Not sure if necessary to have??
    def create_base_run_dir(self, base_run_dir, config_filepath):
        print(
            f"Making directory of simulations at: {base_run_dir}.",
            "Copying {config_filepath} to CONFIG.yaml."
        )

        os.makedirs(base_run_dir, exist_ok=True)
        shutil.copyfile(config_filepath, os.path.join(base_run_dir, "CONFIG.yaml"))

