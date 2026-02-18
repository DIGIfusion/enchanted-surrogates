# run.py
import os
import warnings
import yaml
import argparse
from datetime import datetime
from dask.distributed import print
from enchanted_surrogates.utils.precise_imports import import_executor
from enchanted_surrogates.utils.ascii_art import enchanted_wizard
from enchanted_surrogates.utils.get_batch_dirs import get_batch_dirs
from enchanted_surrogates.utils.load_configuration import load_configuration
import shutil


def main(args: argparse.Namespace, config_path=None):
    """
    Main function for running the simulation workflow.

    Args:
        args (argparse.Namespace): Namespace containing the configuration parameters.
    """

    print(enchanted_wizard)

    print('GETTING EXECUTOR OBJECT')
    executor_type = args.executor.pop("type")
    executor = import_executor(type=executor_type, executor_config=args.executor)
    if not os.path.exists(executor.base_run_dir):
        print('MAKING BASE RUN DIR:', executor.base_run_dir)
        os.makedirs(executor.base_run_dir)
    
    if config_path is not None:
        print(f"Moving config file... from {config_path} to {os.path.join(executor.base_run_dir, os.path.basename(config_path))}")
        try:
            shutil.copy(config_path, os.path.join(executor.base_run_dir, os.path.basename(config_path)))
        except Exception as exe:
            warnings.warn(f"Copying the config file to the base run dir failed.\n \
                            Try using the full path to the config file.\n \
                            Here is the exception raised:\n {exe}")
    
    print("Starting runs...")
    executor.start_runs()
    print("Shutting down scheduler and workers...")
    executor.clean()
    
    if hasattr(args, 'general'):
        
        # Exact filenames to skip
        skip_delete = {
            "enchanted_dataset_fail.csv",
            "enchanted_dataset.csv",
            "ENCHANTED.FINISHED",
            "batch_info.csv",
        }

        # Glob-style patterns to skip
        skip_patterns = [
            "*.yaml",
            "*worker_out*",
        ]
        
        if args.general.get('pack_data_squashfs', False):
            print('PACKING ALL RUN DATA INTO A SquashFS FILE')
            from enchanted_surrogates.utils.convert_directory_to_squashfs import convert_directory_to_squashfs
            convert_directory_to_squashfs(
                args.executor['base_run_dir'],
                squashfs_name="archive.sqsh",
                skip_delete=skip_delete,
                skip_patterns=skip_patterns,
            )

        if args.general.get('pack_data_hdf5', False):
            print('PACKING ALL RUN DATA INTO A HDF5 FILE')
            # the import is placed here so that users don't need h5py installed in their enviroment to use run.py
            from enchanted_surrogates.utils.hdf5 import convert_directory_to_hdf5
            # reduce num files with hdf5
            convert_directory_to_hdf5(args.executor['base_run_dir'], skip_delete=skip_delete, skip_patterns=skip_patterns)
    return


if __name__ == "__main__":
    print(f'{datetime.now()} - Starting Enchanted surrogates.')
    parser = argparse.ArgumentParser(description="Runner")
    parser.add_argument(
        "-cf",
        "--config_file",
        type=str,
        default="base",
        help="Path to the config file",
    )
    config_args = parser.parse_args()
    args = load_configuration(config_args.config_file)
    main(args, config_path=config_args.config_file)
    print(f'{datetime.now()} - Enchanted surrogates finished.')
