# run.py
import yaml
import argparse
from datetime import datetime
from dask.distributed import print
from enchanted_surrogates.utils.precise_imports import import_executor
from enchanted_surrogates.utils.ascii_art import enchanted_wizard
from enchanted_surrogates.utils.get_batch_dirs import get_batch_dirs

def load_configuration(config_path: str) -> argparse.Namespace:
    """
    Loads configuration from a YAML file.

    Args:
        config_path (str): Path to the configuration YAML file.

    Returns:
        argparse.Namespace: Namespace containing the configuration parameters.
    """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    config = argparse.Namespace(**config)
    config.executor["config_filepath"] = config_path

    # In case sampler or runner is defined outside the executor.
    # This only works for non nested workflows.
    if 'sampler_config' not in config.executor:
        if getattr(config, 'sampler', None):
            config.executor['sampler_config'] = config.sampler
        elif 'sampler' in config.executor:
            config.executor['sampler_config'] = config.executor.pop('sampler')
    if 'runner_config' not in config.executor:
        if getattr(config, 'runner', None):
            config.executor['runner_config'] = config.runner
        elif 'runner' in config.executor:
            config.executor['runner_config'] = config.executor.pop('runner')
    print(config)
    return config


def main(args: argparse.Namespace):
    """
    Main function for running the simulation workflow.

    Args:
        args (argparse.Namespace): Namespace containing the configuration parameters.
    """

    print(enchanted_wizard)

    executor_type = args.executor.pop("type")
    executor = import_executor(type=executor_type, executor_config=args.executor)
    print("Starting runs...")
    executor.start_runs()
    print("Shutting down scheduler and workers...")
    executor.clean()
    
    if hasattr(args, 'general'):
        if args.general.get('pack_data_hdf5', True):
            # the import is placed here so that users don't need h5py installed in their enviroment to use run.py
            from enchanted_surrogates.utils.hdf5 import convert_directory_to_hdf5
            # reduce num files with hdf5
            batch_dirs = get_batch_dirs(args.executor['base_run_dir'])
            for batch_dir in batch_dirs:
                convert_directory_to_hdf5(batch_dir, skip_delete=['enchanted_dataset.csv', 'batch_info.csv'])
    return


if __name__ == "__main__":
    print(f'{datetime.now()} - Starting Enchanted surrogates.')
    parser = argparse.ArgumentParser(description="Runner")
    parser.add_argument(
        "-cf",
        "--config_file",
        type=str,
        default="base",
        help="name of configuration file stored in /configs!",
    )
    config_args = parser.parse_args()
    args = load_configuration(config_args.config_file)
    main(args)
    print(f'{datetime.now()} - Enchanted surrogates finished.')
