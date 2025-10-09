# run.py
import yaml
import argparse
from datetime import datetime
from dask.distributed import print
from enchanted_surrogates.utils.precise_imports import import_executor
from enchanted_surrogates.utils.ascii_art import enchanted_wizard
from enchanted_surrogates.utils.hdf5 import convert_directory_to_hdf5

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
    config.executor_kwargs["config_filepath"] = config_path
    return config


def main(args: argparse.Namespace):
    """
    Main function for running the simulation workflow.

    Args:
        args (argparse.Namespace): Namespace containing the configuration parameters.
    """
    
    print(enchanted_wizard)
    
    #Incase sampler or runner is defined outside the executor, only works for non nested workflows
    if getattr(args, 'sampler_kwargs', None):
        args.executor_kwargs['sampler_kwargs'] = args.sampler_kwargs
    if getattr(args, 'runner_kwargs', None):
        args.executor_kwargs['sampler_kwargs'] = args.runner_kwargs
    
    executor_type = args.executor_kwargs.pop("type")
    executor = import_executor(type=executor_type, executor_kwargs=args.executor_kwargs)
    print("Starting runs...")
    executor.start_runs()
    print("Shutting down scheduler and workers...")
    executor.clean()
    if hasattr(args, 'general'):
        if args.general.get('hdf5_data'):
            convert_directory_to_hdf5(executor.base_run_dir)
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
