# run.py
import yaml
import argparse
from datetime import datetime
from dask.distributed import print
from enchanted_surrogates.utils.precise_imports import import_executor


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
    return config


def main(args: argparse.Namespace):
    """
    Main function for running the simulation workflow.

    Args:
        args (argparse.Namespace): Namespace containing the configuration parameters.
    """

    args.executor['runner_args'] = getattr(args, 'runner', None)
    executor_type = args.executor.pop("type")
    executor = import_executor(type=executor_type, executor_kwargs=args.executor)
    print("Starting runs...")
    executor.start_runs()
    print("Shutting down scheduler and workers...")
    executor.clean()
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
