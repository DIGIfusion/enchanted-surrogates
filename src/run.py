"""
Command-line entry point for running Enchanted Surrogates.

This module loads a YAML configuration file, constructs the corresponding
execution namespace, and initializes the Supervisor responsible for
managing sampling, execution, and result handling.
"""

import yaml
import argparse
from datetime import datetime
from dask.distributed import print
from enchanted_surrogates.utils.ascii_art import enchanted_wizard
from enchanted_surrogates.supervisor.supervisor import Supervisor


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


def main(args: argparse.Namespace, config_path=None):
    """
    Main function for running the simulation workflow.

    Args:
        args (argparse.Namespace): Namespace containing the configuration parameters.
        config_path (str or None): Optional path for configuration file where 
            configuration is fetched from.
    """

    print(enchanted_wizard)
    supervisor = Supervisor(args, config_path=config_path)
    supervisor.start()
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
