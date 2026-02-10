# run.py
import os
import sys
import yaml
import argparse
from datetime import datetime
from enchanted_surrogates.utils.precise_imports import import_executor
from enchanted_surrogates.utils.ascii_art import enchanted_wizard
from enchanted_surrogates.utils.logger import get_logger, setup_logging, LoggerConfig
import logging
import shutil

log = get_logger(__name__)

"""
Command-line entry point for running Enchanted Surrogates.

This module loads a YAML configuration file, constructs the corresponding
execution namespace, and initializes the Supervisor responsible for
managing sampling, execution, and result handling.
"""

def load_configuration(config_path: str) -> argparse.Namespace:
    """
    Loads configuration from a YAML file.

    Args:
        config_path (str): Path to the configuration YAML file.

    Returns:
        argparse.Namespace: Namespace containing the configuration parameters.
    """
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    config = argparse.Namespace(**config)
    config.supervisor["config_filepath"] = config_path

    log.debug(config)
    return config

def main(arguments: argparse.Namespace, config_path=None):
    """
    Main function for running the simulation workflow.

    Args:
        args (argparse.Namespace): Namespace containing the configuration parameters.
        config_path (str or None): Optional path for configuration file where 
            configuration is fetched from.
    """
    # Create the base run directory
    if not os.path.exists(args.executor["base_run_dir"]):
        os.makedirs(args.executor["base_run_dir"])

    # Setup logging
    log_dir = os.path.join(args.executor['base_run_dir'], 'logs')
    log_file = os.path.join(log_dir, "main.log")
    log_level = args.logging

    # Store to logger config
    config = LoggerConfig(log_level=log_level, log_dir=log_dir)

    # Create log dir
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)

    setup_logging(config, logging.StreamHandler(stream=sys.stdout), logging.FileHandler(filename=log_file))

    log.info('Enchanted surrogates is starting.')
    log.info(f'Base run directory: {args.executor["base_run_dir"]}')

    # Copying the config file to the base run directory
    if config_path is not None:
        log.debug(f"Copying config file from {config_path} to {os.path.join(args.executor['base_run_dir'], os.path.basename(config_path))}")
        try:
            shutil.copy(config_path, os.path.join(args.executor["base_run_dir"], os.path.basename(config_path)))
        except Exception as exe:
            log.error(f"Copying the config file to the base run dir failed.\n \
                        Try using the full path to the config file.\n \
                        Exception raised:\n {exe}")
    
    print(enchanted_wizard)
    supervisor = Supervisor(arguments, config_path=config_path)
    supervisor.start()


if __name__ == "__main__":
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
