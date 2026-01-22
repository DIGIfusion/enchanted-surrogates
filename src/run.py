# run.py
import os
import threading
import yaml
import argparse
from datetime import datetime
from enchanted_surrogates.utils.precise_imports import import_executor
from enchanted_surrogates.utils.ascii_art import enchanted_wizard
from enchanted_surrogates.utils.logger import get_logger, setup_logging
import shutil

log = get_logger(__name__)

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
    log.debug(config)
    return config




def main(args: argparse.Namespace, config_path=None):
    """
    Main function for running the simulation workflow.

    Args:
        args (argparse.Namespace): Namespace containing the configuration parameters.
    """
    # Create the base run directory
    if not os.path.exists(args.executor["base_run_dir"]):
        os.makedirs(args.executor["base_run_dir"])

    # Setup logging
    log_dir = os.path.join(args.executor['base_run_dir'], 'logs')
    setup_logging(args.logging, log_dir, "main.log")
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

    # Initialize executor
    executor_type = args.executor.pop("type")

    # Add log level and log dir to executor config
    args.executor['log_level'] = args.logging
    args.executor['log_dir'] = log_dir

    executor = import_executor(type=executor_type, executor_config=args.executor)

    if not os.path.exists(executor.base_run_dir):
        os.makedirs(executor.base_run_dir)

    log.info("Starting runs...")
    executor.start_runs()
    executor.clean()
    log.info('Enchanted surrogates has finished.')

    return

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
