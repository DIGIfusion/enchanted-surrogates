# run.py
import os
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

    print(enchanted_wizard)

    executor_type = args.executor.pop("type")
    executor = import_executor(type=executor_type, executor_config=args.executor)
    if not os.path.exists(executor.base_run_dir):
        os.makedirs(executor.base_run_dir)

    setup_logging(args.logging, executor.base_run_dir)
    
    if config_path is not None:
        log.info(f"Moving config file... from {config_path} to {os.path.join(executor.base_run_dir, os.path.basename(config_path))}")
        try:
            shutil.copy(config_path, os.path.join(executor.base_run_dir, os.path.basename(config_path)))
        except Exception as exe:
            log.error(f"Copying the config file to the base run dir failed.\n \
                        Try using the full path to the config file.\n \
                        Here is the exception raised:\n {exe}")
    
    log.info("Starting runs...")
    executor.start_runs()
    log.info("Shutting down scheduler and workers...")
    executor.clean()
    return

if __name__ == "__main__":
    log.info(f'{datetime.now()} - Starting Enchanted surrogates.')
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
    log.info(f'{datetime.now()} - Enchanted surrogates finished.')
