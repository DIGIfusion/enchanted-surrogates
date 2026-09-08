import argparse
from enchanted_surrogates.supervisor.supervisor import Supervisor, LOG_DIR
from enchanted_surrogates.utils.ascii_art import enchanted_wizard_version_7
from enchanted_surrogates.utils.logger import get_logger, setup_logger
from enchanted_surrogates.utils.config_helpers import load_configuration

log = get_logger(__name__)

"""
Command-line entry point for running Enchanted Surrogates.

This module loads a YAML configuration file, constructs the corresponding
execution namespace, and initializes the Supervisor responsible for
managing sampling, execution, and result handling.
"""


def main(arguments: argparse.Namespace, config_path=None):
    """
    Main function for running the simulation workflow.

    Args:
        args (argparse.Namespace): Namespace containing the configuration parameters.
        config_path (str or None): Optional path for configuration file where
            configuration is fetched from.
    """
    supervisor = Supervisor(arguments, config_path=config_path)

    setup_logger(supervisor.base_run_dir, getattr(arguments, "logging", "INFO"), log_dir=LOG_DIR)
    log.info("\n" + enchanted_wizard_version_7)
    log.info("Enchanted surrogates is starting.")
    log.info(f"Base run directory: {supervisor.base_run_dir}")

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
