# run.py
import yaml
import samplers

import executors
import argparse


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
    sampler = getattr(samplers, args.sampler.pop("type"))(**args.sampler)
    executor = getattr(executors, args.executor.pop("type"))(
        sampler=sampler, runner_args=args.runner, **args.executor
    )

    executor.start_runs()
    executor.clean()


if __name__ == "__main__":
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
