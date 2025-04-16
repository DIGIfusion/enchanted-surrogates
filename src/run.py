# run.py
from dask.distributed import print
print('PERFORMING IMPORTS')
import yaml
import os
import argparse
import importlib


import sys

class Tee:
    def __init__(self, file_path):
        self.file = open(file_path, 'w')
        self.terminal = sys.stdout

    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)

    def flush(self):
        self.terminal.flush()
        self.file.flush()


def load_configuration(config_path: str) -> argparse.Namespace:
    """
    Loads configuration from a YAML file.

    Args:
        config_path (str): Path to the configuration YAML file.

    Returns:
        argparse.Namespace: Namespace containing the configuration parameters.
    """
    print('LOADING CONFIGURATION FILE')
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

    if getattr(args, 'sampler', None) != None:
        print("MAKING SAMPLER")
        sampler_type = args.sampler.pop("type")
        sampler = getattr(importlib.import_module(f'samplers.{sampler_type}'),sampler_type)(**args.sampler) 
        args.executor['sampler'] = sampler

    print('MAKING EXECUTOR')    
    # Legacy support for DaskExecutor, in DaskExecutorSimulation the runner should be defined within the executor.
    args.executor['runner_args'] = getattr(args, 'runner', None)
        
    executor_type = args.executor.pop("type")
    executor = getattr(importlib.import_module(f'executors.{executor_type}'),executor_type)(**args.executor) 
    if not os.path.exists(executor.base_run_dir):
        os.makedirs(executor.base_run_dir)
    std_out_path = os.path.join(executor.base_run_dir, 'std_out.txt')
    sys.stdout = Tee(std_out_path)
    print("STARTING RUNS")
    executor.start_runs()
    print("SHUTTING DOWN SCHEDULER AND WORKERS")
    executor.clean()


if __name__ == "__main__":
    print('STARTING ENCHANTED SURROGATES')
    parser = argparse.ArgumentParser(description="Runner")
    parser.add_argument(
        "-cf",
        "--config_file",
        type=str,
        default="base",
        help="name of configuration file stored in /configs!",
    )
    config_args = parser.parse_args()
    print('LOADING ARGUMENTS FROM CONFIG FILE',config_args.config_file)
    args = load_configuration(config_args.config_file)
    print(args)
    main(args)
