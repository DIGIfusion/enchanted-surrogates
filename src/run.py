# run.py
from dask.distributed import print
print('PERFORMING IMPORTS')
import os,sys
sys.path.append(os.path.join(os.path.dirname(__file__)))
import yaml

import argparse
import importlib

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
    print("STARTING RUNS")
    executor.start_runs()
    print("SHUTTING DOWN SCHEDULER AND WORKERS")
    executor.clean()
    return sampler, executor


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
