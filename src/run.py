# run.py
from dask.distributed import print
print('PERFORMING IMPORTS')
import os,sys
sys.path.append(os.path.join(os.path.dirname(__file__)))
import yaml
import os
import argparse
import importlib

import shutil
import sys

# This allows any prints to go to both std_out and a file.
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
    else:
        sampler = 'Sampler not defined, potentially defined in executor'
        
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
    # Get arguments passed to run.py
    parser = argparse.ArgumentParser(description="Runner")
    parser.add_argument(
        "-cf",
        "--config_file",
        type=str,
        default="base",
        help="name of configuration file stored in /configs!",
    )
    config_args = parser.parse_args()
    print('SETTING CONFIG PATH ENV VARIABLE') # workers inherit the inviroment of the scheduler and so can also see this config path and get config arguments if needed
    # this is used in tasks.py to get the run_dir convention
    os.environ['ENCHANTED_CONFIG_PATH'] = config_args.config_file
    print('LOADING ARGUMENTS FROM CONFIG FILE',config_args.config_file)
    args = load_configuration(config_args.config_file)
        # print('debug keys',args.executor.keys())
    base_run_dir = args.executor.get('base_run_dir')
    if base_run_dir == None:
        base_run_dir=args.general.get('top_executor_base_run_dir')
    if base_run_dir == None:
        raise ValueError('You must specify a base_run_dir in the top executor namelist or top_executor_base_run_dir in the general namelist')
    if not os.path.exists(base_run_dir):
        os.makedirs(base_run_dir)
    
    # copying config file to base_run_dir
    shutil.copy(config_args.config_file, os.path.join(base_run_dir, os.path.basename(config_args.config_file)))
    # Channeling prints and errors to both a file and the terminal
    std_out_path = os.path.join(base_run_dir, 'ENCHANTED.out')
    sys.stdout = Tee(std_out_path)
    sys.stderr = sys.stdout
    
    print(args)
    main(args)
