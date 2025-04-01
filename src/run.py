# run.py
from dask.distributed import Client, print
import dask.distributed as dd
import sys

class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)

    def flush(self):
        for f in self.files:
            f.flush()
            
file = open('output2.txt', 'w')
# Create a Tee object that writes to both stdout and the file
tee = Tee(sys.stdout, file)
# Redirect print output to the Tee object
print('Hello, World!', file=tee)

# out_file = open('output.txt', 'w')
# # Redirect print output to the file
# print('Hello, World!', file=out_file)
def print(*args, **kwargs):
    dd.print(*args, file=tee, **kwargs)

print('PERFORMING IMPORTS')
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
    print("MAKING SAMPLER AND EXECUTOR")
    sampler = getattr(samplers, args.sampler.pop("type"))(**args.sampler)
    
    # Legacy support for DaskExecutor, in DaskExecutorSimulation the runner should be defined within the executor.
    args.executor['runner_args'] = getattr(args, 'runner', None)
    
    executor = getattr(executors, args.executor.pop("type"))(sampler = sampler, **args.executor)
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
