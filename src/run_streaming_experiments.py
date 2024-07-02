# run.py
import yaml
import os
import samplers
import executors
import argparse
import numpy as np
import torch
import pandas as pd
from joblib import Parallel, delayed

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


def main(cfg: str, n_repeats: int=1):
    """
    Main function for running the simulation workflow.

    Args:
        args (argparse.Namespace): Namespace containing the configuration parameters.
    """

            
    args = load_configuration(cfg)
    print(args)
    base_filename = args.sampler["filename_save"]
    for i in range(n_repeats):
        print(20*"=", f"\n \t Running with {config}, repeat number {i}\n", 20*"=")
        
        np.random.seed(99)
        save_fname =  base_filename+str(i)
        args.sampler["filename_save"] = save_fname
        sampler = getattr(samplers, args.sampler["type"])(**args.sampler)
        executor = getattr(executors, args.executor["type"])(
            sampler=sampler, runner_args=args.runner, **args.executor
        )

        executor.start_runs()
        executor.clean()


if __name__ == "__main__":


    parser = argparse.ArgumentParser(description="Runner")
    parser.add_argument(
        "-p",
        "--path",
        type=str,
        default="base",
        help="path where configs are stored",
    )

    parser.add_argument(
        "-n",
        "--n_repeats",
        type=int,
        default=1,
        help="Number of NN runs",
    )

    parser.add_argument(
        "-cfs",
        "--configs",
        nargs='+',
        type=str,
        default="base",
        help="list of config files",
    )

    parser_args = parser.parse_args()
    path = parser_args.path
    configs = parser_args.configs
    configs = [os.path.join(path,current_config) for current_config in configs]
    n_repeats = parser_args.n_repeats
    for config in configs:
    # NOTE: ensuring same shots used
        print(20*"=", f"\n \t Running with {config}\n", 20*"=")
        main(config, n_repeats)
    #Parallel(n_jobs=len(configs))(delayed(main)(current_config) for current_config in configs)

