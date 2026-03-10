"""
Utilies for importing multiple executors, samplers and runners.
"""

from dataclasses import dataclass
from enchanted_surrogates.executors.base_executor import Executor
from enchanted_surrogates.samplers.base_sampler import Sampler
from enchanted_surrogates.utils.precise_imports import import_sampler, import_executor

@dataclass
class RunGroup:
    """
    Container for a group of executors, samplers, and runners
    """
    executor: Executor
    sampler: Sampler
    runner: dict

def import_executors(args) -> dict[str, Executor]:
    """
    Imports all executors from config, under 'executors' key
    
    Args
        Dictionary or namespace object, parsed from yaml
    Returns
        Dictionary mapping executor unique name to class instance
    """
    executors = {}
    for name, executor_config in args.executors.items():
        executors[name] = import_executor(executor_config["type"], executor_config)

    return executors

def import_samplers(args) -> dict[str, Sampler]:
    """
    Imports all samplers from config, under 'samplers' key
    
    Args
        Dictionary or namespace object, parsed from yaml
    Returns
        Dictionary mapping sampler unique name to class instance
    """
    samplers = {}
    for name, sampler_config in args.samplers.items():
        samplers[name] = import_sampler(sampler_config["type"], sampler_config)

    return samplers

def import_run_groups(args) -> list[dict]:
    """
    Imports supervisor/run_order from config. 

    Returns
        List of dicts with keys 'executor', 'sampler' and 'runner' mapping
        to their unique names.
    """
    return args.supervisor["run_order"]

def import_saved_files_list(args) -> list[str]:
    """
    Imports supervisor/save_files_list from config.
    
    Returns
        List of strings containing names of files to be saved
    """
    return args.supervisor.get("save_files_list",[])
