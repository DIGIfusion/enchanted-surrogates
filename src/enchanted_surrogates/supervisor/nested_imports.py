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
    executors: list[Executor]
    sampler: Sampler
    runners: list[dict]


def parse_sequential_group(group_config: dict) -> tuple[str, list[str], list[str]]:
    """
    Parse sampler and sequential executors and runners from group config. Executors and runners
    can be defined either as a single value or a list.

    Args:
        group_config (dict): Configuration dictionary parsed from yaml
    Returns:
        out (tuple[str, list[str], list[str]]): Tuple containg name of sampler, list of names
            of executors and list of names of runners
    """
    group_executors = (
        group_config["executor"]
        if type(group_config["executor"]) is list
        else [group_config["executor"]]
    )

    group_runners = (
        group_config["runner"]
        if type(group_config["runner"]) is list
        else [group_config["runner"]]
    )

    if len(group_runners) != len(group_executors):
        raise ValueError(
            "Error when parsing run group config: the amount of runners and "
            + "executors does not match! Config was: " + str(group_config)
        )
    
    sampler = group_config["sampler"]

    return (sampler, group_executors, group_runners)


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
    return args.supervisor.get("save_files_list", [])
