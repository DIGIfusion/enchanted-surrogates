"""
Utilies for importing multiple executors, samplers and runners.
"""

from dataclasses import dataclass
from enchanted_surrogates.utils.logger import get_logger
from enchanted_surrogates.executors.base_executor import Executor
from enchanted_surrogates.samplers.base_sampler import Sampler
from enchanted_surrogates.utils.precise_imports import import_sampler, import_executor

log = get_logger(__name__)

@dataclass
class RunGroup:
    """
    Container for a group of executors, samplers, and runners. Represents one level of depth
    in nested execution. If multiple executors or runners are defined, sequential execution is
    (also) used.
    """
    executors: list[Executor]
    sampler: Sampler
    runners: list[dict]

    def validate(self):
        """
        Raises ValueError if the run group is not set up correctly.
        """
        if len(self.executors) != len(self.runners):
            raise ValueError("The amount of runners and executors should be the same!")
        if len(self.executors) == 0:
            raise ValueError("At least one executor should be specified!")
        if len(self.runners) == 0:
            raise ValueError("At least one runner should be specified!")
        if not self.sampler:
            raise ValueError("Sampler should be specified!")


def parse_sequential_group(group_config: dict) -> tuple[str, list[str], list[str]]:
    """
    Parses sampler and sequential executors and runners from group config. Executors and runners
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


def parse_all_run_groups(args) -> list[RunGroup]:
    """
    Imports and parses all run groups defined in config args in correct nesting order.

    Args:
        args: Namespace/dict parsed from yaml.
    Returns:
        out (list[RunGroup]): List of nested run groups.
    """
    executors = import_executors(args)
    samplers = import_samplers(args)
    group_configs = import_run_groups(args)

    nested_groups: list[RunGroup] = []
    for group in group_configs:
        group_sampler, group_executors, group_runners = parse_sequential_group(group)
        run_group = RunGroup(
            [executors[e] for e in group_executors],
            samplers[group_sampler],
            [{**args.runners[r], "__runner_name": r} for r in group_runners],
        )

        try:
            run_group.validate()
        except ValueError as error:
            raise ValueError(
                "Parsing run group config failed: " + str(error)
                + " Given config was: " + str(group)
            )

        nested_groups.append(run_group)

    return nested_groups


def import_saved_files_list(args) -> list[str]:
    """
    Imports supervisor/save_files_list from config.

    Returns
        List of strings containing names of files to be saved
    """
    return args.supervisor.get("save_files_list", [])
