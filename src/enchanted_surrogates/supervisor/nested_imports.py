from dataclasses import dataclass
from enchanted_surrogates.executors.base_executor import Executor
from enchanted_surrogates.samplers.base_sampler import Sampler
from enchanted_surrogates.utils.precise_imports import import_sampler, import_executor

@dataclass
class RunGroup:
    executor: Executor
    sampler: Sampler
    runner: dict

def import_executors(args) -> dict[str, Executor]:
    executors = {}
    for name, executor_config in args.executors.items():
        executors[name] = import_executor(executor_config["type"], executor_config)
    
    return executors

def import_samplers(args) -> dict[str, Sampler]:
    samplers = {}
    for name, sampler_config in args.samplers.items():
        samplers[name] = import_sampler(sampler_config["type"], sampler_config)
    
    return samplers

def import_run_groups(args) -> list[dict]:
    return args.supervisor["run_order"]