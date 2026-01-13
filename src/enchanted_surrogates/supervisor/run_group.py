from dataclasses import dataclass
from enchanted_surrogates.executors.base_executor import Executor
from enchanted_surrogates.samplers.base_sampler import Sampler

@dataclass
class RunGroup:
    executor: Executor
    sampler: Sampler
    runner: dict