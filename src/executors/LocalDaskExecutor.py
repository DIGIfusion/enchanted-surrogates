# executors/DaskExecutor.py

import os
from dask.distributed import Client
from .base import Executor


class LocalDaskExecutor(Executor):
    def __init__(
            self, sampler, runner_args, base_run_dir: str, worker_args: dict,
            num_workers: int):
        print("Starting Setup")
        self.sampler = sampler
        self.runner_args = runner_args
        self.base_run_dir = base_run_dir
        self.max_samples = sampler.num_samples
        self.num_workers = num_workers

        print(f'Making directory of simulations at: {self.base_run_dir}')
        os.makedirs(base_run_dir, exist_ok=True)

        print('Beginning local Cluster Generation')

        # calling Client with no arguments creates a local cluster
        # it is possible to add arguments like:
        # n_workers=2, threads_per_worker=4
        self.client = Client()
        print('Finished Setup')
