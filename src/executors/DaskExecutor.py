# executors/DaskExecutor.py

import os
from dask.distributed import Client
from dask_jobqueue import SLURMCluster
from .base import Executor


class DaskExecutor(Executor):
    def __init__(
            self, sampler, runner_args, base_run_dir: str, worker_args: dict,
            num_workers: int):
        print("Starting Setup")
        self.sampler = sampler
        self.runner_args = runner_args
        self.base_run_dir = base_run_dir
        self.max_samples = sampler.total_budget
        self.num_workers = num_workers

        print(f'Making directory of simulations at: {self.base_run_dir}')
        os.makedirs(base_run_dir, exist_ok=True)

        print('Beginning Cluster Generation')

        # Create the SLURMCluster and define the resources for each of the
        # SLURM worker jobs.
        # Note, that this is the reservation for ONE SLURM worker job.
        self.cluster = SLURMCluster(**worker_args)

        # This launches the cluster (submits the worker SLURM jobs)
        self.cluster.scale(self.num_workers)
        self.client = Client(self.cluster)
        print('Finished Setup')
