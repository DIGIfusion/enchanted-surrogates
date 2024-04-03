# executors/DaskExecutor.py

from dask.distributed import Client
from dask_jobqueue import SLURMCluster
from .base import Executor


class DaskExecutor(Executor):
    def __init__(self, num_workers, worker_args, **kwargs):
        super().__init__(**kwargs)
        self.num_workers = num_workers  # kwargs.get('num_workers')
        self.worker_args = worker_args  # kwargs.get('worker_args')
        print('Beginning SLURMCluster Generation')

        # Create the SLURMCluster and define the resources for each of the
        # SLURM worker jobs.
        # Note, that this is the reservation for ONE SLURM worker job.
        self.cluster = SLURMCluster(**self.worker_args)

        # This launches the cluster (submits the worker SLURM jobs)
        self.cluster.scale(self.num_workers)
        self.client = Client(self.cluster)
        print('Finished Setup')
