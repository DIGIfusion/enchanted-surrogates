# executors/DaskExecutor.py

from dask.distributed import Client, as_completed, wait
from dask_jobqueue import SLURMCluster
from .base import Executor, run_simulation_task
from common import S

class DaskCPUGPUExecutor(Executor):
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
        self.clients.append(self.client)
        self.cleints = self.clients_tuple_type(simulationrunner=self.client, surrogatetrainer=self.client)

        print('Finished Setup')

    def start_runs(self):
        raise NotImplementedError