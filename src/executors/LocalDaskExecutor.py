# executors/DaskExecutor.py

from dask.distributed import Client
from .DaskExecutor import DaskExecutor


class LocalDaskExecutor(DaskExecutor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print('Beginning local Cluster Generation')

        # calling Client with no arguments creates a local cluster
        # it is possible to add arguments like:
        # n_workers=2, threads_per_worker=4
        self.client = Client()
        print('Finished Setup')
