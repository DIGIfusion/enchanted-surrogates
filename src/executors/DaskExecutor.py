# executors/DaskExecutor.py

from dask.distributed import Client
from dask_jobqueue import SLURMCluster
from .base import Executor
from dask.distributed import as_completed


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

    
    def start_runs(self):
        print(100 * "=")
        print("Starting Database generation")
        print("Creating initial runs")
        futures = []

        # TODO: implement get_initial_parameters() from sampler
        for _ in range(self.sampler.num_initial_points):
            params = self.sampler.get_next_parameter()
            new_future = self.client.submit(
                self.run_simulation_task, self.runner_args, params, self.base_run_dir
            )
            futures.append(new_future)

        print("Starting search")
        seq = as_completed(futures)
        completed = 0
        for future in seq:
            res = future.result()
            completed += 1
            print(res, completed)
            # TODO: is this waiting for an open node or are we just
            # pushing to queue?
            if self.max_samples > completed:
                # TODO: pass the previous result and parameters..
                # For active learning
                params = self.sampler.get_next_parameter()
                if params is None:  # This is hacky
                    continue
                else:
                    new_future = self.client.submit(
                        self.run_simulation_task, self.runner_args, params, self.base_run_dir
                    )
                    seq.add(new_future)


    
