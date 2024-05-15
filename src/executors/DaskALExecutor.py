# executors/DaskExecutor.py

from dask.distributed import Client, as_completed
from dask_jobqueue import SLURMCluster
from .base import Executor, run_simulation_task
from common import S

class DaskALExecutor(Executor):
    sampler_interfaces = [S.ACTIVE]
    def __init__(self, num_activelearner_workers: int, num_simulator_workers: int, worker_args: dict, **kwargs):
        super().__init__(**kwargs)
        self.num_activelearner_workers = num_activelearner_workers  # kwargs.get('num_workers')
        self.num_simulator_workers = num_simulator_workers  # kwargs.get('num_workers')
        self.worker_args = worker_args  # kwargs.get('worker_args')
        self.simulator_worker_args = worker_args['simulator_args']
        self.learner_worker_args = worker_args['activelearner_args']
        print('Beginning SLURMCluster Generation')

        # Create the SLURMCluster and define the resources for each of the
        # SLURM worker jobs.
        # Note, that this is the reservation for ONE SLURM worker job.
        self.simulator_cluster = SLURMCluster(**self.simulator_worker_args)
        self.activelearner_cluster = SLURMCluster(**self.learner_worker_args)

        # This launches the cluster (submits the worker SLURM jobs)
        self.simulator_cluster.scale(self.num_simulator_workers)

        self.simulator_client = Client(self.simulator_cluster)
        print('Finished Setup')
        self.clients.append(self.simulator_client)
        if self.sampler.sampler_interface not in self.sampler_interfaces:
            raise ValueError('Sampler is not allowed for this executor')

    def scale_learner_cluster_client(self, ):
        self.activelearner_cluster.adapt(minimum_jobs=1,maximum_jobs=self.num_activelearner_workers)
        self.activelearner_client = Client(self.activelearner_cluster)
        self.clients.append(self.activelearner_client)
    
    def start_runs(self):
        sampler_interface = self.sampler.sampler_interface
        print(100 * "=")
        print("Starting Database generation")
        print("Creating initial runs")
        futures = []

        initial_parameters = self.sampler.get_initial_parameters()

        for params in initial_parameters:
            new_future = self.simulator_client.submit(
                run_simulation_task, self.runner_args, params, self.base_run_dir
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
            # NOTE: Blocking until empty
            if self.max_samples > completed:
                self.scale_learner_cluster_client()
                param_list = self.sampler.get_next_parameter()
                for params in param_list:
                    new_future = self.simulator_client.submit(
                           run_simulation_task, self.runner_args, params, self.base_run_dir
                       )
                    seq.add(new_future)

                # TODO: pass the previous result and parameters.. (Active Learning )

                #if sampler_interface in [S.BATCH]:
                #    param_list = self.sampler.get_next_parameter()
                #    for params in param_list:
                #        new_future = self.simulator_client.submit(
                #            run_simulation_task, self.runner_args, params, self.base_run_dir
                #        )
                #    seq.add(new_future)
                #elif sampler_interface in [S.SEQUENTIAL]:
                #    params = self.sampler.get_next_parameter()
                #    if params is None:  # This is hacky
                #        continue
                #    else:
                #        new_future = self.simulator_client.submit(
                #            run_simulation_task, self.runner_args, params, self.base_run_dir
                #        )
                #        seq.add(new_future)
                
