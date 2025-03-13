"""
# executors/DaskExecutor.py
Contains logic for executing surrogate workflow on Dask.
"""
import time
from dask.distributed import Client, as_completed, wait
from dask_jobqueue import SLURMCluster, LocalCluster
import uuid
from nn.networks import load_saved_model, run_train_model
from common import S
from .base import Executor, run_simulation_task
import os

class DaskExecutorSimulation(Executor):
    """
    Handles the splitting of samples between dask workers for running a simulation.
    SLURMCluster: https://jobqueue.dask.org/en/latest/index.html

    Attributes:
        n_jobs (int): Number of batch jobs to be created by SLURMcluster.
            In dask-jobqueue, a single Job may include one or more Workers.
        worker_args (dict): Dictionary of arguments for configuring worker nodes.
            The arguments depends on the type of Cluster used.
            - n_workers (int): Number of workers
        client (dask.distributed.Client): Dask client for task submission and execution.
        simulator_client (dask.distributed.Client): Client for simulator tasks.
        surrogate_client (dask.distributed.Client): Client for training surrogate models.
        clients (list): List of Dask clients.
    """

    def __init__(self, worker_args: dict,  do_initialize_client=True, **kwargs):
        # This control will be needed by the pipeline and active learning.
        self.do_initialize_client=do_initialize_client
        super().__init__(**kwargs)        
        self.n_jobs: int = kwargs.get("n_jobs", 1)
        if self.n_jobs == 1:
            raise Warning('n_jobs=1 this means there will only be one dask worker. If you want to run samples in paralell please change <executor: n_jobs:> in the config file to be greater than 1.')
        self.worker_args: dict = worker_args

    def shutdown(self):
        self.client.shutdown()

    def initialize_client(self):
        """
        Initializes the clients based on sampler enum,
        general steps are: initialize cluster, scale cluster, initialize client with cluster

        Args:
            worker_args (dict): Dictionary of arguments for configuring worker nodes.
            **kwargs: Additional keyword arguments.
        """
        print("Initializing DASK clients")
        if self.worker_args.get("local", False):
            # TODO: Increase num parallel workers on local
            self.cluster = LocalCluster(**self.worker_args)
            self.client = Client(self.cluster ,timeout=60)
        else:
            self.cluster = SLURMCluster(**self.worker_args)
            self.cluster.scale(self.n_jobs)
            self.simulator_client = Client(self.simulator_cluster)
            self.clients = [self.simulator_client]
    
    def start_runs(self, samples=None, base_run_directory_is_ready=False):
        if self.initialize_clients:
            self.initialize_client()   
        
        futures = []        
        if base_run_directory_is_ready:
            print('BASE RUN DIRECTORY IS READY:', self.base_run_dir)
            run_dir_s = os.listdir(self.base_run_dir)
            run_dir_s = [directory for directory in run_dir_s if os.path.isdir(directory)]
            for run_dir in run_dir_s:
                new_future = self.client.submit(run_simulation_task, self.runner_args, run_dir)
                futures.append(new_future)
        elif type(samples) != type(None):
            print('SAMPLES HAVE BEEN PROVIDED')
            # we have been given samples so we should use them.
            # This could be done in an active learning executor for example 
            for sample in samples:
                run_dir = os.path.join(self.base_run_dir, str(uuid.uuid4()))
                new_future = self.client.submit(
                    run_simulation_task, self.runner_args, run_dir, sample
                )
                futures.append(new_future)
        else:
            # Otherwise we assume this is the first time this is running and we need
            # to get the initial samples from the sampler
            print("GENERATING INITIAL SAMPLES:")
            samples = self.sampler.get_initial_parameters()            
            for sample in samples:
                run_dir = os.path.join(self.base_run_dir, str(uuid.uuid4()))
                new_future = self.client.submit(
                    run_simulation_task, self.runner_args, run_dir, sample
                )
                futures.append(new_future)
        seq = wait(futures)
        outputs = []
        for res in seq.done:
            outputs.append(res.result())
        if self.output_dir is not None:
            output_file_path = os.path.join(self.output_dir, "runner_returns")
            print("SAVING OUTPUT IN:", output_file_path)
            with open(output_file_path, "w") as out_file:
                for output in outputs:
                    out_file.write(str(output) + "\n\n")
        print("Finished sequential runs")
        return outputs

    