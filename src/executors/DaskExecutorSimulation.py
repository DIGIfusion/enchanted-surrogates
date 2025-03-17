"""
# executors/DaskExecutor.py
Contains logic for executing surrogate workflow on Dask.
"""
import time
from time import sleep
from dask.distributed import Client, as_completed, wait, LocalCluster
from dask_jobqueue import SLURMCluster
import uuid
from common import S
import runners
import os
import traceback
from .tasks import run_simulation_task

class DaskExecutorSimulation():
    """
    Handles the splitting of samples between dask workers for running a simulation.
    SLURMCluster: https://jobqueue.dask.org/en/latest/index.html

    Attributes:
    """
    def __init__(self, sampler=None, do_initialize_client=True, **kwargs):
        # This control will be needed by the pipeline and active learning.
        self.sampler = sampler
        self.do_initialize_client=do_initialize_client
        runner_args = kwargs.get("runner")
        if type(runner_args) == type(None):
            raise ValueError('''
                             Every ExecutorSimulation needs a runner. 
                             This class defines how the code/simulation should be ran.
                             Here is an example of how it should look in the configs file
                             executor:
                                type: DaskExecutorSimulation
                                runner:
                                    type: SIMPLErunner
                                    executable_path: /path/to/to/simple.sh
                                    other_params: {}
                                    base_run_dir: /path/to/base/run
                                    output_dir: /path/to/base/out''')
        self.runner = getattr(runners, runner_args["type"])(**runner_args)
        self.worker_args: dict = kwargs.get("worker_args")
        self.n_jobs: int = kwargs.get("n_jobs", 1)
        self.runner_return_path = kwargs.get("runner_return_path")
        if self.n_jobs == 1:
            raise Warning('n_jobs=1 this means there will only be one dask worker. If you want to run samples in paralell please change <executor: n_jobs:> in the config file to be greater than 1.')
        
        self.base_run_dir = kwargs.get("base_run_dir")
        if self.base_run_dir==None:
            raise ValueError('''Enchanted surrogates handeles the creation of run directories. 
                             You must supply a base_run_dir in your configs file. Example:
                             executor:
                                type: DaskExecutorSimulation
                                base_run_dir: /project/project_462000451/test-enchanted/trial-dask
                             ...
                             ...
                             ''')
        self.base_out_dir = kwargs.get("base_out_dir")    
        
    def clean(self):
        self.client.shutdown()

    def initialize_client(self):
        """
        Initializes the client
        Args:
            worker_args (dict): Dictionary of arguments for configuring worker nodes.
            **kwargs: Additional keyword arguments.
        """
        print("Initializing DASK client")
        if self.worker_args.get("local", False):
            # TODO: Increase num parallel workers on local
            print('MAKING A LOCAL CLUSTER')
            self.cluster = LocalCluster(**self.worker_args)
        else:
            print('MAKING A SLURM CLUSTER')
            print('self.worker_args', self.worker_args)
            self.cluster = SLURMCluster(**self.worker_args)
            self.cluster.scale(self.n_jobs)    
            
        self.client = Client(self.cluster ,timeout=180)
            
    def start_runs(self, samples=None, base_run_directory_is_ready=False):
        print(f"STARTING RUNS FOR RUNNER {self.runner.type}, FROM WITHIN A {__class__}")
        if self.do_initialize_client:
            self.initialize_client()   
        
        futures = []
        print('MAKING AND SUBMITTING DASK FUTURES')
        if base_run_directory_is_ready:
            print('BASE RUN DIRECTORY IS READY:', self.base_run_dir)
            run_dir_s = os.listdir(self.base_run_dir)
            run_dir_s = [directory for directory in run_dir_s if os.path.isdir(directory)]
            for run_dir in run_dir_s:
                new_future = self.client.submit(run_simulation_task, self.runner, run_dir)
                futures.append(new_future)
        elif type(samples) != type(None):
            print('SAMPLES HAVE BEEN PROVIDED')
            # we have been given samples so we should use them.
            # This could be done in an active learning executor for example 
            for sample in samples:
                run_dir = os.path.join(self.base_run_dir, str(uuid.uuid4()))
                new_future = self.client.submit(
                    run_simulation_task, self.runner, run_dir, sample
                )
                futures.append(new_future)
        else:
            # Otherwise we assume this is the first time this is running and we need
            # to get the initial samples from the sampler
            print("GENERATING INITIAL SAMPLES:")
            samples = self.sampler.get_initial_parameters()
            print("MAKING AND SUBMITTING DASK FUTURES")         
            for sample in samples:
                random_run_id = str(uuid.uuid4())
                run_dir = os.path.join(self.base_run_dir, random_run_id)
                if self.base_out_dir == None:
                    out_dir = run_dir
                else:
                    out_dir = os.path.join(self.base_out_dir, random_run_id)
                new_future = self.client.submit(
                    run_simulation_task, self.runner, run_dir, out_dir, sample 
                )
                futures.append(new_future)
        seq = wait(futures)
        outputs = []
        for res in seq.done:
            outputs.append(res.result())
        if self.runner_return_path is not None:
            print("SAVING OUTPUT IN:", self.runner_return_path)
            with open(self.runner_return_path, "w") as out_file:
                for output in outputs:
                    out_file.write(str(output) + "\n\n")
        print("Finished sequential runs")
        return outputs

    