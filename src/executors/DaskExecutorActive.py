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
import warnings
from common import S
import executors
class DaskExecutorActive():
    """
    Handles the splitting of samples between dask workers for running a simulation.
    SLURMCluster: https://jobqueue.dask.org/en/latest/index.html

    Attributes:
    """
    
    def __init__(self, sampler=None, **kwargs):
        # This control will be needed by the pipeline and active learning.
        self.sampler = sampler
        assert self.sampler.sampler_interface == S.ACTIVE
        
        self.runner_return_path = kwargs.get("runner_return_path")
        self.runner_return_headder = kwargs.get("runner_return_headder")
        
        if self.runner_return_path != None and self.runner_return_headder != None:
            print('MAKING RUNNER RETURN FILE')
            with open(self.runner_return_path, 'w') as file:
                file.write(self.runner_return_headder+'\n')
        
        self.base_run_dir = kwargs.get("base_run_dir")
        
        static_executor_kwargs = kwargs['static_executor']
        self.static_executor = getattr(executors, static_executor_kwargs['type'])(**static_executor_kwargs)
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
            print(f'MAKING A LOCAL CLUSTER FOR {self.runner.__class__.__name__}')
            self.cluster = LocalCluster(**self.worker_args)
        else:
            print(f'MAKING A SLURM CLUSTER FOR {self.runner.__class__.__name__}')
            self.cluster = SLURMCluster(**self.worker_args)
            self.cluster.scale(self.n_jobs)    
            print('THE JOB SCRIPT FOR A WORKER IS:')
            print(self.cluster.job_script())
            
        self.client = Client(self.cluster ,timeout=180)
            
    def start_runs(self):
        print(f"STARTING RUNS FOR RUNNER {self.runner.__class__.__name__}, FROM WITHIN A {__class__}")
        
        print('MAKING CLUSTER')
        self.initialize_client()
        
        futures = []
        print("GENERATING INITIAL SAMPLES:")
        samples = self.sampler.get_initial_parameters()
        
        run_dirs = [None]*len(samples)
        if self.base_run_dir==None:
            warnings.warn('''
                            No base_run_dir has been provided. It is now assumed that the runner being used does not need a run_dir and will be passed None.
                            This could be true if the runner is executing a python function and not a simulation.
                            Otherwise see how to insert a base_run_dir into a config file below:
                            Example
                            
                            executor:
                                type: DaskExecutorSimulation
                                base_run_dir: /project/path/to/base_run_dir/
                            ...
                            ...
                        ''')
        else: # Make run_dirs
            print("MAKING RUN DIRECTORIES")
            for index, sample in enumerate(samples):
                random_run_id = str(uuid.uuid4())
                run_dir = os.path.join(self.base_run_dir, random_run_id)
                os.system(f'mkdir {run_dir} -p')
                run_dirs[index] = run_dir
                     
        print("MAKING AND SUBMITTING DASK FUTURES")         
        for index, sample in enumerate(samples):
            new_future = self.client.submit(
                run_simulation_task, self.runner, run_dirs[index], sample 
            )
            futures.append(new_future)
        
        print("DASK FUTURES SUBMITTED, WAITING FOR THEM TO COMPLETE")
        seq = wait(futures)
        outputs = []
        for res in seq.done:
            outputs.append(res.result())
        if self.runner_return_path is not None:
            print("SAVING OUTPUT IN:", self.runner_return_path)
            with open(self.runner_return_path, "a") as out_file:
                for output in outputs:
                    out_file.write(str(output) + "\n\n")
        print("Finished sequential runs")
        return outputs

    