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

class DaskExecutorSimulation():
    """
    Handles the splitting of samples between dask workers for running a simulation.
    SLURMCluster: https://jobqueue.dask.org/en/latest/index.html

    Attributes:
    """
    def __init__(self, sampler=None, **kwargs):
        # This control will be needed by the pipeline and active learning.
        self.sampler = sampler
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
                                    output_dir: /path/to/base/out
                            ''')
        self.runner = getattr(runners, runner_args["type"])(**runner_args)
        self.worker_args: dict = kwargs.get("worker_args")
        self.n_jobs: int = kwargs.get("n_jobs", 1)
        
        if self.n_jobs == 1:
            warnings.warn('n_jobs=1 this means there will only be one dask worker. If you want to run samples in paralell please change <executor: n_jobs:> in the config file to be greater than 1.')
        
        self.base_run_dir = kwargs.get("base_run_dir")
        self.runner_return_path = kwargs.get("runner_return_path")
        self.runner_return_headder = kwargs.get("runner_return_headder", f'{__class__}: no runner_return_headder, was provided in configs file')
        if self.base_run_dir==None and self.runner_return_path==None:
            warnings.warn(f'NO base_run_dir or runner_return_path WAS DEFINED FOR {__class__}')
        elif self.runner_return_path==None and self.base_run_dir!=None:
            self.runner_return_path = os.path.join(self.base_run_dir, 'runner_return.txt')
        
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
        

        print("GENERATING INITIAL SAMPLES:")
        params = self.sampler.get_initial_parameters()
        futures = self.submit_batch_of_params(params, self.base_run_dir)
        
        print("DASK FUTURES SUBMITTED, WAITING FOR THEM TO COMPLETE")
        seq = wait(futures)
        outputs = []
        for res in seq.done:
            outputs.append(res.result())
            
        if self.runner_return_path is not None:
            print("SAVING OUTPUT IN:", self.runner_return_path)
            with open(self.runner_return_path, "w") as out_file:
                out_file.write(self.runner_return_headder+'\n')
                for output in outputs:
                    out_file.write(str(output)+"\n")
        print("Finished sequential runs")
        return outputs
    
    def submit_batch_of_params(self, params: dict, base_run_dir:str=None):
        run_dirs = [None]*len(params)
        if base_run_dir==None:
            base_run_dir = self.base_run_dir
        if base_run_dir==None:
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
            for index, sample in enumerate(params):
                random_run_id = str(uuid.uuid4())
                run_dir = os.path.join(base_run_dir, random_run_id)
                os.system(f'mkdir {run_dir} -p')
                run_dirs[index] = run_dir
                     
        print("MAKING AND SUBMITTING DASK FUTURES")      
        futures = []   
        for index, sample in enumerate(params):
            new_future = self.client.submit(
                run_simulation_task, self.runner, run_dirs[index], sample 
            )
            futures.append(new_future)
        return futures
        
    