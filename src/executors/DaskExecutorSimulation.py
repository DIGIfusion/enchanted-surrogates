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
import importlib
# import runners
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
        self.runner_args = kwargs.get("runner")
        if type(self.runner_args) == type(None):
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
        self.worker_args: dict = kwargs.get("worker_args")
        self.n_jobs: int = kwargs.get("n_jobs", 1)
        self.runner_return_path = kwargs.get("runner_return_path")
        self.runner_return_headder = kwargs.get("runner_return_headder")
        
        self.worker_args: dict = kwargs["worker_args"]
        self.n_jobs: int = kwargs.get("n_jobs", 1)
        if self.n_jobs == 1:
            warnings.warn('n_jobs=1 this means there will only be one dask worker. If you want to run samples in paralell please change <executor: n_jobs:> in the config file to be greater than 1.')
        
        self.base_run_dir = kwargs.get("base_run_dir")
        self.runner_return_path = kwargs.get("runner_return_path")
        self.runner_return_headder = kwargs.get("runner_return_headder", f'{self.__class__}: no runner_return_headder, was provided in configs file')
        if self.base_run_dir==None and self.runner_return_path==None:
            warnings.warn(f'NO base_run_dir or runner_return_path WAS DEFINED FOR {self.__class__}')
        elif self.runner_return_path==None and self.base_run_dir!=None:
            self.runner_return_path = os.path.join(self.base_run_dir, 'runner_return.txt')
        
        
    def clean(self):
        self.client.shutdown()

    def initialize_client(self, slurm_out_dir=None):
        """
        Initializes the client
        Args:
            worker_args (dict): Dictionary of arguments for configuring worker nodes.
            **kwargs: Additional keyword arguments.
        """
        if slurm_out_dir != None:
            jed = self.worker_args.get('job_extra_directives')
            if type(jed) == type(None):
                self.worker_args['job_extra_directives']=[f'-o {slurm_out_dir}/%x.%j.out',f'-e {slurm_out_dir}/%x.%j.err']
            else:
                self.worker_args['job_extra_directives']+=[f'-o {slurm_out_dir}/%x.%j.out',f'-e {slurm_out_dir}/%x.%j.err']
        print("Initializing DASK client")
        if self.worker_args.get("local", False):
            # TODO: Increase num parallel workers on local
            print(f"MAKING A LOCAL CLUSTER FOR {self.runner_args['type']}")
            self.cluster = LocalCluster(**self.worker_args)
        else:
            print(f"MAKING A SLURM CLUSTER FOR {self.runner_args['type']}")
            self.cluster = SLURMCluster(**self.worker_args)
            self.cluster.scale(self.n_jobs)    
            print('THE JOB SCRIPT FOR A WORKER IS:')
            print(self.cluster.job_script())
            
        self.client = Client(self.cluster ,timeout=180)
            
    def start_runs(self):
        if self.base_run_dir==None:
            raise ValueError('When executing start_runs of {__class__} a self.base_run_dir must be specified.')
        else:
            if not os.path.exists(self.base_run_dir):
                os.mkdir(self.base_run_dir)
    
        if os.path.exists(os.path.join(self.base_run_dir, 'FINNISHED')):
            raise FileExistsError(f'''The file: {self.base_run_dir}/FINNISHED, exists.
                                  This signifies that there is already data in this folder. 
                                  Aborting to avoid accidental data mixing.''' )
        
        print(f"STARTING RUNS FOR RUNNER {self.runner_args['type']}, FROM WITHIN A {__class__}")
        
        print('MAKING CLUSTER')
        self.initialize_client(slurm_out_dir=self.base_run_dir)
        

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
        with open(os.path.join(self.base_run_dir,'FINNISHED'), 'w') as file:
            file.write(f'FINNISHED, {__class__}')
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
        n_samples = len(params)
        for index, sample in enumerate(params):
            new_future = self.client.submit(
                run_simulation_task, runner_args=self.runner_args, run_dir=run_dirs[index], params=sample 
            )
            futures.append(new_future)
        return futures
    
    def write_summary(self, directory, *args, **kwargs):
        if 'write_summary' in dir(self.runner):
            self.runner.write_summary(directory, *args, **kwargs)
        
    