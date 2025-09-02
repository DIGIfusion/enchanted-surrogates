import os
from .base_executor import Executor
from dask.distributed import Client, as_completed, wait, LocalCluster, print
from dask_jobqueue import SLURMCluster
import time
import warnings
from .simulation_task import run_simulation_task
import importlib
import uuid
import pandas as pd

class DaskExecutor(Executor):
    """
    Handles execution of surrogate workflow on Dask.
    SLURMCluster: https://jobqueue.dask.org/en/latest/index.html
    """
    
    def __init__(self, base_run_dir, sampler_args, *args, **kwargs):
        self.type = kwargs.get('type')
        self.base_run_dir=base_run_dir
        self.scale_n_jobs = kwargs.get('scale_n_jobs',1)
        self.SLURMcluster_args = kwargs.get('SLURMcluster_args')
        sampler_type = sampler_args.pop("type")
        self.sampler = getattr(importlib.import_module(f'samplers.{sampler_type}'),sampler_type)(**sampler_args)
        self.cluster=None
        self.client=None
        self.expected_number_of_workers = None
        
    
        
    def start_cluster(self, slurm_out_dir=None):
        print('MAKING CLUSTER')
        if not slurm_out_dir:
            slurm_out_dir = os.path.join(self.base_run_dir,'worker_out_DaskExecutor')
            jed = self.SLURMcluster_args.get('job_extra_directives')
            if not jed:
                self.SLURMcluster_args['job_extra_directives']=[f'-o {slurm_out_dir}/%x.%j.out',f'-e {slurm_out_dir}/%x.%j.err']
            else:
                self.SLURMcluster_args['job_extra_directives']+=[f'-o {slurm_out_dir}/%x.%j.out',f'-e {slurm_out_dir}/%x.%j.err']    
        self.cluster = SLURMCluster(**self.SLURMcluster_args)
        self.cluster.scale(self.scale_n_jobs)
        print('THE JOB SCRIPT FOR A WORKER IS:')
        print(self.cluster.job_script())
        
        self.client = Client(self.cluster ,timeout=180)
        print('SCHEDULER ADDRESS',self.cluster.scheduler_address)
        print('DASHBOARD LINK',self.client.dashboard_link)        
        
        self.expected_number_of_workers = self.scale_n_jobs * int(self.SLURMcluster_args.get('processes',1))
        print(f"Waiting for {self.expected_number_of_workers} workers to start...")
        
        for _ in range(self.expected_number_of_workers+120):
            print(f"Connected to {len(workers)} workers out of expected {self.expected_number_of_workers}.\n")
            workers = self.client.scheduler_info()["workers"]
            if len(workers) == self.expected_number_of_workers:
                break
            if len(workers) > self.expected_number_of_workers:
                warnings.warn(f"More workers ({len(workers)}) than expected ({self.expected_number_of_workers})")
                break
            time.sleep(1)
        
        print('WORKER INFORMATION:')
        for addr, info in workers.items():
            print(f"Worker {addr}:")
            print(f"  CPUs: {info['nthreads']}")
            print(f"  Memory: {info['memory_limit'] / 1e9:.2f} GB")
            print(f"  Resources: {info.get('resources', {})}\n")
            
    def start_runs(self):
        start = time.time()
        if not os.path.exists(self.base_run_dir):
            os.makedirs(self.base_run_dir)
        
        if os.path.exists(os.path.join(self.base_run_dir, 'ENCHANTED.FINNISHED')):
            raise FileExistsError(f'''The file: {self.base_run_dir}/ENCHANTED.FINNISHED, exists.
                                  This signifies that there is already data in this folder. 
                                  Aborting to avoid accidental data mixing.''' )
        
        print(f"STARTING RUNS FOR RUNNER {self.runner_args['type']}, FROM WITHIN A {__class__}")
        
        if not self.client:
            self.start_cluster()
        
        all_futures = []
        
        while self.sampler.has_budget:
            samples = self.sampler.get_next_samples()
            futures = self.submit_batch(samples)
            all_futures.extend(futures)
        
        dfs = []
        for future in as_completed(all_futures):
            result = future.result()
            dfs.append(pd.DataFrame(result))
        df_dataset = pd.concat(dfs)
        df_dataset.to_csv(os.path.join(self.base_run_dir, 'enchanted_dataset.csv'))
            
    def submit_batch(self, samples):
        futures = []
        for sample_params in samples:
            sample_run_dir = os.path.join(self.base_run_dir, str(uuid.uuid4()))  # TODO. uuid.uuid should probably have a random seed ? 
            new_future = self.client.submit(
                run_simulation_task, self.runner_args, sample_run_dir, sample_params
            )
            futures.append(new_future)
        return futures
        