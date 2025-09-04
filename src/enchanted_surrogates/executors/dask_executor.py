import os
from .base_executor import Executor
from dask.distributed import Client, as_completed, wait, LocalCluster, print
from dask_jobqueue import SLURMCluster
from dask.distributed import LocalCluster
import subprocess
import time
import warnings
from .simulation_task import run_simulation_task as run_simulation_task_origional
import importlib
import uuid
import pandas as pd


def run_simulation_task_monkey_patch(*args, **kwargs):
    """
    Monkey simulation_task.run_simulation_task to import dask.distribted.print

    This allows the prints in this function to show on the std output stream and not only in the dask worker log files.
    This is important as there is as there is a print to show the traceback of errors happening on the dask workers.
    """
    from dask.distributed import print
    return run_simulation_task_origional(*args, **kwargs)

run_simulation_task = run_simulation_task_monkey_patch

class DaskExecutor(Executor):
    """
    Handles execution of surrogate workflow on Dask.
    SLURMCluster: https://jobqueue.dask.org/en/latest/index.html
    """

    def __init__(self, base_run_dir, sampler_args, runner_args, *args, **kwargs):
        self.type = kwargs.get('type')
        self.base_run_dir=base_run_dir
        self.scale_n_jobs = kwargs.get('scale_n_jobs',1)
        self.SLURMcluster_args = kwargs.get('SLURMcluster_args')
        self.LocalCluster_args = kwargs.get('LocalCluster_args')
        sampler_type = sampler_args.pop("type")
        self.sampler = getattr(importlib.import_module(f'enchanted_surrogates.samplers'),sampler_type)(**sampler_args)
        self.block_unitil_cluster_started = kwargs.get('block_unitil_cluster_started', False) # for debugging purposes only
        self.runner_args = runner_args
        self.cluster=None
        self.client=None
        self.expected_number_of_workers = None

    def start_cluster(self, slurm_out_dir=None):
        """ TODO: Docstring """
        print('MAKING CLUSTER')
        worker_logs_dir = None
        if self.SLURMcluster_args:
            slurm_out_dir = os.path.join(self.base_run_dir,'worker_out_DaskExecutor')
            worker_logs_dir = slurm_out_dir
            if not os.path.exists(slurm_out_dir):
                os.makedirs(slurm_out_dir)
            jed = self.SLURMcluster_args.get('job_extra_directives')
            if not jed:
                self.SLURMcluster_args['job_extra_directives']=[f'-o {slurm_out_dir}/%x.%j.out',f'-e {slurm_out_dir}/%x.%j.err']
            else:
                self.SLURMcluster_args['job_extra_directives']+=[f'-o {slurm_out_dir}/%x.%j.out',f'-e {slurm_out_dir}/%x.%j.err']    
            print('FOR WORKER SLURM OUT, SEE:',slurm_out_dir)
            self.cluster = SLURMCluster(**self.SLURMcluster_args)
            self.cluster.scale(self.scale_n_jobs)
            print('THE JOB SCRIPT FOR A WORKER IS:')
            print(self.cluster.job_script())
            
            self.client = Client(self.cluster ,timeout=180)
            print('SCHEDULER ADDRESS',self.cluster.scheduler_address)
            print('DASHBOARD LINK',self.client.dashboard_link)        
            
            if self.block_unitil_cluster_started:
                print('WAIT UNTILL ALL dask-wor JOBS ARE RUNNING')
                self.wait_for_all_dask_jobs_running()
                
                self.expected_number_of_workers = self.scale_n_jobs * int(self.SLURMcluster_args.get('processes',1))
                
        elif self.LocalCluster_args:
            self.cluster = LocalCluster(**self.LocalCluster_args)
            self.client = Client(self.cluster)
            self.expected_number_of_workers = self.LocalCluster_args['n_workers']
            
        if self.block_unitil_cluster_started:
            print(f"Waiting for {self.expected_number_of_workers} workers to start...")
                
            workers = self.client.scheduler_info()["workers"]
            for _ in range(self.expected_number_of_workers+120):
                print(f"Connected to {len(workers)} workers out of expected {self.expected_number_of_workers}.\n")
                workers = self.client.scheduler_info()["workers"]
                if len(workers) == self.expected_number_of_workers:
                    break
                if len(workers) > self.expected_number_of_workers:
                    warnings.warn(f"More workers ({len(workers)}) than expected ({self.expected_number_of_workers})")
                    break
                time.sleep(1)
            
            if len(workers) == 0:
                raise ValueError(f'NO WORKERS SUCCEDED TO START, PLEASE CHECK WORKER SLURM OUT AT: {worker_logs_dir}')
            
            if len(workers) < self.expected_number_of_workers:
                warnings.warn(f'ONLY {len(workers)} out of {self.expected_number_of_workers} EXPECTED WORKERS STARTED. PLEASE CHECK WORKER LOGS AT: {worker_logs_dir}')
            
            print('WORKER INFORMATION:')
            for addr, info in workers.items():
                print(f"Worker {addr}:")
                print(f"  CPUs: {info['nthreads']}")
                print(f"  Memory: {info['memory_limit'] / 1e9:.2f} GB")
                print(f"  Resources: {info.get('resources', {})}\n")

    def wait_for_all_dask_jobs_running(self, poll_interval=1):
        """ TODO: Docstring """
        print("Waiting for all Dask jobs to enter RUNNING state...")

        while True:
            try:
                # Run squeue --me and capture output
                result = subprocess.run(['squeue', '--me'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                output = result.stdout

                # Filter lines containing 'dask-wor'
                dask_lines = [line for line in output.splitlines() if 'dask-wor' in line]

                if not dask_lines:
                    print("No Dask jobs found in queue.")
                    time.sleep(poll_interval)
                    continue

                # Check job states
                all_running = True
                for line in dask_lines:
                    fields = line.split()
                    job_id = fields[0]
                    job_state = fields[4]  # Typically the 5th column is state

                    if job_state != 'R':
                        print('='*100)
                        print('\n'.join(dask_lines))
                        print(f"Job {job_id} is in state {job_state} — waiting...")
                        print('='*100)
                        all_running = False
                        break

                if all_running:
                    print("All Dask jobs are RUNNING.")
                    return

                time.sleep(poll_interval)

            except Exception as e:
                print(f"Error while checking squeue: {e}")
                time.sleep(poll_interval)

    def clean(self):
        self.shutdown_cluster()

    def shutdown_cluster(self):
        """
        This will also shut down the scheduler which may not be desired if the scheduler is controlling other clusters
        to only shutdown the workers see shutdown_workers
        """
        self.client.shutdown()

    def start_runs(self):
        start = time.time()
        print('BASE RUN DIR:', self.base_run_dir)
        if not os.path.exists(self.base_run_dir):
            print('MAKING BASE RUN DIR:',self.base_run_dir)
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
            print('FUTURE RESULT',result, type(result))
            result = {k:[v] for k,v in result.items()}
            dfs.append(pd.DataFrame(result))
        df_dataset = pd.concat(dfs)
        df_dataset.to_csv(os.path.join(self.base_run_dir, 'enchanted_dataset.csv'), index=False)

        print('WALLTIME FOR ENCHANTED SURROGATES:', time.time()-start,'sec')
        print('DATASET IS WRITTEN HERE:',os.path.join(self.base_run_dir, 'enchanted_dataset.csv'))
        print('WRITTING ENCHANTED.FINNISHED FILE, SEE base_run_dir:',self.base_run_dir)
        with open(os.path.join(self.base_run_dir,'ENCHANTED.FINNISHED'), 'w') as file:
            file.write(f'ENCHANTED.FINNISHED, {__class__}')
        print('CLUSTER SHUTDOWN')
        self.shutdown_cluster()


    def submit_batch(self, samples):
        """ TODO: Docstring """
        futures = []
        for sample_params in samples:
            sample_run_dir = os.path.join(self.base_run_dir, str(uuid.uuid4()))  # TODO. uuid.uuid should probably have a random seed ? 
            new_future = self.client.submit(
                run_simulation_task, self.runner_args, sample_run_dir, sample_params
            )
            futures.append(new_future)
        return futures
