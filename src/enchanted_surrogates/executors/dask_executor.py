import os
from .base_executor import Executor
from dask.distributed import Client, as_completed, wait, LocalCluster
from dask.distributed import print as dask_print
from enchanted_surrogates.executors import simulation_task
# Patch print inside the module if it uses bare `print()` calls
simulation_task.print = dask_print
# Override local print
print = dask_print
# Alias the task function
run_simulation_task = simulation_task.run_simulation_task

from dask_jobqueue import SLURMCluster
from dask.distributed import LocalCluster
from enchanted_surrogates.utils.precise_imports import import_sampler
import subprocess
import time
import warnings
import importlib
import uuid
import pandas as pd

class DaskExecutor(Executor):
    """
    Handles execution of surrogate workflow on Dask.
    Supports both SLURMCluster and LocalCluster for distributed task execution.
    SLURMCluster: https://jobqueue.dask.org/en/latest/index.html
    """

    def __init__(self, base_run_dir, sampler_kwargs, runner_kwargs, *args, **kwargs):
        """
        Initializes the DaskExecutor.

        Args:
            base_run_dir (str): Base directory for storing run outputs.
            sampler_kwargs (dict): Arguments for the sampler, including its type.
            runner_kwargs (dict): Arguments for the runner.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments, including:
                - type (str): Type of executor.
                - scale_n_jobs (int): Number of jobs to scale the cluster to.
                - SLURMcluster_args (dict): Arguments for SLURMCluster.
                - LocalCluster_args (dict): Arguments for LocalCluster.
                - block_unitil_cluster_started (bool): Whether to block until the cluster is fully started.
        """
        print('INITIALISING DASK EXECUTOR')
        self.type = kwargs.get('type')
        self.base_run_dir=base_run_dir
        self.scale_n_jobs = kwargs.get('scale_n_jobs',1)
        self.SLURMcluster_args = kwargs.get('SLURMcluster_args')
        self.LocalCluster_args = kwargs.get('LocalCluster_args')
        sampler_type = sampler_kwargs.pop("type")
        self.sampler = import_sampler(type=sampler_type, sampler_kwargs=sampler_kwargs) #getattr(importlib.import_module(f'enchanted_surrogates.samplers'),sampler_type)(**sampler_args)
        
        self.block_unitil_cluster_started = kwargs.get('block_unitil_cluster_started', False) # for debugging purposes only
        self.runner_args = runner_kwargs
        self.cluster=None
        self.client=None
        self.expected_number_of_workers = None

    def start_cluster(self, slurm_out_dir=None):
        """
        Starts a Dask cluster using either SLURMCluster or LocalCluster.

        If SLURMCluster is used, it sets up SLURM-specific configurations, including output directories
        for worker logs. If LocalCluster is used, it initializes a local Dask cluster.

        Args:
            slurm_out_dir (str, optional): Directory for SLURM output logs. Defaults to None.

        Raises:
            ValueError: If no workers are successfully started.
            Warning: If fewer workers than expected are started.
        """
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
                workers = self.client.scheduler_info()["workers"]
                print(f"Connected to {len(workers)} workers out of expected {self.expected_number_of_workers}.\n")
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
        """
        Waits for all Dask jobs submitted to SLURM to reach the RUNNING state.

        This method repeatedly checks the SLURM job queue for jobs with the prefix 'dask-wor'.
        If any job is not in the RUNNING state, it waits and retries until all jobs are running.

        Args:
            poll_interval (int, optional): Time interval (in seconds) between checks. Defaults to 1.

        Raises:
            Exception: If an error occurs while checking the SLURM queue.
        """
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
        """
        Cleans up resources by shutting down the Dask cluster.

        This method is intended to be called when the executor is no longer needed.
        """
        self.shutdown_cluster()

    def shutdown_cluster(self):
        """
        Shuts down the Dask cluster, including the scheduler and workers.

        Note:
            This will also shut down the scheduler, which may not be desired if the scheduler
            is controlling other clusters. To only shut down the workers, use a different method.
        """
        self.client.shutdown()

    def start_runs(self):
        """
        Starts the execution of simulation tasks using the configured Dask cluster.

        This method initializes the base run directory, checks for existing data to avoid overwrites,
        and submits tasks to the Dask cluster in batches. It collects results from completed tasks
        and writes them to a CSV file.

        Raises:
            FileExistsError: If the base run directory contains a file indicating a completed run.
        """
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
        print('CLUSTER STARTED')
        all_futures = []

        while self.sampler.has_budget:
            samples = self.sampler.get_next_samples()
            futures = self.submit_batch(samples)
            all_futures.extend(futures)
        print(f'{len(all_futures)} FUTURES HAVE BEEN SENT')
        dfs = []
        num_success = 0
        for i, future in enumerate(as_completed(all_futures)):
            result = future.result()
            if result['success'] == True:
                num_success += 1
            # print('FUTURE RESULT',result, type(result))
            result = {k:[v] for k,v in result.items()}
            dfs.append(pd.DataFrame(result))
            print(f"[{i}/{len(all_futures)}] Futures Completed ({(i/len(all_futures))*100:.1f}%)","|",f"[{num_success}/{len(all_futures)}] Futures Succeded ({(num_success/len(all_futures))*100:.1f}%)")
            print('_'*100)
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
        """
        Submits a batch of simulation tasks to the Dask cluster.

        Each task is submitted with its own unique run directory. The tasks are executed
        asynchronously, and their futures are returned for tracking.

        Args:
            samples (list): List of sample parameters for the simulation tasks.

        Returns:
            list: List of futures representing the submitted tasks.
        """
        futures = []
        for sample_params in samples:
            sample_run_dir = os.path.join(self.base_run_dir, str(uuid.uuid4()))  # TODO. uuid.uuid should probably have a random seed ? 
            new_future = self.client.submit(
                run_simulation_task, self.runner_args, sample_run_dir, sample_params
            )
            futures.append(new_future)
        return futures
