import os
import subprocess
import time
import pandas as pd

from dask_jobqueue import SLURMCluster
from dask.distributed import LocalCluster
from dask.distributed import Client, as_completed, wait, LocalCluster, get_worker, get_client

from .base_executor import Executor
from enchanted_surrogates.utils.logger import get_logger, setup_logging

from enchanted_surrogates.executors import simulation_task
from enchanted_surrogates.utils.make_run_dir import make_run_dir
from enchanted_surrogates.utils.precise_imports import import_sampler

from dask.distributed import WorkerPlugin

# This setups logging for every worker
class LogPlugin(WorkerPlugin):
    def __init__(self, log_level, log_dir):
        self.log_level = log_level
        self.log_dir = log_dir

    def setup(self, worker):
        setup_logging(self.log_level, self.log_dir, f"{worker.id}.log")


log = get_logger(__name__)

# Alias the task function
run_simulation_task = simulation_task.run_simulation_task

class DaskExecutor(Executor):
    """
    Handles execution of surrogate workflow on Dask.
    Supports both SLURMCluster and LocalCluster for distributed task execution.
    SLURMCluster: https://jobqueue.dask.org/en/latest/index.html
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the DaskExecutor.

        Args:
            base_run_dir (str): Base directory for storing run outputs.
            sampler_config (dict): Arguments for the sampler, including its type.
            runner_config (dict): Arguments for the runner.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments, including:
                - type (str): Type of executor.
                - scale_n_jobs (int): Number of jobs to scale the cluster to.
                - SLURMcluster_config (dict): Arguments for SLURMCluster.
                - LocalCluster_config (dict): Arguments for LocalCluster.
                - block_unitil_cluster_started (bool): Whether to block until the cluster is fully started.
        """
        super().__init__(*args, **kwargs)
        log.info('INITIALISING DASK EXECUTOR')

        self.type = kwargs.get('type')
        if self.sampler_config:
            self.sampler_type = self.sampler_config.pop("type")
            self.sampler = import_sampler(
                sampler_type=self.sampler_type, sampler_config=self.sampler_config)
        self.scale_n_jobs = kwargs.get('scale_n_jobs', 1)
        self.timeout = kwargs.get('timeout', 1e10)
        self.SLURMcluster_config = kwargs.get('SLURMcluster_config')
        self.LocalCluster_config = kwargs.get('LocalCluster_config')
        self.block_until_cluster_started = kwargs.get('block_until_cluster_started', False)  # for debugging purposes only
        self.cluster = None
        self.client = None
        self.expected_number_of_workers = None

        # Store log level and log dir
        self.log_level = kwargs.get('log_level')        
        self.log_dir = kwargs.get('log_dir')        

    def find_line_in_seff_output(self, lines, entry):
        """
        Helper function to quickly find the required line in the seff output

        Params:
            lines (list): list of lines
            entry (str): the entry that is being looked for
        Returns:
            str: time or percentage from the corresponding line, defaults to ""
        """
        return next((line.replace(entry,"").strip() for line in lines if line.startswith(entry)),"")


    def get_slurm_usage_info(self, job_id=None):
        """
        Params:
            job_id (list[int]): If you wish to only find the slurm usage info from one specific job pass this
        Returns:
            list: dictionary containing the output info from running seff
        """
        job_ids = job_id if job_id else self.get_all_dask_job_ids()
        job_info = []

        for job_id in job_ids:
            try:
                result = subprocess.run(['seff', job_id], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                output = result.stdout
                seff_lines = output.splitlines()

                if len(output.strip()) == 0:
                    continue
                
                cpu_time = self.find_line_in_seff_output(seff_lines, "CPU Utilized:")
                cpu_efficiency = self.find_line_in_seff_output(seff_lines, "CPU Efficiency:")
                memory_used = self.find_line_in_seff_output(seff_lines, "Memory Utilized:")
                memory_efficiency = self.find_line_in_seff_output(seff_lines, "Memory Efficiency:")

                hours, minutes, seconds = map(int, cpu_time.split(":"))
                cpu_secs = hours * 3600 + minutes * 60 + seconds
                
                job_info.append({
                    'cpu_time': cpu_time,
                    'cpu_time_seconds': cpu_secs,
                    'cpu_efficiency': cpu_efficiency,
                    'memory_efficiency': memory_efficiency,
                    'memory_used': memory_used,
                    'job_id': job_id
                })
            except Exception as e:
                log.error(f"Error fetching SLURM resource usage for job {job_id}: {e}")


        return job_info

    def get_all_dask_job_ids(self):
        """
        Runs squeue to figure out all jobs from the cluster
        Returns:
            list: A list of Dask job IDs.
        """
        try:
            jobs = []
            result = subprocess.run(['squeue', '--me'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            dask_lines = [line for line in result.stdout.splitlines() if 'sys/dash' not in line and "enc_dask_worker" in line]

            if not dask_lines:
                log.debug("No Dask jobs found in queue.")
                return []

            for line in dask_lines:
                fields = line.split()
                jobs.append(fields[0])

            return jobs

        except Exception as e:
            log.warning(f"Error while checking squeue: {e}")
            return []

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
        log.info('Creating a cluster...')
        worker_logs_dir = None

        if self.SLURMcluster_config:
            self.expected_number_of_workers = self.scale_n_jobs * int(self.SLURMcluster_config.get('processes',1))

            if not slurm_out_dir:
                slurm_out_dir = os.path.join(self.base_run_dir,'worker_out_DaskExecutor')
            worker_logs_dir = slurm_out_dir
            if not os.path.exists(slurm_out_dir):
                os.makedirs(slurm_out_dir)
            jed = self.SLURMcluster_config.get('job_extra_directives')
            if not jed:
                self.SLURMcluster_config['job_extra_directives']=[f'-o {slurm_out_dir}/%x.%j.out',f'-e {slurm_out_dir}/%x.%j.err', '-J enc_dask_worker']
            else:
                self.SLURMcluster_config['job_extra_directives']+=[f'-o {slurm_out_dir}/%x.%j.out',f'-e {slurm_out_dir}/%x.%j.err', '-J enc_dask_worker']
            log.info(f'Output of SLURM workers saved in: {slurm_out_dir}')
            self.cluster = SLURMCluster(silence_logs=False, **self.SLURMcluster_config)
            self.cluster.scale(self.scale_n_jobs)
            log.debug(f'The job script for a worker is:\n{self.cluster.job_script()}')
            
            self.client = Client(self.cluster ,timeout=180)
            log.info(f'SCHEDULER ADDRESS: {self.cluster.scheduler_address}')
            log.info(f'DASHBOARD LINK: {self.client.dashboard_link}')        
            
            if self.block_until_cluster_started:
                log.info('WAIT UNTILL ALL dask-wor JOBS ARE RUNNING')
                self.wait_for_all_dask_jobs_running()
                                
        elif self.LocalCluster_config:
            self.expected_number_of_workers = self.LocalCluster_config['n_workers']
            self.cluster = LocalCluster(silence_logs=False, **self.LocalCluster_config)
            self.client = Client(self.cluster)

        # Register the log plugin
        plugin = LogPlugin(self.log_level, self.log_dir)
        self.client.register_plugin(plugin)

        if self.block_until_cluster_started:
            log.info(f"Waiting for {self.expected_number_of_workers} workers to start...")
            for i in range(1,self.expected_number_of_workers+2):
                if i == self.expected_number_of_workers+1:
                    timeout_ = 3
                    try:
                        self.client.wait_for_workers(i, timeout=timeout_)
                        log.warning(f'MORE WORKERS WERE STARTED THAN THE EXPECTED {self.expected_number_of_workers}')
                    except TimeoutError:
                        log.error(f"IN {timeout_} SEC NO UNEXPECTED WORKERS WERE STARTED.\n")
                else:
                    self.client.wait_for_workers(i, timeout=self.expected_number_of_workers+120)
                    log.info(f"Connected to {i} workers out of expected {self.expected_number_of_workers}.\n")
                
            workers = self.client.scheduler_info()["workers"]            
            log.info('SOME WORKER INFORMATION:')
            for addr, info in workers.items():
                log.info(f"Worker {addr}:")
                log.info(f"  CPUs: {info['nthreads']}")
                log.info(f"  Memory: {info['memory_limit'] / 1e9:.2f} GB")
                log.info(f"  Resources: {info.get('resources', {})}\n")


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
        log.info("Waiting for all Dask jobs to enter RUNNING state...")

        while True:
            try:
                # Run squeue --me and capture output
                result = subprocess.run(['squeue', '--me'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                output = result.stdout

                # Filter lines containing 'dask-wor'
                dask_lines = [line for line in output.splitlines() if 'dask-wor' in line]

                if not dask_lines:
                    log.info("No Dask jobs found in queue.")
                    time.sleep(poll_interval)
                    continue

                # Check job states
                all_running = True
                for line in dask_lines:
                    fields = line.split()
                    job_id = fields[0]
                    job_state = fields[4]  # Typically the 5th column is state

                    if job_state == 'PD':
                        log.info('='*100)
                        log.info('\n'.join(dask_lines))
                        log.info(f"Job {job_id} is in state {job_state} — waiting...")
                        log.info('='*100)
                        all_running = False
                        break

                if all_running:
                    log.info("All Dask jobs are RUNNING.")
                    return

                time.sleep(poll_interval)

            except Exception as e:
                log.error(f"Error while checking squeue: {e}")
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

    def count_alive_workers(self):
        num_alive_workers = None
        if self.client:
            count = 1
            while not num_alive_workers:
                try:
                    self.client.wait_for_workers(count, timeout=0.1)
                except TimeoutError:
                    num_alive_workers = count - 1
                count += 1
        return num_alive_workers

    def scale(self, num_workers):
        if self.count_alive_workers == self.scale_n_jobs:
            self.cluster.scale(num_workers)
        self.scale_n_jobs = num_workers

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
        assert self.base_run_dir
        assert self.sampler
        if not os.path.exists(self.base_run_dir):
            log.info(f'MAKING BASE RUN DIR: {self.base_run_dir}')
            os.makedirs(self.base_run_dir)

        if self.sampler_type not in {'BayesianOptimizationSampler'}:
            if os.path.exists(os.path.join(self.base_run_dir, 'ENCHANTED.FINISHED')):
                raise FileExistsError(f'''The file: {self.base_run_dir}/ENCHANTED.FINISHED, exists.
                                      This signifies that there is already data in this folder. 
                                      Aborting to avoid accidental data mixing.''' )

        log.info(f"STARTING RUNS FOR RUNNER {self.runner_config['type']}, FROM WITHIN A {__class__}")

        if not self.client:
            self.start_cluster()

        all_job_ids = self.get_all_dask_job_ids()
        log.info('CLUSTER STARTED')
        all_futures = []

        while self.sampler.has_budget:
            samples = self.sampler.get_next_samples()
            futures = self.submit_batch(samples)
            if self.sampler_type in {'BayesianOptimizationSampler'}:
                try: 
                    wait(futures, timeout=self.timeout)
                except:
                    log.error("TIMEOUT OF SOME OF THE SAMPLES")
            all_futures.extend(futures)
        log.info(f'{len(all_futures)} FUTURES HAVE BEEN SENT')
        dfs = []
        num_success = 0
        if self.sampler_type in {'BayesianOptimizationSampler'}:
            try: 
                wait(futures, timeout=self.timeout)
                if self.sampler.plot_GPR_flag:
                    # Build the result dictionary and set plot frequency to 1 to 
                    # plot the GPR for the final state of the optimization.
                    # Plotting is presently implemented in the train_surrogate function.
                    self.sampler.build_result_dictionary(self.sampler.base_run_dir)
                    self.sampler.plot_frequency = 1
                    self.sampler.train_surrogate()
            except:
                log.error("Timeout of some of the samples")
        else:
            for i, future in enumerate(as_completed(all_futures)):
                result = future.result()
                if result['success'] == True:
                    num_success += 1
                # print('FUTURE RESULT',result, type(result))
                result = {k:[v] for k,v in result.items()}
                dfs.append(pd.DataFrame(result))
                info_msg = (
                    f"[{i}/{len(all_futures)}] Futures Completed ({(i/len(all_futures))*100:.1f}%) | "
                    f"[{num_success}/{len(all_futures)}] Futures Succeeded ({(num_success/len(all_futures))*100:.1f}%)"
                )
                log.info(info_msg)
            df_dataset = pd.concat(dfs)
            df_dataset.to_csv(os.path.join(self.base_run_dir, 'enchanted_dataset.csv'), index=False)
            log.info(f'DATASET IS WRITTEN HERE: {os.path.join(self.base_run_dir, "enchanted_dataset.csv")}')

        log.debug(f'WRITTING ENCHANTED.FINISHED FILE, SEE base_run_dir: {self.base_run_dir}')
        with open(os.path.join(self.base_run_dir,'ENCHANTED.FINISHED'), 'w') as file:
            file.write(f'ENCHANTED.FINISHED, {__class__}')

        job_info = self.get_slurm_usage_info(all_job_ids)
        total_cpu_time = sum(job['cpu_time_seconds'] for job in job_info)/3600
        cpu_ps = subprocess.run(["ps", "--no-headers", "-o", "etimes=", "-p", str(os.getpid())], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if cpu_ps.returncode == 0:
            headnode_secs = int(cpu_ps.stdout.strip())
        else:
            headnode_secs = None
            log.info(f"Fetching head node CPU time failed! STDOUT from ps: {cpu_ps.stdout}")

        log.info(
            "\n======== RUNTIME SUMMARY ========\n"
            f"Walltime (s):            {time.time()-start:.6f}\n"
            f"Total CPU hours used:    {total_cpu_time:.6f}\n"
            f"Head node CPU hours (h): {headnode_secs/3600:.6f}\n"
            "================================="
        )
        log.info('Shutting down cluster.')
        self.shutdown_cluster()


    def submit_batch(self, samples, base_run_dir=None, client=None, include_fut_to_rundir=False):
        """
        Submits a batch of simulation tasks to the Dask cluster.

        Each task is submitted with its own unique run directory. The tasks are executed
        asynchronously, and their futures are returned for tracking.

        Args:
            samples (list): List of sample parameters for the simulation tasks.

        Returns:
            list: List of futures representing the submitted tasks.
        """
        
        if not client:
            client = self.client
        
        if not base_run_dir:
            base_run_dir = self.base_run_dir
        assert base_run_dir
        
        futures = []
        run_dirs = []
        fut_to_rundir = {}
        for sample_params in samples:
            sample_run_dir = make_run_dir(base_run_dir=base_run_dir, prepend=self.runner_config['type']) 
            run_dirs.append(sample_run_dir)
            new_future = client.submit(
                run_simulation_task, self.runner_config, sample_run_dir, sample_params
            )
            futures.append(new_future)
            fut_to_rundir[new_future.key] = sample_run_dir
        p_info = [str(sample_params)+f' | {rd}' for sample_params,rd in zip(samples,run_dirs)]
        log.info(f"{len(futures)} DASK FUTURES HAVE BEEN SUBMITTED FOR RUNNER: {self.runner_config['type']}")
        log.debug("Samples:\n" + "\n".join(p_info))
        
        if include_fut_to_rundir:
            return futures, fut_to_rundir
        else:
            return futures
