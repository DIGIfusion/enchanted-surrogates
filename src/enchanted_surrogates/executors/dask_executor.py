"""

## Overview

Handles execution of surrogate workflow on Dask.
Supports both SLURMCluster and LocalCluster for distributed task execution.
SLURMCluster: https://jobqueue.dask.org/en/latest/index.html

---

## Clusters

### Local cluster

Can be used for running on a local machine with multiple cores. Useful for testing or small scale runs.

Arguments:

```
n_workers: 2,
threads_per_worker: 1,
memory_limit: '12GB',
processes: 1
```

Example configuration: /configs/example_dask_local.yaml

### SLURM cluster

Arguments for the SLURM workers.

```
account: 'project_xxx',
queue: 'medium',
cores: 1,
memory: '12GB',
processes: 1,
walltime: '00:20:00',
config_name: 'slurm',
interface: 'ib0',
```

Example configuration: /configs/example_dask_slurm.yaml

### Notes

Other arguments:

```
job_script_prologue: ['module load your-modules-here',],
job_extra_directives: [
    '-o tmp_path_hm/worker_out_MishkaRunner_1/%x.%j.out',
    '-e tmp_path_hm/worker_out_MishkaRunner_1/%x.%j.err'],
```
"""

import os
import subprocess
import sys
import time
import pandas as pd
import logging


from dask_jobqueue import SLURMCluster
from dask.distributed import LocalCluster
from dask.distributed import (
    Client,
    as_completed,
    wait,
    LocalCluster,
    get_worker,
    get_client,
)
from dask.distributed import print as dask_print

from .base_executor import Executor
from enchanted_surrogates.utils.logger import get_logger, setup_logging, LoggerConfig

from enchanted_surrogates.executors import simulation_task
from enchanted_surrogates.utils.make_run_dir import make_run_dir
from enchanted_surrogates.utils.precise_imports import import_sampler

from dask.distributed import WorkerPlugin


# Console log handler for SLURM, uses dask.distributed.print
class SLURMStreamHandler(logging.Handler):
    def __init__(self) -> None:
        logging.Handler.__init__(self)

    def emit(self, record) -> None:
        dask_print(self.formatter.format(record))


class SLURMLogPlugin(WorkerPlugin):
    def __init__(self, config: LoggerConfig):
        self.config = config

    def setup(self, worker):
        log_file = os.path.join(self.config.log_dir, f"{worker.id}.log")
        dask_handler = SLURMStreamHandler()
        setup_logging(self.config, dask_handler, logging.FileHandler(filename=log_file))


class DaskLocalLogPlugin(WorkerPlugin):
    def __init__(self, config: LoggerConfig):
        self.config = config

    def setup(self, worker):
        log_file = os.path.join(self.config.log_dir, f"{worker.id}.log")
        setup_logging(
            self.config,
            logging.StreamHandler(stream=sys.stdout),
            logging.FileHandler(filename=log_file),
        )


log = get_logger(__name__)

# Alias the task function
run_simulation_task = simulation_task.run_simulation_task


class DaskExecutor(Executor):
    def __init__(self, *args, **kwargs):
        """
        Initializes the DaskExecutor.

        Args:
            base_run_dir (str): Base directory for storing run outputs.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments, including:
                - type (str): Type of executor.
                - scale_n_jobs (int): Number of jobs to scale the cluster to.
                - SLURMcluster_config (dict): Arguments for SLURMCluster.
                - LocalCluster_config (dict): Arguments for LocalCluster.
                - block_unitil_cluster_started (bool): Whether to block until the cluster is fully started.
        """
        super().__init__(*args, **kwargs)
        log.info("INITIALISING DASK EXECUTOR")
        self.scale_n_jobs = kwargs.get("scale_n_jobs", 1)
        self.timeout = kwargs.get("timeout", 1e10)
        self.SLURMcluster_config = kwargs.get("SLURMcluster_config")
        self.LocalCluster_config = kwargs.get("LocalCluster_config")
        self.block_until_cluster_started = kwargs.get(
            "block_until_cluster_started", False
        )  # for debugging purposes only
        self.cluster = None
        self.client = None
        self.expected_number_of_workers = None
        self.slurm_job_ids = set()
        self.is_closed = False

    def find_line_in_seff_output(self, lines, entry):
        """
        Helper function to quickly find the required line in the seff output

        Params:
            lines (list): list of lines
            entry (str): the entry that is being looked for
        Returns:
            str: time or percentage from the corresponding line, defaults to ""
        """
        return next(
            (
                line.replace(entry, "").strip()
                for line in lines
                if line.startswith(entry)
            ),
            "",
        )

    def is_running_on_slurm(self):
        """
        Checks if code is running on slurm or locally. This is done via checking if seff exists.
        Retuns:
            bool: true is on slurm false otherwise
        """
        try:
            proc = subprocess.run(
                ["which", "seff"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=1,
            )
        except Exception:
            return False

        return proc.returncode == 0

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
                result = subprocess.run(
                    ["seff", str(job_id)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                output = result.stdout
                seff_lines = output.splitlines()

                if len(output.strip()) == 0:
                    continue

                cpu_time = self.find_line_in_seff_output(seff_lines, "CPU Utilized:")
                cpu_efficiency = self.find_line_in_seff_output(
                    seff_lines, "CPU Efficiency:"
                )
                memory_used = self.find_line_in_seff_output(
                    seff_lines, "Memory Utilized:"
                )
                memory_efficiency = self.find_line_in_seff_output(
                    seff_lines, "Memory Efficiency:"
                )

                hours, minutes, seconds = map(int, cpu_time.split(":"))
                cpu_secs = hours * 3600 + minutes * 60 + seconds

                job_info.append(
                    {
                        "cpu_time": cpu_time,
                        "cpu_time_seconds": cpu_secs,
                        "cpu_efficiency": cpu_efficiency,
                        "memory_efficiency": memory_efficiency,
                        "memory_used": memory_used,
                        "job_id": job_id,
                    }
                )
            except Exception as e:
                if self.is_running_on_slurm():
                    log.error(
                        f"Error fetching SLURM resource usage for job {job_id}: {e}"
                    )
                else:
                    log.error("Not running on SLURM. skipping resources")
                    return [
                        {
                            "cpu_time": "00:00:00",
                            "cpu_time_seconds": 0,
                            "cpu_efficiency": "100%",
                            "memory_efficiency": "100%",
                            "memory_used": "0",
                            "job_id": 0,
                        }
                    ]

        return job_info

    def get_all_dask_job_ids(self):
        """
        Runs squeue to figure out all jobs from the cluster
        Returns:
            list: A list of Dask job IDs.
        """
        try:
            jobs = []
            result = subprocess.run(
                ["squeue", "--me"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            dask_lines = [
                line
                for line in result.stdout.splitlines()
                if "sys/dash" not in line and "enc_dask_worker" in line
            ]

            if not dask_lines:
                log.debug("No Dask jobs found in queue.")
                return []

            for line in dask_lines:
                fields = line.split()
                jobs.append(fields[0])

            return jobs

        except Exception as e:
            if self.is_running_on_slurm():
                log.warning(f"Error while checking squeue: {e}")
            return []

    def start_cluster(self):
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
        log.info("Creating a cluster...")
        slurm_out_dir = LoggerConfig().log_dir

        if self.SLURMcluster_config:
            self.expected_number_of_workers = self.scale_n_jobs * int(
                self.SLURMcluster_config.get("processes", 1)
            )

            log.info(f"Output of SLURM workers saved in: {slurm_out_dir}")
            self.cluster = SLURMCluster(silence_logs=False, **self.SLURMcluster_config)
            self.cluster.scale(self.scale_n_jobs)
            log.debug(f"The job script for a worker is:\n{self.cluster.job_script()}")

            self.client = Client(self.cluster, timeout=180)

            # Register the log plugin
            plugin = SLURMLogPlugin(LoggerConfig())
            self.client.register_plugin(plugin, name="LogPlugin")

            log.info(f"SCHEDULER ADDRESS: {self.cluster.scheduler_address}")
            log.info(f"DASHBOARD LINK: {self.client.dashboard_link}")

            if self.block_until_cluster_started:
                log.info("WAIT UNTILL ALL dask-wor JOBS ARE RUNNING")
                self.wait_for_all_dask_jobs_running()

        elif self.LocalCluster_config:
            self.expected_number_of_workers = self.LocalCluster_config["n_workers"]
            self.cluster = LocalCluster(silence_logs=False, **self.LocalCluster_config)
            self.client = Client(self.cluster)

            # Register the log plugin
            plugin = DaskLocalLogPlugin(LoggerConfig())
            self.client.register_plugin(plugin, name="LogPlugin")

        if self.block_until_cluster_started:
            log.info(
                f"Waiting for {self.expected_number_of_workers} workers to start..."
            )
            for i in range(1, self.expected_number_of_workers + 2):
                if i == self.expected_number_of_workers + 1:
                    timeout_ = 3
                    try:
                        self.client.wait_for_workers(i, timeout=timeout_)
                        log.warning(
                            f"MORE WORKERS WERE STARTED THAN THE EXPECTED {self.expected_number_of_workers}"
                        )
                    except TimeoutError:
                        log.error(
                            f"IN {timeout_} SEC NO UNEXPECTED WORKERS WERE STARTED.\n"
                        )
                else:
                    self.client.wait_for_workers(
                        i, timeout=self.expected_number_of_workers + 120
                    )
                    log.info(
                        f"Connected to {i} workers out of expected {self.expected_number_of_workers}.\n"
                    )

            workers = self.client.scheduler_info()["workers"]
            log.info("SOME WORKER INFORMATION:")
            for addr, info in workers.items():
                log.info(f"Worker {addr}:")
                log.info(f"  CPUs: {info['nthreads']}")
                log.info(f"  Memory: {info['memory_limit'] / 1e9:.2f} GB")
                log.info(f"  Resources: {info.get('resources', {})}\n")

        # Only for SLURM cluster
        if hasattr(self.cluster, "workers"):
            try:
                self.slurm_job_ids.update(self.cluster.workers.keys())
            except Exception:
                pass

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
                result = subprocess.run(
                    ["squeue", "--me"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                output = result.stdout

                # Filter lines containing 'dask-wor'
                dask_lines = [
                    line for line in output.splitlines() if "dask-wor" in line
                ]

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

                    if job_state == "PD":
                        log.info("=" * 100)
                        log.info("\n".join(dask_lines))
                        log.info(f"Job {job_id} is in state {job_state} — waiting...")
                        log.info("=" * 100)
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
        if self.is_closed:
            log.debug("Trying to close cluster that has already been closed!")
            return

        self.shutdown_cluster()
        self.is_closed = True

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

    def execute(self, input: list[(str, dict)], runner_config):
        """
        Starts the execution of simulation tasks using the configured Dask cluster.

        This method initializes the base run directory, checks for existing data to avoid overwrites,
        and submits tasks to the Dask cluster in batches. It collects results from completed tasks
        and writes them to a CSV file.

        Raises:
            FileExistsError: If the base run directory contains a file indicating a completed run.
        """

        if not self.client:
            self.start_cluster()
        log.info("CLUSTER STARTED")

        # keep futures for BayesianOptimizationSampler
        futures = self.submit_batch(input, runner_config)

        dfs = []
        num_success = 0
        total = len(futures)

        log.info(f"Collecting results from {total} futures...")

        for i, future in enumerate(as_completed(futures), start=1):
            try:
                result = future.result()
            except Exception as e:
                log.error(f"[{i}/{total}] Future failed with exception:", e)
                continue

            if isinstance(result, dict) and result.get("success") is True:
                num_success += 1

            try:
                df = pd.DataFrame({k: [v] for k, v in result.items()})
                dfs.append(df)
            except Exception as e:
                log.error("Failed to convert result to DataFrame:", e)
                continue

    def submit_batch(
        self,
        run_dir_sample_pairs,
        runner_config,
        base_run_dir=None,
        client=None,
    ):
        """
        Submits a batch of simulation tasks to the Dask cluster.

        Each task is submitted with its own unique run directory. The tasks are executed
        asynchronously, and their futures are returned for tracking.

        Args:
            run_dir_sample_pairs (list): List of rundir, sample parameters for the simulation tasks.

        Returns:
            list: List of futures representing the submitted tasks.
        """
        if not client:
            client = self.client
        assert client is not None

        futures = []
        for run_dir, sample_params in run_dir_sample_pairs:
            new_future = client.submit(
                run_simulation_task, runner_config, run_dir, sample_params
            )
            futures.append(new_future)

        log.info(
            f"{len(futures)} DASK FUTURES SUBMITTED for runner {runner_config['type']}"
        )
        return futures
