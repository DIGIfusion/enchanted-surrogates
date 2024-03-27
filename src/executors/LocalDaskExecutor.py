# executors/DaskExecutor.py

import os
from dask.distributed import Client, as_completed
# from dask_jobqueue import SLURMCluster
import uuid
import runners
# from dask import delayed, compute


def run_simulation_task(runner_args, params_from_sampler, base_run_dir):
    print("Making Run dir")
    run_dir = os.path.join(base_run_dir, str(uuid.uuid4()))
    os.mkdir(run_dir)
    runner = getattr(runners, runner_args['type'])(**runner_args)
    result = runner.single_code_run(params_from_sampler, run_dir)
    return result, params_from_sampler


class LocalDaskExecutor:
    def __init__(
            self, sampler, runner_args, base_run_dir: str, worker_args: dict,
            num_workers: int):
        print("Starting Setup")
        self.sampler = sampler
        self.runner_args = runner_args
        self.base_run_dir = base_run_dir
        self.max_samples = sampler.num_samples
        self.num_workers = num_workers

        print(f'Making directory of simulations at: {self.base_run_dir}')
        os.makedirs(base_run_dir, exist_ok=True)

        print('Beginning local Cluster Generation')

        # calling Client with no arguments creates a local cluster
        # it is possible to add arguments like:
        # n_workers=2, threads_per_worker=4
        self.client = Client()
        print('Finished Setup')

    def start_runs(self):
        print(100*'=')
        print('Starting Database generation')
        print('Creating initial runs')
        futures = []

        # TODO: implement get_initial_parameters() from sampler
        for _ in range(5):
            params = self.sampler.get_next_parameter()
            new_future = self.client.submit(
                run_simulation_task, self.runner_args, params,
                self.base_run_dir)
            futures.append(new_future)

        print('Starting search')
        seq = as_completed(futures)
        completed = 0
        for future in seq:
            res = future.result()
            completed += 1
            print(res, completed)
            # TODO: is this waiting for an open node or are we just pushing to
            #  queue?
            if self.max_samples > completed:
                # TODO: pass the previous result and parameters..
                params = self.sampler.get_next_parameter()
                new_future = self.client.submit(
                    run_simulation_task, self.runner_args, params,
                    self.base_run_dir)
                seq.add(new_future)
