# executors/base.py
import os
import shutil
from abc import ABC
import uuid
from dask.distributed import as_completed
import runners
from common import S


def run_simulation_task(runner_args, params_from_sampler, base_run_dir):
    print("Making Run dir")
    run_dir = os.path.join(base_run_dir, str(uuid.uuid4()))
    os.mkdir(run_dir)
    runner = getattr(runners, runner_args["type"])(**runner_args)
    result = runner.single_code_run(params_from_sampler, run_dir)
    return result, params_from_sampler


class Executor(ABC):
    def __init__(
        self, sampler, runner_args, base_run_dir, config_filepath, *args, **kwargs
    ):
        print("Starting Setup")
        self.sampler = sampler  # kwargs.get('sampler')
        self.runner_args = runner_args  # kwargs.get('runner_args')
        self.base_run_dir = base_run_dir  # , kwargs.get('base_run_dir')
        self.max_samples = self.sampler.total_budget
        self.config_filepath = config_filepath  # kwargs.get('config_filepath')
        self.client = None
        print(config_filepath)
        print(f"Making directory of simulations at: {self.base_run_dir}")
        os.makedirs(self.base_run_dir, exist_ok=True)

        print("Base Executor Initialization")

        shutil.copyfile(config_filepath, os.path.join(self.base_run_dir, "CONFIG.yaml"))

    def start_runs(self):
        sampler_interface = self.sampler.sampler_interface
        print(100 * "=")
        print("Starting Database generation")
        print("Creating initial runs")
        futures = []

        initial_parameters = self.sampler.get_initial_parameters()

        for params in initial_parameters:
            new_future = self.client.submit(
                run_simulation_task, self.runner_args, params, self.base_run_dir
            )
            futures.append(new_future)

        print("Starting search")
        seq = as_completed(futures)
        completed = 0
        for future in seq:
            res = future.result()
            completed += 1
            print(res, completed)
            # TODO: is this waiting for an open node or are we just
            # pushing to queue?
            if self.max_samples > completed:
                # TODO: pass the previous result and parameters.. (Active Learning )
                if sampler_interface in [S.BATCH]: 
                    param_list = self.sampler.get_next_parameter() 
                    for params in param_list: 
                        new_future = self.client.submit(
                            run_simulation_task, self.runner_args, params, self.base_run_dir
                        )
                    seq.add(new_future)
                elif sampler_interface in [S.SEQUENTIAL]: 
                    params = self.sampler.get_next_parameter()
                    if params is None:  # This is hacky
                        continue
                    else:
                        new_future = self.client.submit(
                            run_simulation_task, self.runner_args, params, self.base_run_dir
                        )
                        seq.add(new_future)
