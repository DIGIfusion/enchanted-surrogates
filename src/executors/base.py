# executors/base.py

from abc import ABC
import uuid
import runners
from dask.distributed import as_completed
import os


def run_simulation_task(runner_args, params_from_sampler, base_run_dir):
    print("Making Run dir")
    run_dir = os.path.join(base_run_dir, str(uuid.uuid4()))
    os.mkdir(run_dir)
    runner = getattr(runners, runner_args['type'])(**runner_args)
    result = runner.single_code_run(params_from_sampler, run_dir)
    return result, params_from_sampler


class Executor(ABC):
    def start_runs(self):
        print(100*'=')
        print('Starting Database generation')
        print('Creating initial runs')
        futures = []

        # TODO: implement get_initial_parameters() from sampler
        for _ in range(self.sampler.num_initial_points):
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
            # TODO: is this waiting for an open node or are we just
            # pushing to queue?
            if self.max_samples > completed:
                # TODO: pass the previous result and parameters..
                # For active learning
                params = self.sampler.get_next_parameter()
                if params is None:  # This is hacky
                    continue
                else:
                    new_future = self.client.submit(
                        run_simulation_task, self.runner_args,
                        params, self.base_run_dir)
                    seq.add(new_future)
