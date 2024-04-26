# executors/DaskExecutor.py

from .base import Executor

class GENEScanExecutor(Executor):
    def __init__(self):
        raise NotImplementedError
    
    def start_runs(self):
        print(100 * "=")
        print("Starting Dataset generation")
        print("Creating initial runs")
        raise NotImplementedError
        # # TODO: implement get_initial_parameters() from sampler
        # for _ in range(self.sampler.num_initial_points):
        #     params = self.sampler.get_next_parameter()
        #     new_future = self.client.submit(
        #         self.run_simulation_task, self.runner_args, params, self.base_run_dir
        #     )
        #     futures.append(new_future)

        # print("Starting search")
        # seq = as_completed(futures)
        # completed = 0
        # for future in seq:
        #     res = future.result()
        #     completed += 1
        #     print(res, completed)
        #     # TODO: is this waiting for an open node or are we just
        #     # pushing to queue?
        #     if self.max_samples > completed:
        #         # TODO: pass the previous result and parameters..
        #         # For active learning
        #         params = self.sampler.get_next_parameter()
        #         if params is None:  # This is hacky
        #             continue
        #         else:
        #             new_future = self.client.submit(
        #                 self.run_simulation_task, self.runner_args, params, self.base_run_dir
        #             )
        #             seq.add(new_future)


    
