# executors/DaskExecutor.py

from dask.distributed import Client, as_completed, wait
from .base import Executor, run_simulation_task, run_train_model
from common import S

class LocalDaskExecutor(Executor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print('Beginning local Cluster Generation')

        # calling Client with no arguments creates a local cluster
        # it is possible to add arguments like:
        # n_workers=2, threads_per_worker=4
        self.client = Client()
        self.clients.append(self.client)
        print('Finished Setup')

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

        completed = 0
        print("Starting search")
        if sampler_interface in [S.SEQUENTIAL]:
            seq = as_completed(futures)
            for future in seq:
                res = future.result()
                completed += 1
                print(res, completed)
                if self.max_samples > completed:
                    params = self.sampler.get_next_parameter()
                    if params is None:  # This is hacky
                        continue
                    else:
                        new_future = self.client.submit(
                            run_simulation_task, self.runner_args, params, self.base_run_dir
                        )
                        seq.add(new_future)
        elif sampler_interface in [S.BATCH]:
            while completed < self.max_samples: 
                seq = wait(futures)
                param_list = self.sampler.get_next_parameter() 
                futures = []
                for params in param_list: 
                    new_future = self.client.submit(
                        run_simulation_task, self.runner_args, params, self.base_run_dir
                    )
                    futures.append(new_future)
        elif sampler_interface in [S.ACTIVE, S.ACTIVEDB]: 
            wait(futures)
            # Do the active learning step and model training
            train, valid = self.sampler.get_train_valid()
            # we need to figure out how to handle multiple regressors with one output each
            train = Dataset(train)
            valid = Dataset(train)
                
            new_model_training = self.client.submit(
                run_train_model, train, valid
            )
            trained_model = wait(new_model_training)

            param_list = self.sampler.get_next_parameter(train, valid, ) 
            futures = []
            for params in param_list: 
                new_future = self.client.submit(
                    run_simulation_task, self.runner_args, params, self.base_run_dir
                )
                futures.append(new_future)

            


            
                        

        # seq = as_completed(futures)
        
        # §for future in seq:
        # §    res = future.result()
        # §    completed += 1
        # §    print(res, completed)
        # §    # TODO: is this waiting for an open node or are we just
        # §    # pushing to queue?
        # §    if self.max_samples > completed:
        # §        # TODO: pass the previous result and parameters.. (Active Learning )
        # §        if sampler_interface in [S.BATCH]: 
        # §            param_list = self.sampler.get_next_parameter() 
        # §            for params in param_list: 
        # §                new_future = self.client.submit(
        # §                    run_simulation_task, self.runner_args, params, self.base_run_dir
        # §                )
        # §            seq.add(new_future)
        # §        elif sampler_interface in [S.SEQUENTIAL]: 
        # §            params = self.sampler.get_next_parameter()
        # §            if params is None:  # This is hacky
        # §                continue
        # §            else:
        # §                new_future = self.client.submit(
        # §                    run_simulation_task, self.runner_args, params, self.base_run_dir
        # §                )
        # §                seq.add(new_future)