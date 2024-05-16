# executors/DaskExecutor.py

from dask.distributed import Client, as_completed, wait
from .base import Executor, run_simulation_task, run_train_model
from common import S
from torch.utils.data import Dataset


class LocalDaskExecutor(Executor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print('Beginning local Cluster Generation')

        # calling Client with no arguments creates a local cluster
        # it is possible to add arguments like:
        # n_workers=2, threads_per_worker=4
        self.client = Client(timeout=60)
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
                print(seq)
                param_list = self.sampler.get_next_parameter()
                futures = []
                for params in param_list: 
                    new_future = self.client.submit(
                        run_simulation_task, self.runner_args, params, self.base_run_dir
                    )
                    futures.append(new_future)
                completed += self.sampler.batch_size
        elif sampler_interface in [S.ACTIVE, S.ACTIVEDB]:
            
            while completed < self.max_samples: 
                # NOTE: ------ Run Samples ------
                seq = wait(futures) # outputs should be a list of tuples we ignore in this case

                # NOTE: ------ Collect outputs ------
                outputs = []
                for res in seq.done:
                    outputs.append(res.result())
                outputs = self.sampler.collect_batch_results(outputs) # TODO: this should probably be in parser
                # outputs is a tensor of the batch - it's the training data (for the active learning)
                # for static pool it outputs the idxs that need to be appended/deleted

                # NOTE: ------ Update the pool and training data from outputs ------
                self.sampler.parser.update_pool_and_train(outputs)


                train, valid = self.sampler.parser.get_train_valid()
                x_train, y_train = train[:, self.sampler.parser.input_col_idxs], train[:,self.sampler.parser.output_col_idxs]
                x_valid, y_valid = valid[:, self.sampler.parser.input_col_idxs], valid[:,self.sampler.parser.output_col_idxs] 
                train_data, valid_data = (x_train, y_train), (x_valid, y_valid)

                # NOTE: ------ Submit model training job ------

                # model = Regressor(self.sampler.model_kwargs) # 
                # we need to figure out how to handle multiple regressors with one output each
                # train = Dataset(train[:,self.sampler.parser.input_col_idxs], train[:,self.sampler.parser.output_col_idxs])
                # valid = Dataset(valid[:,self.sampler.parser.input_col_idxs], valid[:,self.sampler.parser.output_col_idxs])
                
                new_model_training = self.client.submit(
                    run_train_model, self.sampler.model_kwargs, train_data, valid_data, self.sampler.train_kwargs
                )

                train_model_run = wait([new_model_training])

                # NOTE: ------ Collect model training job ------
                trained_model_res = train_model_run.done[0]
                _, _, trained_model = trained_model_res
                
                # NOTE: ------ Check if out of budget ------
                # TODO: We should break likley much earlier? if completed greator than budget 
                completed += self.sampler.batch_size

                # NOTE: ------ Do active learning sampling ------
                param_list = self.sampler.get_next_parameter(trained_model) 

                # NOTE: ------ Prepare next simulator runs ------
                futures = []
                for params in param_list: 
                    new_future = self.client.submit(
                        run_simulation_task, self.runner_args, params, self.base_run_dir
                    )
                    futures.append(new_future)
            self.sampler.dump_results()