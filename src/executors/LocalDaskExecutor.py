# executors/LocalDaskExecutor.py
# Local version (without SLURM)
import time 
from dask.distributed import Client, as_completed, wait
from common import S
from nn.networks import run_train_model, create_model
from .base import Executor, run_simulation_task


class LocalDaskExecutor(Executor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print('Beginning local Cluster Generation')

        self.client = Client(timeout=60)
        self.simulator_client = self.client
        self.surrogate_client = self.client
        self.clients = [self.client]
        print('Finished Setup')

    def submit_batch_of_params(self, param_list: list[dict]) -> list:
        futures = []
        for params in param_list:
            new_future = self.simulator_client.submit(
                run_simulation_task, self.runner_args, params, self.base_run_dir
            )
            futures.append(new_future)
        return futures

    def start_runs(self):
        sampler_interface = self.sampler.sampler_interface
        print(100 * "=")

        print("Creating initial runs")
        initial_parameters = self.sampler.get_initial_parameters()
        futures = self.submit_batch_of_params(initial_parameters)

        completed = 0

        if sampler_interface in [S.SEQUENTIAL]:
            seq = as_completed(futures)
            for future in seq:
                res = future.result()
                completed += 1
                if self.max_samples > completed:
                    params = self.sampler.get_next_parameter()
                    new_future = self.simulator_client.submit(
                        run_simulation_task, self.runner_args, params, self.base_run_dir
                    )
                    seq.add(new_future)

        elif sampler_interface in [S.BATCH]:
            while completed < self.max_samples:
                seq = wait(futures)
                param_list = self.sampler.get_next_parameter()
                futures = self.submit_batch_of_params(param_list)
                completed += len(futures)
                print('BATCH SAMPLER', 20*'=', completed, 20*'=')

        elif sampler_interface in [S.ACTIVE, S.ACTIVEDB]:
            iterations = 0
            while True:
                print(20*'=', f'Iteration {iterations};', f'samples collected: {completed}', 20*'=')
                # NOTE: ------ Run Samples and block until completed ------
                seq = wait(futures) # outputs should be a list of tuples we ignore in this case

                # NOTE: ------ Collect outputs ------
                outputs = []
                for res in seq.done:
                    outputs.append(res.result())
                outputs = self.sampler.collect_batch_results(outputs) # TODO: this should probably be in parser
                # outputs is a tensor of the batch - it's the training data (for the active learning)
                # for static pool it outputs the idxs that need to be appended/deleted
                
                # NOTE: ------ Check if out of budget ------
                completed += len(outputs)
                if completed > self.max_samples:
                    print(self.max_samples, completed)
                    break
                
                # NOTE: ------ Update the pool and training data from outputs ------
                self.sampler.parser.update_pool_and_train(outputs)

                print('Updated Pool size: ', len(self.sampler.parser.pool), 'Train size: ', len(self.sampler.parser.train))

                # NOTE: Collect next data for training
                train_data, valid_data = self.sampler.parser.get_train_valid_datasplit()
                self.sampler.model_kwargs['input_dim'] = train_data[0].shape[-1]
                self.sampler.model_kwargs['output_dim'] = train_data[1].shape[-1]

                # NOTE: ------ Submit model training job ------
                print('Going to training with ', 'X_tr', train_data[0].shape, 'Y_tr', train_data[1].shape, 'X_vl', valid_data[0].shape,'Y_vl', valid_data[1].shape)
                
                # TODO: write the data instead! 
                time_starttrain = time.time()
                new_model_training = self.surrogate_client.submit(
                    run_train_model, self.sampler.model_kwargs, train_data, valid_data, self.sampler.train_kwargs
                )

                train_model_run = wait([new_model_training])
                print(f'Elapsed train task time: {time.time() - time_starttrain}')

                # NOTE: ------ Collect model training job ------
                trained_model_res = [res.result() for res in train_model_run.done]
                metrics, trained_model_state_dict = trained_model_res[0]
                for metric_name, metric_vals in metrics.items(): 
                    print(f'\n{metric_name}: min: {min(metric_vals)}, max: {max(metric_vals)} @ epoch {metric_vals.index(min(metric_vals))}\n', metric_vals)
                
                self.sampler.update_metrics(metrics)
                # NOTE: ------ Do active learning sampling ------
                # NOTE: Dask doesn't like returning the model, something with pickling, and we have to use the state dict

                example_model = create_model(self.sampler.model_kwargs) # nn.Sequential(nn.Linear(4, 100), nn.ReLU(), nn.Linear(100, 100), nn.ReLU(), nn.Linear(100, 1))
                example_model.load_state_dict(trained_model_state_dict)
                param_list = self.sampler.get_next_parameter(example_model)
                
                # NOTE: ------ Prepare next simulator runs ----
                futures = self.submit_batch_of_params(param_list)
                iterations += 1
                
            self.sampler.dump_results(base_run_dir = self.base_run_dir)