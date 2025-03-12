"""
# executors/DaskExecutor.py

Contains logic for executing surrogate workflow on Dask.

"""

import time
from dask.distributed import Client, as_completed, wait
from dask_jobqueue import SLURMCluster
from nn.networks import load_saved_model, run_train_model
from common import S
from .base import Executor, run_simulation_task
import os


class DaskExecutor(Executor):
    """
    Handles execution of surrogate workflow on Dask.
    SLURMCluster: https://jobqueue.dask.org/en/latest/index.html

    Attributes:
        n_jobs (int): Number of batch jobs to be created by SLURMcluster.
            In dask-jobqueue, a single Job may include one or more Workers.
        worker_args (dict): Dictionary of arguments for configuring worker nodes.
            The arguments depends on the type of Cluster used.
            - n_workers (int): Number of workers
        client (dask.distributed.Client): Dask client for task submission and execution.
        simulator_client (dask.distributed.Client): Client for simulator tasks.
        surrogate_client (dask.distributed.Client): Client for training surrogate models.
        clients (list): List of Dask clients.
    """

    def __init__(self, worker_args: dict, **kwargs):
        super().__init__(**kwargs)
        self.n_jobs: int = kwargs.get("n_jobs", 8)
        self.worker_args: dict = worker_args
        print("Beginning SLURMCluster Generation")
        self.initialize_clients()
        print("Finished Setup")

    def clean(self):
        """
        Preforms cleaning of the dask clients
        """
        for client in self.clients:
            client.close()

    def initialize_clients(self):
        """
        Initializes the clients based on sampler enum,
        general steps are: initialize cluster, scale cluster, initialize client with cluster

        Args:
            worker_args (dict): Dictionary of arguments for configuring worker nodes.
            **kwargs: Additional keyword arguments.
        """
        sampler_type: S = self.sampler.sampler_interface
        print("Initializing DASK clients")
        if self.worker_args.get("local", False):
            # TODO: Increase num parallel workers on local
            # self.simulator_cluster = LocalCluster(**self.worker_args)
            self.client = Client(timeout=60)
            self.simulator_client = self.client
            self.surrogate_client = self.client
            self.clients = [self.client]

        elif sampler_type in [S.SEQUENTIAL, S.BATCH]:
            self.simulator_cluster = SLURMCluster(**self.worker_args)
            self.simulator_cluster.scale(self.n_jobs)
            self.simulator_client = Client(self.simulator_cluster)
            self.clients = [self.simulator_client]

        elif (
            sampler_type in [S.ACTIVE, S.ACTIVEDB]
            and isinstance(self.worker_args.get("simulator_args"), dict)
            and isinstance(self.worker_args.get("surrogate_args"), dict)
        ):
            simulator_args = self.worker_args["simulator_args"]
            surrogate_args = self.worker_args["surrogate_args"]

            simulator_workers = self.worker_args["simulator_workers"]
            surrogate_workers = self.worker_args["surrogate_workers"]

            self.simulator_cluster = SLURMCluster(**simulator_args)
            self.surrogate_cluster = SLURMCluster(**surrogate_args)

            self.simulator_cluster.scale(simulator_workers)
            self.surrogate_cluster.scale(surrogate_workers)

            self.simulator_client = Client(self.simulator_cluster)
            self.surrogate_client = Client(self.surrogate_cluster)

            self.clients = [self.simulator_client, self.surrogate_client]
        else:
            raise ValueError(
                "Make sure that the config has simulator_args and surrogate_args if using "
                + "ACTIVE samplers"
            )

    def submit_batch_of_params(self, param_list: list[dict]) -> list:
        """Submits a batch of parameters to the class

        Raises:
            ValueError: If the configuration is incomplete for ACTIVE sampler types.
        """
        futures = []
        print(param_list)
        for params in param_list:
            new_future = self.simulator_client.submit(
                run_simulation_task, self.runner_args, params, self.base_run_dir
            )
            futures.append(new_future)
        print(futures)
        return futures

    def start_runs(self):
        sampler_interface: S = self.sampler.sampler_interface
        print(100 * "=")
        print("Creating initial runs")

        print("Generating samples:")
        initial_parameters = self.sampler.get_initial_parameters()
        print("Number of initial samples generated:", len(initial_parameters))
        print("Making Dask futures")
        futures = self.submit_batch_of_params(initial_parameters)
        completed = 0
        if sampler_interface in [S.SEQUENTIAL]:    
            seq = wait(futures)
            if self.output_dir is not None:
                outputs = []
                for res in seq.done:
                    outputs.append(res.result())
                output_file_path = os.path.join(self.output_dir, "sequential")
                print("SAVING OUTPUT IN:", self.output_dir)
                with open(output_file_path, "w") as out_file:
                    for output in outputs:
                        out_file.write(str(output) + "\n\n")
            print("Finished sequential runs")

        elif sampler_interface in [S.BATCH]:
            while completed < self.sampler.total_budget:
                seq = wait(futures)
                param_list = self.sampler.get_next_parameter()
                futures = self.submit_batch_of_params(param_list)
                completed += len(futures)
                print("BATCH SAMPLER", 20 * "=", completed, 20 * "=")

        elif sampler_interface in [S.ACTIVE, S.ACTIVEDB]:
            iterations = 0
            while True:
                print(
                    20 * "=",
                    f"Iteration {iterations}; ",
                    20 * "=",
                )

                # NOTE: ------ Run Samples and block until completed ------
                seq = wait(
                    futures
                )  # outputs should be a list of tuples we ignore in this case

                # NOTE: ------ Collect outputs ------
                outputs = []
                for res in seq.done:
                    outputs.append(res.result())
                outputs = self.sampler.collect_batch_results(outputs)

                # NOTE: ------ Check if out of budget ------
                completed += len(outputs)
                if completed > self.sampler.total_budget:
                    break

                # NOTE: ------ Update the pool and training data from outputs ------
                self.sampler.parser.update_pool_and_train(outputs)
                print(self.sampler.parser.print_dset_sizes())

                # NOTE: Collect next data for training

                train, valid, test, pool = (
                    self.sampler.parser.get_unscaled_train_valid_test_pool_from_self()
                )

                # rescale data and pool
                train, valid, test, pool = (
                    self.sampler.parser.scale_train_val_test_pool(
                        train, valid, test, pool
                    )
                )

                train_data, valid_data = self.sampler.parser.make_train_valid_datasplit(
                    train, valid
                )

                self.sampler.model_kwargs["input_dim"] = train_data[0].shape[-1]
                self.sampler.model_kwargs["output_dim"] = train_data[1].shape[-1]

                # NOTE: ------ Submit model training job ------
                print(
                    "Going to training with ",
                    "X_tr",
                    train_data[0].shape,
                    "Y_tr",
                    train_data[1].shape,
                    "X_vl",
                    valid_data[0].shape,
                    "Y_vl",
                    valid_data[1].shape,
                )

                # TODO: profile this vs write the data instead
                time_starttrain = time.time()
                new_model_training = self.surrogate_client.submit(
                    run_train_model,
                    self.sampler.model_kwargs,
                    train_data,
                    valid_data,
                    self.sampler.train_kwargs,
                )

                train_model_run = wait([new_model_training])

                # NOTE: ------ Collect model training job ------
                trained_model_res = [res.result() for res in train_model_run.done]
                metrics, trained_model_state_dict = trained_model_res[0]
                print(f"Elapsed train task time: {time.time() - time_starttrain}")

                for metric_name, metric_vals in metrics.items():
                    # print(f'\n{metric_name}: min: {min(metric_vals)},
                    # max: {max(metric_vals)}
                    # @ epoch {metric_vals.index(min(metric_vals))}\n', metric_vals)
                    print(
                        f"\n{metric_name}: max: {max(metric_vals)} ",
                        f"@ epoch {metric_vals.index(max(metric_vals))}, ",
                        f"min: {min(metric_vals)} @ epoch {metric_vals.index(min(metric_vals))}\n",
                    )

                self.sampler.update_metrics(metrics)
                # NOTE: ------ Do active learning sampling ------

                example_model = load_saved_model(
                    self.sampler.model_kwargs, trained_model_state_dict
                )
                param_list = self.sampler.get_next_parameter(example_model, train, pool)

                # NOTE: ------ Prepare next simulator runs ----
                futures = self.submit_batch_of_params(param_list)
                iterations += 1
                self.sampler.dump_iteration_results(
                    self.base_run_dir, iterations, trained_model_state_dict
                )
            # self.sampler.dump_iteration_results(
            #    base_run_dir=self.base_run_dir, iterations=iterations)
