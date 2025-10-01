import os
import sys
# Dynamically calculate the path to the 'src' directory
current_file = os.path.abspath(__file__)
tests_dir = os.path.dirname(
    os.sep.join(
        os.path.normpath(current_file).split(os.sep)[:current_file.split(os.sep).index("tests") + 1])
)
src_path = os.path.join(os.path.dirname(tests_dir), "src")
sys.path.append(src_path)

from enchanted_surrogates.executors.dask_nested_executor import DaskNestedExecutor


def test_dask_nested_executor(tmp_path):
    tmp_path = '/users/danieljordan/enchanted_plugins/enchanted-surrogates/tests/automated_tests_no_HPC/nested_test'

    print('TESTING DASK EXECUTOR')
    config = {}

    # -- Executor
    executor_kwargs = {
        'type': 'DaskNestedExecutor',
        'base_run_dir': tmp_path,
        'start_cluster_when_needed': True,
        'shutdown_finished_clusters': True,
        'block_until_cluster_started': True,  # default False: for debugging purposes
        'sampler_kwargs': {
            'type': 'NestedSampler',
            'samplers': {
                'code1_sampler': {  # any name can be used for the samplers, the number of samplers must be the same as the number of executors
                    'type': 'GridSampler',
                    'bounds': [[-5, 5], [0, 1]],
                    'parameters': ['c1', 'c2'],
                    'num_samples': 2
                },
                'code2_sampler': {
                    'type': 'GridSampler',
                    'bounds': [[-5, 5], [0, 1]],
                    'parameters': ['c3', 'c4'],
                    'num_samples': 2
                },
                'code3_sampler': {
                    'type': 'GridSampler',
                    'bounds': [[-5, 5], [0, 1]],
                    'parameters': ['c5', 'c6'],
                    'num_samples': 2
                },

            },
        },
        'executors': {
            'code1_executor': {
                'type': 'DaskExecutor',
                'runner_kwargs': {
                    'type': 'ExampleRunner',
                    'parameter_mode': 0  # just to make sure each code does something different
                },
                'LocalCluster_kwargs': {
                    'name': 'es-dask_cluster',
                    'n_workers': 3,
                    'threads_per_worker': 1
                },
            },
            'code2_executor': {
                'type': 'DaskExecutor',
                'runner_kwargs': {
                    'type': 'ExampleRunner',
                    'parameter_mode': 1  # just to make sure each code does something different
                },
                'LocalCluster_kwargs': {
                    'name': 'es-dask_cluster',
                    'n_workers': 3,
                    'threads_per_worker': 1
                }
            },
            'code3_executor': {
                'type': 'DaskExecutor',
                'runner_kwargs': {
                    'type': 'ExampleRunner',
                    'parameter_mode': 2  # just to make sure each code does something different
                },
                'LocalCluster_kwargs': {
                    'name': 'es-dask_cluster',
                    'n_workers': 3,
                    'threads_per_worker': 1
                }
            }
        }
    }

    if os.path.exists(executor_kwargs['base_run_dir']):
        print('REMOVING OLD BASE RUN DIR: ', executor_kwargs['base_run_dir'])
        os.system(f"rm -r {executor_kwargs['base_run_dir']}")

    # create the executor
    executor = DaskNestedExecutor(**executor_kwargs, **config)

    # executor.start_cluster()
    # assert executor.expected_number_of_workers == len(executor.client.scheduler_info()["workers"])

    executor.start_runs()

    assert os.path.exists(os.path.join(executor_kwargs['base_run_dir'], 'ENCHANTED.FINISHED'))

    # TODO clean up test
    # shutil.rmtree(base_run_dir)


def test_dask_nested_executor_executor_reuse(tmp_path):
    print('TESTING DASK EXECUTOR - WITH EXECUTOR REUSE')
    config = {}

    # -- Executor
    executor_kwargs = {
        'type': 'DaskNestedExecutor',
        'base_run_dir': tmp_path,
        'start_cluster_when_needed': True,
        'shutdown_finished_clusters': True,
        'block_until_cluster_started': True,  # default False: for debugging purposes
        'sampler_kwargs': {
            'type': 'NestedSampler',
            'samplers': {
                'code1_sampler': {  # any name can be used for the samplers, the number of samplers must be the same as the number of executors
                    'type': 'GridSampler',
                    'bounds': [[-5, 5], [0, 1]],
                    'parameters': ['c1', 'c2'],
                    'num_samples': 2
                },
                'code2_sampler': {
                    'type': 'GridSampler',
                    'bounds': [[-5, 5], [0, 1]],
                    'parameters': ['c3', 'c4'],
                    'num_samples': 2
                },
                'code3_sampler': {
                    'type': 'GridSampler',
                    'bounds': [[-5, 5], [0, 1]],
                    'parameters': ['c5', 'c6'],
                    'num_samples': 2
                },

            },
        },
        'executors': {
            'code1_executor': {
                'type': 'DaskExecutor',
                'runner_kwargs': {
                    'type': 'ExampleRunner',
                    'parameter_mode': 0  # just to make sure each code does something different
                },
                'LocalCluster_kwargs': {
                    'name': 'es-dask_cluster',
                    'n_workers': 5,
                    'threads_per_worker': 1
                },
            },
            'code2_executor': {
                'type': 'code1_executor',  # reuse code 1 executor, but change the runner, means both runners will be ran on the same dask cluster
                'runner_kwargs': {
                    'type': 'ExampleRunner',
                    'parameter_mode':  1 # just to make sure each code does something different
                },
            },
            'code3_executor': {
                'type': 'DaskExecutor',
                'runner_kwargs': {
                    'type': 'ExampleRunner',
                    'parameter_mode': 2  # just to make sure each code does something different
                },
                'LocalCluster_kwargs': {
                    'name': 'es-dask_cluster',
                    'n_workers': 5,
                    'threads_per_worker': 1
                }
            }
        }
    }

    if os.path.exists(executor_kwargs['base_run_dir']):
        print('REMOVING OLD BASE RUN DIR: ', executor_kwargs['base_run_dir'])
        os.system(f"rm -r {executor_kwargs['base_run_dir']}")

    # create the executor
    executor = DaskNestedExecutor(**executor_kwargs, **config)

    # executor.start_cluster()
    # assert executor.expected_number_of_workers == len(executor.client.scheduler_info()["workers"])

    executor.start_runs()

    assert os.path.exists(os.path.join(executor_kwargs['base_run_dir'], 'ENCHANTED.FINISHED'))

    # TODO clean up test
    # shutil.rmtree(base_run_dir)
