import os
import sys
import shutil
from ..utils.append_es_to_path import append_es_to_path
append_es_to_path()
# Dynamically calculate the path to the 'src' directory
current_file = os.path.abspath(__file__)
tests_dir = os.path.dirname(
    os.sep.join(
        os.path.normpath(current_file).split(os.sep)[:current_file.split(os.sep).index("tests") + 1]))
src_path = os.path.join(os.path.dirname(tests_dir), "src")
sys.path.append(src_path)

from enchanted_surrogates.executors.dask_executor import DaskExecutor


def test_dask_executor():
    print('TESTING DASK EXECUTOR')
    config = {}

    # -- Executor
    executor_config = {
        'type': 'DaskExecutor',
        'base_run_dir': f"{os.path.dirname(__file__)}/example_base_run_dir",
        'block_unitil_cluster_started': True,  # default False: for debugging purposes
        'sampler_config': {
            'type': 'RandomSampler',
            'bounds': [[-5, 5], [0, 1]],
            'parameters': ['c1', 'c2'],
            'budget': 10
        },
        'runner_config': {
            'type': 'ExampleRunner'
        },
        'LocalCluster_config': {
            'name': 'es-dask_cluster',
            'n_workers': 5,
            'threads_per_worker': 1
        }
    }

    if os.path.exists(executor_config['base_run_dir']):
        print('REMOVING OLD BASE RUN DIR: ', executor_config['base_run_dir'])
        os.system(f"rm -r {executor_config['base_run_dir']}")

    # create the executor
    executor = DaskExecutor(**executor_config, **config)

    # executor.start_cluster()
    # assert executor.expected_number_of_workers == len(executor.client.scheduler_info()["workers"])

    executor.start_runs()

    assert os.path.exists(os.path.join(executor_config['base_run_dir'], 'ENCHANTED.FINISHED'))

    # TODO clean up test
    shutil.rmtree(executor_config['base_run_dir'])
