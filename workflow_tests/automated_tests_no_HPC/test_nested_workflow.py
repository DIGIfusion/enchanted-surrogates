"""
Basic tests for the full workflow when using nested executors and samplers.
Each executor should be tested in a separate function to avoid conflicts."""
import os
import glob
import shutil

from types import SimpleNamespace
from enchanted_surrogates.executors import LocalExecutor, JoblibExecutor, DaskExecutor
from enchanted_surrogates.supervisor.supervisor import Supervisor

# Configs to be reused in test cases
dask_config = {
    'type': 'DaskExecutor',
    'LocalCluster_config': {
        'name': 'es-dask_cluster',
        'n_workers': 2,
        'threads_per_worker': 1
    }
}
local_config = {
    'type': 'LocalExecutor'
}
sampler_config_1 = {
    'type': 'GridSampler', 
    'bounds': [[-5, 5], [0, 1]], 
    'parameters': ['c1', 'c2'],
    'num_samples': 2
}
sampler_config_2 = {
    'type': 'GridSampler', 
    'bounds': [[-5, 5], [0, 1]], 
    'parameters': ['c3', 'c4'],
    'num_samples': 2
}
sampler_config_3 = {
    'type': 'GridSampler', 
    'bounds': [[-5, 5], [0, 1]], 
    'parameters': ['c5', 'c6'],
    'num_samples': 2
}
runner_config_1 = {
    'type': 'ExampleRunner',
    'parameter_mode': 0,
    'sleep_sec': 0.1
}
runner_config_2 = {
    'type': 'ExampleRunner',
    'parameter_mode': 1,
    'sleep_sec': 0.1
}
runner_config_3 = {
    'type': 'ExampleRunner',
    'parameter_mode': 2,
    'sleep_sec': 0.1
}

def test_nested_dask_executors(tmp_path):
    args = SimpleNamespace(
        executors = {
            # New executors are created, but they using the same config
            'e1': dask_config.copy(),
            'e2': dask_config.copy()
        },
        samplers = {
            's1': sampler_config_1.copy(),
            's2': sampler_config_2.copy()
        },
        runners = {
            'r1': runner_config_1.copy(),
            'r2': runner_config_2.copy()
        },
        supervisor = {
            'base_run_dir': tmp_path,
            'run_order': [
                {
                    'executor': 'e1',
                    'sampler': 's1',
                    'runner' : 'r1'
                },
                {
                    'executor': 'e2',
                    'sampler': 's2',
                    'runner' : 'r2'
                }
            ]
        }
    )

    supervisor = Supervisor(args)
    supervisor.start()


def test_nested_dask_executors_with_executor_reuse(tmp_path):
    args = SimpleNamespace(
        executors = {
            'e1': dask_config.copy()
        },
        samplers = {
            's1': sampler_config_1.copy(),
            's2': sampler_config_2.copy(),
            's3': sampler_config_3.copy()
        },
        runners = {
            'r1': runner_config_1.copy(),
            'r2': runner_config_2.copy(),
            'r3': runner_config_3.copy()
        },
        supervisor = {
            'base_run_dir': tmp_path,
            'run_order': [
                {
                    'executor': 'e1',
                    'sampler': 's1',
                    'runner' : 'r1'
                },
                {
                    'executor': 'e1',
                    'sampler': 's2',
                    'runner' : 'r2'
                },
                {
                    'executor': 'e1',
                    'sampler': 's3',
                    'runner' : 'r3'
                }
            ]
        }
    )

    supervisor = Supervisor(args)
    supervisor.start()
