import os, sys
import os
project_root = os.sep.join(os.path.normpath(__file__).split(os.sep)[:__file__.split(os.sep).index("enchanted-surrogates")+1])
sys.path.append(os.path.join(project_root, 'src'))

import pytest 
from enchanted_surrogates.executors.dask_executor import DaskExecutor
# import glob 
import shutil


def test_dask_executor():
    print('TESTING DASK EXECUTOR')
    config = {}

    # -- Executor
    executor_kwargs = {
        'type': 'DaskExecutor',
        'base_run_dir': f"{os.path.dirname(__file__)}/example",
        'block_unitil_cluster_started': True, # default False: for debugging purposes
        'sampler_kwargs':{
            'type': 'RandomSampler',
            'bounds':[[-5, 5], [0, 1]],
            'parameters':['c1', 'c2'],
            'total_budget':50
        },
        
        'runner_kwargs':{
            'type': 'ExampleRunner'
        },
        
        'LocalCluster_kwargs':{
            'name':'es-dask_cluster', 
            'n_workers':5,
            'threads_per_worker':1
        }       
    }

    if os.path.exists(executor_kwargs['base_run_dir']):
        print('REMOVING OLD BASE RUN DIR: ',executor_kwargs['base_run_dir'])
        os.system(f"rm -r {executor_kwargs['base_run_dir']}")

    # create the executor
    executor = DaskExecutor(**executor_kwargs, **config)

    # executor.start_cluster()
    # assert executor.expected_number_of_workers == len(executor.client.scheduler_info()["workers"])

    executor.start_runs()

    assert os.path.exists(os.path.join(executor_kwargs['base_run_dir'],'ENCHANTED.FINNISHED'))

    # TODO clean up test
    # shutil.rmtree(base_run_dir)

if __name__ == "__main__":
    test_dask_executor()