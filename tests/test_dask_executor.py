import os, sys
sys.path.append('/users/danieljordan/enchanted-surrogates/src/')

import pytest 
from enchanted_surrogates.executors.dask_executor import DaskExecutor
# import glob 
import shutil


def test_dask_executor():
    config = {}

    # -- User Config
    user_config = {
        'path_to_enchanted-surrogates':'/users/danieljordan/enchanted-surrogates/src/',
        'activate_env_command':'export PATH=/scratch/project_462000954/enchanted_container_lumi3/bin:$PATH',
    }

    # -- Executor
    executor_args = {
        'type': 'DaskExecutor',
        'base_run_dir': f"{os.path.dirname(__file__)}/example",
        'block_unitil_cluster_started': True, # default False: for debugging purposes
        'sampler_args':{
            'type': 'RandomSampler',
            'bounds':[[-5, 5], [0, 1]],
            'parameters':['c1', 'c2'],
            'total_budget':50
        },
        
        'runner_args':{
            'type': 'ExampleRunner'
        },
        
        'LocalCluster_args':{
            'name':'es-dask_cluster', 
            'n_workers':5,
            'threads_per_worker':1
        }       
    }

    if os.path.exists(executor_args['base_run_dir']):
        print('REMOVING OLD BASE RUN DIR: ',executor_args['base_run_dir'])
        os.system(f"rm -r {executor_args['base_run_dir']}")

    # create the executor
    executor = DaskExecutor(**executor_args, **config)

    # executor.start_cluster()
    # assert executor.expected_number_of_workers == len(executor.client.scheduler_info()["workers"])

    executor.start_runs()

    assert os.path.exists(os.path.join(executor_args['base_run_dir'],'ENCHANTED.FINNISHED'))

    # TODO clean up test
    # shutil.rmtree(base_run_dir)

if __name__ == "__main__":
    test_dask_executor()