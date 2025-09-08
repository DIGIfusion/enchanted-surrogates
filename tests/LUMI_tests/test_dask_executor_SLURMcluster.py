import os, sys
project_root = os.sep.join(os.path.normpath(__file__).split(os.sep)[:__file__.split(os.sep).index("enchanted-surrogates")+1])
sys.path.append(os.path.join(project_root, 'src'))

import pytest 
from enchanted_surrogates.executors import DaskExecutor
# import glob 
import shutil
import json

def test_dask_executor():
    config = {}

    # -- User Config
    # Load JSON file into a Python dict
    with open(os.path.join(os.path.dirname(__file__),"user_config.json"), "r") as file:
        user_config = json.load(file)
    
    assert user_config['path_to_enchanted-surrogates']
    assert user_config['activate_env_command']
    assert user_config['project']
    
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
        'SLURMcluster_args':{
            'name':'es-dask_cluster', # This can be used by dask to seperate clusters and avoid confusion
            'cores':1, #
            'memory':'500MB', # Memory per node to be split between the number of workers on that node
            'walltime':'00:10:00', # Max walltime for the entire cluster
            'processes':1, # number of workers per node, (also per sbatch as SLURMcluster submits one sbatch per node)
            'interface':'nmn0', # changes for every system, try one and dask will suggest alternatives: ib0, ib1, bond0, bond1, nmn0, nmn1, eno1, eno2, eth0, eth1, etc.
            'account':user_config['project'],
            'queue':'small', # this changes --partition in the SBATCH script that starts the workers 
            'job_extra_directives':["--nodes=1"], # SBATCH is prepended to these and they are included in the sbatch that starts the workers
            'job_script_prologue':[ # these lines are insertered into the sbatch just before the workers are activated
                f"cd {user_config['path_to_enchanted-surrogates']}",
                user_config['activate_env_command'],
                f"export PYTHONPATH={user_config['path_to_enchanted-surrogates']}:$PYTHONPATH"
            ]
        },
        'scale_n_jobs': 2 # used by dask-jobqueue to submit n sbatch jobs where each job requests a single node and starts SLURMcluster_args['processes'] number workers on each node
    }

    if os.path.exists(executor_args['base_run_dir']):
        print('REMOVING OLD BASE RUN DIR: ',executor_args['base_run_dir'])
        os.system(f"rm -r {executor_args['base_run_dir']}")

    # create the executor
    executor = DaskExecutor(**executor_args, **config)
    
    executor.start_runs()

    assert os.path.exists(os.path.join(executor_args['base_run_dir'],'ENCHANTED.FINNISHED'))

    # TODO clean up test
    # shutil.rmtree(base_run_dir)

if __name__ == "__main__":
    test_dask_executor()
