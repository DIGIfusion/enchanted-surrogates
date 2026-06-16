import os, sys
from ..utils.append_es_to_path import append_es_to_path
append_es_to_path()
from enchanted_surrogates.executors import DaskExecutor
from enchanted_surrogates.supervisor.supervisor import Supervisor
# import glob 
import shutil
import json
from types import SimpleNamespace

def test_dask_executor():
    config = {}

    # -- User Config
    # Load JSON file into a Python dict
    if os.path.exists(os.path.join(os.path.dirname(__file__),f"user_config_{os.environ['USER']}.json")):
        user_config_file = os.path.join(os.path.dirname(__file__),f"user_config_{os.environ['USER']}.json")
    else:
        raise FileNotFoundError('PLEASE MAKE A USER CONFIG FILE FOR THE LUMI TESTS CALLED user_config_<your-lumi-username>.json')
    with open(user_config_file, "r") as file:
        user_config = json.load(file)
    
    assert user_config['path_to_enchanted-surrogates']
    assert user_config['activate_env_command']
    assert user_config['project']

    base_run_dir = f"{os.path.dirname(__file__)}/example_base_run_dir"
    
    # -- Executor
    executor_config = {
        'type': 'DaskExecutor',
        'block_unitil_cluster_started': True, # default False: for debugging purposes
        'SLURMcluster_config':{
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
        'scale_n_jobs': 2 # used by dask-jobqueue to submit n sbatch jobs where each job requests a single node and starts SLURMcluster_config['processes'] number workers on each node
    }
    sampler_config = {
        'type': 'RandomSampler',
        'bounds':[[-5, 5], [0, 1]],
        'parameters':['c1', 'c2'],
        'total_budget':50
    }
    runner_config = {
        'type': 'ExampleRunner'
    }

    args = SimpleNamespace(
        executors = { 'e1': executor_config },
        samplers = { 's1': sampler_config },
        runners = { 'r1': runner_config },
        supervisor = {
            'base_run_dir': base_run_dir,
            'run_order': [
                {
                    'executor': 'e1',
                    'sampler': 's1',
                    'runner' : 'r1'
                }
            ]
        }
    )

    if os.path.exists(base_run_dir):
        print('REMOVING OLD BASE RUN DIR: ', base_run_dir)
        os.system(f"rm -r {base_run_dir}")

    supervisor = Supervisor(args)
    supervisor.start()

    #assert os.path.exists(os.path.join(base_run_dir,'ENCHANTED.FINISHED'))

    # TODO clean up test
    # shutil.rmtree(base_run_dir)

if __name__ == "__main__":
    test_dask_executor()
