import pytest 
from enchanted_surrogates.executors import LocalExecutor
from enchanted_surrogates.samplers.random_sampler import RandomSampler 
import glob 
import os 
import shutil

""" 

"""
# TODO: build and destroy test 

def test_full_workflow(): 
    config = {}
    # -- sampler 
    # TODO: test different samplers
    bounds = [[-5, 5], [0, 1]]
    parameters = ['c1', 'c2']
    total_budget = 50
    sampler = RandomSampler(bounds, total_budget, parameters)

    # -- runner args
    runner_args = {
        'type': 'ExampleRunner'
    }

    # --- other 

    # TODO: some other file 
    # __file__ + './test_full_workflow'
    
    base_run_dir = f"{os.path.dirname(__file__)}/example"

    # create the executor 
    executor = LocalExecutor(sampler=sampler, runner_args=runner_args, base_run_dir=base_run_dir, **config)
    executor.start_runs() 
    # This should create {total_budget} folders with ??? inside 

    assert len(glob.glob(os.path.join(base_run_dir, "*"))) == total_budget

    executor.clean() 

    # TODO clean up test
    shutil.rmtree(base_run_dir)