import pytest 
import os 
import sys 

sys.path.append(os.getcwd() + "/src")
from samplers import RandSampler, RandBatchSampler
import run

def test_initialization(): 
    sampler = RandSampler(
        bounds = [[0, 9], [0, 9]], num_samples = 10000, parameters=["a", "b"]
        )
    batch_size = 5
    sampler = RandBatchSampler( 
        bounds = [[-5, 5], [-5, 5]], 
        batch_size = batch_size, 
        total_budget = 100,
        parameters=["a", "b"]
    )

    initializing_batch = sampler.get_initial_parameters() 
    assert len(initializing_batch) == batch_size

def test_sampling(): 
    sampler = RandSampler(
        bounds = [[0, 9], [0, 9]], num_samples = 10000, parameters=["a", "b"]
        )

    for _ in range(10): 
        parameter_dict = sampler.get_next_parameter() 
        for key, value in parameter_dict.items(): 
            assert value > -0.0001
            assert value < 9.0001 
    
    batch_size = 5
    sampler = RandBatchSampler( 
        bounds = [[-5, 5], [-5, 5]], 
        batch_size = batch_size, 
        total_budget = 100,
        parameters=["a", "b"]
    )
    assert len(sampler.get_next_parameter()) == batch_size
    for _ in range(10): 
        list_parameter_dict = sampler.get_next_parameter() 
        for parameter_dict in list_parameter_dict:
            for key, value in parameter_dict.items(): 
                assert value > -5
                assert value < 5

