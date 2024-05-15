import pytest
import os
import sys
import numpy as np 
sys.path.append(os.getcwd() + "/src")
from samplers import Grid
import run

def test_initialization():
    sampler = Grid(bounds=[[0, 9], [0, 9]], num_samples=10, parameters=["a", "b"])
    assert len(sampler.samples) == 100
    assert sampler.samples[0] == [0.0, 0.0]
    assert sampler.samples[15] == [1.0, 5.0]
    assert sampler.samples[99] == [9.0, 9.0]

    initial_parameters = sampler.get_initial_parameters()

    assert initial_parameters[0] == {'a': 0.0, 'b': 0.0}
    assert initial_parameters[15] == {'a': 1.0, 'b': 5.0}
    assert initial_parameters[99] == {'a': 9.0, 'b': 9.0}
    

def test_initialization2():
    sampler = Grid(
        bounds=[[0, 2], [0, 3], [0, 4]],
        num_samples=[3, 4, 5],
        parameters=["a", "b", "c"],
    )
    assert len(sampler.samples) == 60
    assert sampler.samples[0] == [0.0, 0.0, 0.0]
    assert sampler.samples[26] == [1.0, 1.0, 1.0]
    assert sampler.samples[59] == [2.0, 3.0, 4.0]

    initial_parameters = sampler.get_initial_parameters()

    assert initial_parameters[0] == {'a': 0.0, 'b': 0.0, 'c': 0.0}
    assert initial_parameters[26] == {'a': 1.0, 'b': 1.0, 'c': 1.0}
    assert initial_parameters[59] == {'a': 2.0, 'b': 3.0, 'c': 4.0}
    
def test_next_parameter():
    sampler = Grid(bounds=[[0, 9], [0, 9]], num_samples=[10, 5], parameters=["a", "b"])
    assert sampler.get_next_parameter() == {"a": 0.0, "b": 0.0}
    assert sampler.get_next_parameter() == {"a": 0.0, "b": 2.25}

def test_total_num_samples(): 
    sampler = Grid(bounds=[[0, 9], [0, 9]], num_samples=[10, 5], parameters=["a", "b"])

    assert sampler.num_initial_points == np.prod(sampler.num_samples)
    assert sampler.num_initial_points == 50
