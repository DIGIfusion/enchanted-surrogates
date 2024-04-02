import pytest
import os
import sys

sys.path.append(os.getcwd() + "/src")
from samplers import HypercubeSampler
import run


def test_initialization():
    sampler = HypercubeSampler(
        bounds=[[0, 9], [0, 9]], num_samples=10, parameters=["a", "b"]
    )
    assert len(sampler.hypercube_grid) == 100
    assert sampler.hypercube_grid[0] == [0.0, 0.0]
    assert sampler.hypercube_grid[15] == [1.0, 5.0]
    assert sampler.hypercube_grid[99] == [9.0, 9.0]


def test_next_parameter():
    sampler = HypercubeSampler(
        bounds=[[0, 9], [0, 9]], num_samples=10, parameters=["a", "b"]
    )
    assert sampler.get_next_parameter() == [0.0, 0.0]
    assert sampler.get_next_parameter() == [0.0, 1.0]
