import os
import sys
import numpy as np

sys.path.append(os.getcwd() + "/src")
from samplers import RandomSampler


def test_output_shape():
    bounds = [[0, 1], [0, 100]]
    num_samples = [5, 25]
    parameters = ["a", "b"]
    sampler = RandomSampler(
        bounds=bounds, num_samples=num_samples, parameters=parameters
    )
    assert len(sampler.samples) == np.prod(num_samples), "Output shape is incorrect"


def test_output_range():
    bounds = [[0, 1], [0, 100]]
    num_samples = [5, 25]
    parameters = ["a", "b"]
    sampler = RandomSampler(
        bounds=bounds, num_samples=num_samples, parameters=parameters
    )
    samples = np.array(sampler.samples)
    assert np.all(samples[:, 0] >= bounds[0][0]) and np.all(
        samples[:, 0] <= bounds[0][1]
    ), "Samples are not within boundaries"
    assert np.all(samples[:, 1] >= bounds[1][0]) and np.all(
        samples[:, 1] <= bounds[1][1]
    ), "Samples are not within boundaries"


def test_functionality():
    result = True
    try:
        bounds = [[0, 1], [0, 100]]
        num_samples = [5, 25]
        parameters = ["a", "b"]
        sampler = RandomSampler(
            bounds=bounds, num_samples=num_samples, parameters=parameters
        )
        sampler.get_next_parameter()
    except Exception as e:
        print(e)
        result = False
    assert sampler.get_next_parameter() != sampler.get_next_parameter()
    assert result
