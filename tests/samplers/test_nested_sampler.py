# tests/test_nested_sampler.py

import pytest
import numpy as np
from enchanted_surrogates.samplers.nested_sampler import NestedSampler

def test_get_next_samples_returns_dicts_within_bounds():
    samplers = {
        'random': {
            'type': 'RandomSampler',
            'bounds': [[-5, 5], [0, 1]],
            'parameters': ['c1', 'c2'],
            'budget': 4
        },
        'array': {
            'type': 'ArraySampler',
            'bounds': [[-5, 5], [0, 1]],
            'parameters': ['c1', 'c2'],
            'budget': 4
        },
    }

    nested_sampler = NestedSampler(samplers=samplers, budget=4, batch_size=1)
    samples = nested_sampler.get_next_samples()

    assert all(-5 <= sample["c1"] <= 5 for sample in samples)
    assert all(0 <= sample["c2"] <= 1 for sample in samples)


def test_multiple_get_next_samples_raises_exception():
    samplers = {
        'random': {
            'type': 'RandomSampler',
            'bounds': [[-5, 5], [0, 1]],
            'parameters': ['c1', 'c2'],
            'budget': 4
        },
        'array': {
            'type': 'ArraySampler',
            'bounds': [[-5, 5], [0, 1]],
            'parameters': ['c1', 'c2'],
            'budget': 4
        },
    }

    nested_sampler = NestedSampler(samplers=samplers, budget=4, batch_size=1)
    _ = nested_sampler.get_next_samples()
    

    with pytest.raises(NotImplementedError) as excinfo:
        _ = nested_sampler.get_next_samples()
    
    assert excinfo

