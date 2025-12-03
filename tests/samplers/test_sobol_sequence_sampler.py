import pytest
import numpy as np
from enchanted_surrogates.samplers.sobol_sequence import SobolSequence

def test_get_next_samples_returns_dicts_within_bounds():
    bounds = [(0, 1), (10, 20)]
    params = ["x", "y"]
    sampler = SobolSequence(bounds=bounds, budget=8, parameters=params, batch_size=1)

    samples = sampler.get_next_samples()

    assert isinstance(samples, list)
    assert len(samples) == 1
    sample = samples[0]

    # Parameter names match
    assert set(sample.keys()) == set(params)

    # Values are within bounds
    assert 0 <= sample["x"] <= 1
    assert 10 <= sample["y"] <= 20

    # Counter increased
    assert sampler.submitted == 1

def test_randomness_with_fixed_seed():
    bounds = [(0, 1)]
    params = ["a"]
    fixed_seed = 10
    sampler1 = SobolSequence(bounds=bounds, budget=8, parameters=params, seed=fixed_seed)
    sampler2 = SobolSequence(bounds=bounds, budget=8, parameters=params, seed=fixed_seed)
    
    samples1 = sampler1.get_next_samples()
    samples2 = sampler2.get_next_samples()

    assert samples1 == samples2

def test_budget_must_be_power_of_two():
    bounds = [(0, 1)]
    params = ["x"]
    budget = 5
    with pytest.warns():
        sampler = SobolSequence(bounds=bounds, budget=budget, parameters=params)

    # Budget is changed to be a power of two
    assert sampler.budget != budget
    assert (sampler.budget & (sampler.budget - 1) == 0) and sampler.budget != 0