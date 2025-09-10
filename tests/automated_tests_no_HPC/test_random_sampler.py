# tests/test_random_sampler.py

import pytest
import numpy as np
from enchanted_surrogates.samplers.random_sampler import RandomSampler


def test_get_next_samples_returns_dicts_within_bounds():
    bounds = [(0, 1), (10, 20)]
    params = ["x", "y"]
    sampler = RandomSampler(bounds=bounds, total_budget=10, parameters=params)
    sampler.submitted = 0  # ensure counter exists

    samples = sampler.get_next_samples()

    assert isinstance(samples, list)
    assert len(samples) == 1
    sample = samples[0]

    # Keys match parameters
    assert set(sample.keys()) == set(params)

    # Values lie within bounds
    assert 0 <= sample["x"] <= 1
    assert 10 <= sample["y"] <= 20

    # Counter increments
    assert sampler.submitted == 1


def test_batch_sample_size(monkeypatch):
    bounds = [(0, 1), (10, 20)]
    params = ["x", "y"]
    sampler = RandomSampler(bounds=bounds, total_budget=10, parameters=params)
    sampler.BATCH_SAMPLE_SIZE = 3
    sampler.submitted = 0

    samples = sampler.get_next_samples()

    assert isinstance(samples, list)
    assert len(samples) == 3
    assert all(set(s.keys()) == set(params) for s in samples)
    assert sampler.submitted == 3


def test_randomness_with_fixed_seed():
    bounds = [(0, 1)]
    params = ["a"]
    sampler = RandomSampler(bounds=bounds, total_budget=5, parameters=params)
    sampler.submitted = 0

    np.random.seed(42)
    samples1 = sampler.get_next_samples()

    np.random.seed(42)
    samples2 = sampler.get_next_samples()

    # With the same seed, results should be reproducible
    assert samples1 == samples2
