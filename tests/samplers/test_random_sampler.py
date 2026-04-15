# tests/test_random_sampler.py

import pytest
import numpy as np
from enchanted_surrogates.samplers.random_sampler import RandomSampler


def test_get_next_samples_returns_dicts_within_bounds():
    bounds = [(0, 1), (10, 20)]
    params = ["x", "y"]
    sampler = RandomSampler(bounds=bounds, budget=10, parameters=params, batch_size=1)
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
    sampler = RandomSampler(bounds=bounds, budget=10, parameters=params)
    sampler.batch_size = 3
    sampler.submitted = 0

    samples = sampler.get_next_samples()

    assert isinstance(samples, list)
    assert len(samples) == 3
    assert all(set(s.keys()) == set(params) for s in samples)
    assert sampler.submitted == 3


def test_randomness_with_fixed_seed():
    bounds = [(0, 1)]
    params = ["a"]
    sampler = RandomSampler(bounds=bounds, budget=5, parameters=params)
    sampler.submitted = 0

    np.random.seed(42)
    samples1 = sampler.get_next_samples()

    np.random.seed(42)
    samples2 = sampler.get_next_samples()

    # With the same seed, results should be reproducible
    assert samples1 == samples2


def test_skip():
    bounds = [(0, 1)]
    params = ["a"]
    budget = 100
    batch_size = 2
    sampler = RandomSampler(
        bounds=bounds, budget=budget, parameters=params, batch_size=batch_size
    )

    jump = 25
    sampler.skip(jump)

    batches = []
    while sampler.has_budget:
        batches.append(sampler.get_next_samples())

    assert len(batches) == (budget / batch_size) - jump


def test_random_distributions():
    sampler = RandomSampler(bounds=[(0, 1)], budget=1, parameters=["x"])

    sampler.distribution = "uniform"
    uniform_sample = sampler.sample_from_distribution(0.0, 1.0)
    assert 0.0 <= uniform_sample <= 1.0

    sampler.distribution = "loguniform"
    loguniform_sample = sampler.sample_from_distribution(1.0, 10.0)
    assert 1.0 <= loguniform_sample <= 10.0

    sampler.distribution = "normal"
    np.random.seed(0)
    normal_samples = [
        sampler.sample_from_distribution(0.0, 6.0) for _ in range(1000)]
    assert np.isfinite(normal_samples).all()
    assert 0.0 <= np.mean(normal_samples) <= 6.0
    assert np.std(normal_samples) > 0.0

    sampler.distribution = "int_uniform"
    int_sample = sampler.sample_from_distribution(1, 5)
    assert isinstance(int_sample, (int, np.integer))
    assert 1 <= int_sample <= 5

    sampler.distribution = "not_a_distribution"
    with pytest.raises(ValueError, match="Unknown distribution"):
        sampler.sample_from_distribution(0.0, 1.0)
