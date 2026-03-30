import numpy as np
from enchanted_surrogates.samplers.latin_hypercube_sampler import LatinHypercubeSampler

def test_get_next_samples_returns_dicts_within_bounds():
    bounds = [(0, 1), (10, 20)]
    params = ["c1", "c2"]
    sampler = LatinHypercubeSampler(bounds=bounds, budget=10, parameters=params, batch_size=1)

    samples = sampler.get_next_samples()

    assert isinstance(samples, list)
    assert len(samples) == 1
    sample = samples[0]

    assert set(sample.keys()) == set(params)
    assert 0 <= sample["c1"] <= 1
    assert 10 <= sample["c2"] <= 20

    assert sampler.submitted == 1

def test_batch_size():
    bounds = [(0, 1), (10, 20)]
    params = ["c1", "c2"]
    sampler = LatinHypercubeSampler(bounds=bounds, budget=10, parameters=params, batch_size=5)

    samples = sampler.get_next_samples()

    assert len(samples) == 5
    assert all(set(s.keys()) == set(params) for s in samples)
    assert sampler.submitted == 5


def test_lhs_covers_all_strata():
    """Each cut in each dimension has exactly one point"""
    bounds = [(0, 1), (0, 1)]
    params = ["c1", "c2"]
    n = 10
    sampler = LatinHypercubeSampler(bounds=bounds, budget=n, parameters=params, batch_size=n)

    samples = sampler.get_next_samples()

    for param, (low, high) in zip(params, bounds):
        values = sorted(s[param] for s in samples)
        bin_width = (high - low) / n
        for i, val in enumerate(values):
            assert low + i * bin_width <= val <= low + (i + 1) * bin_width, (
                f"{param}: Value {val} does not belong to gap [{low + i*bin_width}, {low + (i+1)*bin_width}]"
            )

def test_reproducibility_with_fixed_seed():
    bounds = [(0, 1), (5, 10)]
    params = ["c1", "c2"]
    sampler = LatinHypercubeSampler(bounds=bounds, budget=5, parameters=params)

    np.random.seed(42)
    samples1 = sampler.get_next_samples()

    np.random.seed(42)
    sampler.submitted = 0
    samples2 = sampler.get_next_samples()

    assert samples1 == samples2

def test_submitted_counter_accumulates():
    bounds = [(0, 1)]
    params = ["c1"]
    sampler = LatinHypercubeSampler(bounds=bounds, budget=10, parameters=params, batch_size=4)

    sampler.get_next_samples()
    sampler.get_next_samples()

    assert sampler.submitted == 8


def test_default_batch_size_equals_budget():
    bounds = [(0, 1)]
    params = ["c1"]
    sampler = LatinHypercubeSampler(bounds=bounds, budget=7, parameters=params)

    assert sampler.batch_size == 7

def test_last_batch_does_not_exceed_budget():
    bounds = [(0, 1)]
    params = ["c1"]
    sampler = LatinHypercubeSampler(bounds=bounds, budget=10, parameters=params, batch_size=3)

    all_samples = []
    for _ in range(4):  # last batch should be only one, as budget runs out
        all_samples += sampler.get_next_samples()

    assert len(all_samples) == 10
    assert sampler.submitted == 10