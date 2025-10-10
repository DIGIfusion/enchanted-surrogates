# tests/test_array_sampler.py

import pytest
from enchanted_surrogates.samplers.array_sampler import ArraySampler


def test_generate_parameters_creates_cartesian_product():
    bounds = [[5, 7, 77, 199], [0.02, 0.2]]
    sampler = ArraySampler(bounds=bounds, total_budget=None, parameters=["a", "b"])

    params = list(sampler.generate_parameters())
    expected = [
        [5, 0.02], [5, 0.2],
        [7, 0.02], [7, 0.2],
        [77, 0.02], [77, 0.2],
        [199, 0.02], [199, 0.2],
    ]
    assert params == expected
    assert sampler.total_budget == len(expected)


def test_total_budget_limit_raises_exception():
    # 11 parameters with 11 values each => 11**11 combinations (~2.85e11 > 1e5)
    bounds = [list(range(11))] * 11
    with pytest.raises(Exception) as excinfo:
        ArraySampler(bounds=bounds, total_budget=None, parameters=[f"x{i}" for i in range(11)])
    assert "array sampling" in str(excinfo.value)


def test_get_next_samples_returns_dicts(monkeypatch):
    bounds = [[1, 2], ["x", "y"]]
    sampler = ArraySampler(bounds=bounds, total_budget=None, parameters=["num", "char"], )

    # Patch batch_size to 2 for testing
    sampler.batch_size = 2
    sampler.submitted = 0  # ensure counter exists

    samples = sampler.get_next_samples()
    assert len(samples) == 2
    assert isinstance(samples, list)
    assert all(isinstance(s, dict) for s in samples)
    assert all(set(s.keys()) == {"num", "char"} for s in samples)
    assert sampler.submitted == 2

    # Get another batch
    samples2 = sampler.get_next_samples()
    assert len(samples2) == 2
    assert sampler.submitted == 4
    assert samples != samples2  # Should get different samples in next batch
