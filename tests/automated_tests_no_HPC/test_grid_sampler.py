# tests/test_grid_sampler.py

import pytest
import numpy as np
from enchanted_surrogates.samplers.grid_sampler import GridSampler


def test_generate_parameters_creates_expected_grid():
    bounds = [(0, 1), (10, 20)]
    num_samples = [2, 3]
    params = ["x", "y"]

    sampler = GridSampler(bounds=bounds, num_samples=num_samples, parameters=params)

    samples = list(sampler.generate_parameters())
    expected_x = np.linspace(0, 1, 2)
    expected_y = np.linspace(10, 20, 3)

    expected = [[x, y] for x in expected_x for y in expected_y]
    assert samples == expected
    assert sampler.budget == len(expected)


def test_total_budget_limit_raises_exception():
    # 11 parameters with 11 samples each → 11**11 = 285 billion > 100000
    bounds = [(0, 1)] * 11
    num_samples = [11] * 11
    params = [f"p{i}" for i in range(11)]

    with pytest.raises(Exception) as excinfo:
        GridSampler(bounds=bounds, num_samples=num_samples, parameters=params)
    assert "grid search" in str(excinfo.value)


def test_get_next_samples_iterates_sequentially():
    bounds = [(1, 2), (100, 200)]
    num_samples = [2, 2]
    params = ["a", "b"]

    sampler = GridSampler(bounds=bounds, num_samples=num_samples, parameters=params)

    all_samples = []
    while True:
        batch = sampler.get_next_samples()
        if not batch:
            break
        all_samples.extend(batch)

    expected_a = np.linspace(1, 2, 2)
    expected_b = np.linspace(100, 200, 2)
    expected = [{"a": x, "b": y} for x in expected_a for y in expected_b]

    assert all_samples == expected
    assert sampler.submitted == sampler.budget


def test_get_next_samples_respects_batch_size():
    bounds = [(0, 1)]
    num_samples = [3]
    params = ["x"]

    sampler = GridSampler(bounds=bounds, num_samples=num_samples, parameters=params)
    sampler.batch_size = 2

    batch1 = sampler.get_next_samples()
    batch2 = sampler.get_next_samples()
    batch3 = sampler.get_next_samples()  # should be empty

    assert len(batch1) == 2
    assert len(batch2) == 1
    assert batch3 == []
    assert sampler.submitted == 3  # total budget consumed


def test_generate_parameters_is_repeatable():
    bounds = [(0, 1)]
    num_samples = [2]
    params = ["x"]

    sampler = GridSampler(bounds=bounds, num_samples=num_samples, parameters=params)

    first_call = list(sampler.generate_parameters())
    second_call = list(sampler.generate_parameters())

    assert first_call == second_call
