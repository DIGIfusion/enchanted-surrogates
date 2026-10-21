import pytest
from enchanted_surrogates.samplers.sobol_sequence_sampler import SobolSequenceSampler


def test_get_next_samples_returns_dicts_within_bounds():
    bounds = [(0, 1), (10, 20)]
    params = ["x", "y"]
    sampler = SobolSequenceSampler(bounds=bounds, budget=8, parameters=params, batch_size=1)

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
    sampler1 = SobolSequenceSampler(bounds=bounds, budget=8, parameters=params, seed=fixed_seed)
    sampler2 = SobolSequenceSampler(bounds=bounds, budget=8, parameters=params, seed=fixed_seed)

    samples1 = sampler1.get_next_samples()
    samples2 = sampler2.get_next_samples()

    assert samples1 == samples2


def test_budget_must_be_power_of_two():
    bounds = [(0, 1)]
    params = ["x"]
    budget = 5
    with pytest.warns():
        sampler = SobolSequenceSampler(bounds=bounds, budget=budget, parameters=params)

    # Budget is changed to be a power of two
    assert sampler.budget != budget
    assert (sampler.budget & (sampler.budget - 1) == 0) and sampler.budget != 0


def test_num_repeats_repeats_every_point_in_a_row_with_a_repeat_index():
    bounds = [(300, 550), (50, 100)]
    params = ["a", "b"]
    plain = SobolSequenceSampler(bounds=bounds, budget=4, parameters=params, seed=4).get_next_samples()
    sampler = SobolSequenceSampler(bounds=bounds, budget=4, parameters=params, seed=4, num_repeats=3)

    assert sampler.budget == 12 and sampler.n_points == 4
    assert sampler.parameters == ["a", "b", "repeat"]
    samples = sampler.get_next_samples()
    assert len(samples) == 12
    assert [s["repeat"] for s in samples] == [0, 1, 2] * 4
    assert all(isinstance(s["repeat"], int) for s in samples)
    # the same 4 points as without repeats, each 3 times in a row
    for i, point in enumerate(plain):
        for s in samples[3 * i: 3 * i + 3]:
            assert (s["a"], s["b"]) == (point["a"], point["b"])
    assert not sampler.has_budget


def test_num_repeats_with_batches():
    sampler = SobolSequenceSampler(bounds=[(0, 1)], budget=4, parameters=["x"], seed=1, num_repeats=3, batch_size=6)
    assert [len(sampler.get_next_samples()) for _ in range(2)] == [6, 6]
    assert not sampler.has_budget


def test_without_num_repeats_there_is_no_repeat_key():
    sampler = SobolSequenceSampler(bounds=[(0, 1)], budget=4, parameters=["x"], seed=1)
    assert sampler.parameters == ["x"]
    assert all("repeat" not in s for s in sampler.get_next_samples())


def test_num_repeats_must_be_positive():
    with pytest.raises(ValueError, match="num_repeats"):
        SobolSequenceSampler(bounds=[(0, 1)], budget=4, parameters=["x"], num_repeats=0)
