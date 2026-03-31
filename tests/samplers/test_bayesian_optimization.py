from enchanted_surrogates.samplers.bayesian_optimization_sampler import (
    BayesianOptimizationSampler,
)

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def set_seeds(seed: int = 0) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_bo_sampler(n_dims=2, batch_size=5, budget=200, bounds=None):
    bounds = bounds or [(0.0, 1.0)] * n_dims
    parameters = [f"c{i}" for i in range(n_dims)]

    sampler = BayesianOptimizationSampler(
        bounds=bounds,
        budget=budget,
        parameters=parameters,
        batch_size=batch_size,
    )
    return sampler, bounds, parameters


def make_random_points(bounds, n_points, seed=0):
    rng = np.random.default_rng(seed)
    lows = np.array([b[0] for b in bounds], dtype=float)
    highs = np.array([b[1] for b in bounds], dtype=float)
    return rng.uniform(lows, highs, size=(n_points, len(bounds)))


def register_observations(sampler, parameters, points, objective):
    rows = []
    for p in points:
        rows.append(
            {
                **{k: float(v) for k, v in zip(parameters, p)},
                "output": float(objective(p)),
                "success": True,
            }
        )

    # The current sampler implementation expects an iterable, not a DataFrame.
    sampler.register_future(rows)


def test_bo_returns_correct_shape_and_bounds():
    set_seeds(0)
    sampler, bounds, parameters = make_bo_sampler(n_dims=3, batch_size=5)

    samples = sampler.get_next_samples()

    # Current implementation returns its initial candidate pool.
    assert len(samples[0]) == len(sampler.parameters)

    lows = np.array([b[0] for b in bounds], dtype=float)
    highs = np.array([b[1] for b in bounds], dtype=float)

    for s in samples:
        assert set(s.keys()) == set(parameters)
        values = np.array([s[p] for p in parameters], dtype=float)
        assert values.shape == (3,)
        assert np.all(values >= lows)
        assert np.all(values <= highs)


def test_bo_respects_existing_observations():
    set_seeds(1)
    sampler, bounds, parameters = make_bo_sampler(n_dims=2, batch_size=4)

    seed_points = make_random_points(bounds, 10, seed=1)

    def objective(x):
        return np.sum(x)

    register_observations(sampler, parameters, seed_points, objective)

    new_samples = sampler.get_next_samples()
    new_points = np.array(
        [[s[p] for p in parameters] for s in new_samples], dtype=float
    )

    # None of the proposed points should exactly duplicate an observed point.
    for p in new_points:
        assert not np.any(np.all(np.isclose(seed_points, p), axis=1))


def test_bo_improves_on_simple_quadratic_smoke():
    set_seeds(2)
    sampler, bounds, parameters = make_bo_sampler(n_dims=2, batch_size=3)

    # Minimum at (0.2, 0.2).
    center = np.array([0.2, 0.2], dtype=float)

    def objective(x):
        return float(np.sum((x - center) ** 2))

    best_distances = []

    for _ in range(8):
        samples = sampler.get_next_samples()
        points = np.array([[s[p] for p in parameters] for s in samples], dtype=float)

        distances = np.linalg.norm(points - center, axis=1)
        best_distances.append(float(np.min(distances)))

        register_observations(sampler, parameters, points, objective)

    # Later rounds should be at least as good as early rounds in this smoke test.
    assert min(best_distances[-3:]) <= min(best_distances[:3])
