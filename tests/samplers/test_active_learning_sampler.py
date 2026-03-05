import pytest
import numpy as np
from enchanted_surrogates.samplers.active_learning_sampler import ActiveLearningSampler

# Helpers

# Helper to create Active Learning samplers
def make_sampler(n_dims=2, batch_size=5, budget=200, bounds=None):
    bounds = [(0.0, 1.0)] * n_dims
    parameters = [f"c{i}" for i in range(n_dims)]
    sampler: ActiveLearningSampler = ActiveLearningSampler(
        bounds=bounds,
        budget=budget,
        parameters=parameters,
        batch_size=batch_size
    )
    return sampler, bounds, parameters

# Helper for making seeding points
def make_seeding_points(
        bounds: list[tuple[float,float]],
        n_points: int,
        exclude_region: list[tuple[float,float]] | None = None,
        seed: int = 1,
    ) -> np.ndarray:
    
    rng = np.random.default_rng(seed)
    lows = np.array([b[0] for b in bounds])
    highs = np.array([b[1] for b in bounds])
    n_dims = len(bounds)

    points = []
    while len(points) < n_points:
        p = rng.uniform(lows,highs,size=n_dims)
        if exclude_region is not None:
            ex_lows = np.array([r[0] for r in exclude_region])
            ex_highs = np.array([r[1] for r in exclude_region])
            if np.all((p > ex_lows) & (p < ex_highs)):
                continue
        points.append(p)

    return np.array(points)

# Registering some set of observations, getting new y values for the function
def seed_observations(
    sampler: ActiveLearningSampler,
    parameters: list[str],
    bounds: list[tuple[float,float]],
    points: np.ndarray,
    objective=None,    
):
    lows = np.array([b[0] for b in bounds])
    highs = np.array([b[1] for b in bounds])
    if not np.all((points >= lows) & (points <= highs)):
        raise ValueError(
            f"Some points outside of bounds: {bounds}"
        )
    if objective is None:
        objective = lambda row: float(np.sum(row))
    for row in points:
        params = dict(zip(parameters, row))
        sampler.register_future((params, objective(row)))

# Tests

class TestInit:
    @pytest.mark.parametrize("n_dims", [1,2,5,10])
    def test_candidates_shape_scales_with_dims(self, n_dims):
        sampler, _, _ = make_sampler(n_dims=n_dims, batch_size=4)
        assert sampler.candidates.shape == (4, n_dims)
    
    def test_candidates_within_bounds(self):
        bounds = [(0.2, 0.8), (1.0, 5.0)]
        sampler, bounds, _ = make_sampler(n_dims=2, batch_size=20, bounds=bounds)
        lows = np.array([b[0] for b in bounds])
        highs = np.array([b[1] for b in bounds])
        assert np.all(sampler.candidates >= lows)
        assert np.all(sampler.candidates <= highs)

def test_parameter_order_preserved():
        sampler, bounds, parameters = make_sampler()
        point = make_seeding_points(bounds, n_points=1, seed=1)[0]
        sampler.register_future((dict(zip(parameters, point)), 0.0))
        assert sampler.X_obs[0, 0] == pytest.approx(point[0])
        assert sampler.X_obs[0, 1] == pytest.approx(point[1])

class TestFallbackSamples:
    def test_returns_correct_batch_size(self):
        sampler, _, _ = make_sampler(batch_size=5)
        assert len(sampler.get_fallback_samples()) == 5
    
    def test_samples_within_bounds(self):
        bounds = [(0.2,0.8),(1.0,5.0)]
        sampler, bounds, _ = make_sampler(bounds=bounds)
        lows = np.array([b[0] for b in bounds])
        highs = np.array([b[1] for b in bounds])
        samples = sampler.get_fallback_samples()
        for s in samples:
            values = np.array(list(s.values()))
            assert np.all(values >= lows)
            assert np.all(values <= highs)
    
    @pytest.mark.parametrize("n_dims", [1,2,5])
    def test_samples_are_dicts_with_correct_keys(self, n_dims):
        sampler, _, parameters = make_sampler(n_dims=n_dims)
        for s in sampler.get_fallback_samples():
            assert set(s.keys()) == set(parameters)
    
    def test_submitted_increments_by_batch_size(self):
        sampler, _, _ = make_sampler(batch_size=5)
        sampler.get_fallback_samples()
        assert sampler.submitted == 5

class TestGetNextSamples:
    @pytest.mark.parametrize("n_dims", [1, 2, 5, 10])
    def test_returns_correct_batch_size(self, n_dims):
        sampler, bounds, parameters = make_sampler(n_dims=n_dims, batch_size=4)
        points = make_seeding_points(bounds, n_points=10)
        seed_observations(sampler, parameters, bounds, points)
        sampler.candidates = make_seeding_points(bounds, n_points=50, seed=99)
        assert len(sampler.get_next_samples()) == 4

    @pytest.mark.parametrize("n_dims", [1, 2, 5, 10])
    def test_output_keys_match_parameters_and_stays_within_bounds(self, n_dims):
        sampler, bounds, parameters = make_sampler(n_dims=n_dims)
        lows = np.array([b[0] for b in bounds])
        highs = np.array([b[1] for b in bounds])
        points = make_seeding_points(bounds, n_points=10)
        seed_observations(sampler, parameters, bounds, points)
        sampler.candidates = make_seeding_points(bounds, n_points=50, seed=99)
        for s in sampler.get_next_samples():
            assert set(s.keys()) == set(parameters)
            values = np.array([s[p] for p in parameters])
            assert np.all(values >= lows)
            assert np.all(values <= highs)
            
    def test_submitted_increments_after_active_query(self):
        sampler, bounds, parameters = make_sampler(batch_size=4)
        points = make_seeding_points(bounds, n_points=10)
        seed_observations(sampler, parameters, bounds, points)
        sampler.candidates = make_seeding_points(bounds, n_points=50, seed=99)
        sampler.get_next_samples()
        assert sampler.submitted == 4

    def test_smoke_many_rounds(self):
        sampler, bounds, parameters = make_sampler(n_dims=2, batch_size=5)
        for _ in range(50):
            samples = sampler.get_next_samples()
            points = np.array([[s[p] for p in parameters] for s in samples])
            seed_observations(sampler, parameters, bounds, points)
        assert sampler.X_obs.shape[0] == 50 * 5