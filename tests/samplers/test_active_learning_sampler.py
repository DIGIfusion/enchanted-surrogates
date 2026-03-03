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
    
class TestTargetOptimizing:
    @pytest.mark.parametrize("objective,uncertain_region,certain_region,bounds,description", [
        (
            lambda x: x ** 2,
            (1.5, 2.0),   # high curvature means model uncertain here with sparse coverage
            (0.0, 0.5),   # low curvature means model confident here
            [(0.0,2.0)],
            "quadratic: uncertainty concentrates at high x",
        ),
        (
            lambda x: np.sin(x),
            (0.0, 0.2),   # near the zero, sparse coverage
            (1.4, 1.6),   # near the peak, dense coverage
            [(0.0,1.6)],
            "sinusoidal: uncertainty at trough when peak is well covered",
        ),
        (
            lambda x: np.exp(x),
            (1.5, 2.0),   # steep gradient
            (0.0, 0.5),   # flat region
            [(0.0,2.0)],
            "exponential: uncertainty where gradient is steepest",
        ),
    ])
    def test_qs_prefers_uncertain_region_by_objective(
        self, objective, uncertain_region, certain_region, bounds, description
    ):
        # QS should prefer regions where there is more uncertainity

        sampler, _, parameters = make_sampler(n_dims=1, batch_size=20, bounds=bounds)

        # Dense observations in the certain region
        certain_pts = make_seeding_points([certain_region], n_points=20, seed=1)
        seed_observations(
            sampler, parameters, 
            bounds=[certain_region],  # temporarily narrow bounds for generation
            points=certain_pts,
            objective=lambda row: objective(row[0])
        )

        # Sparse observations in the uncertain region
        sparse_pts = make_seeding_points([uncertain_region], n_points=3, seed=2)
        seed_observations(
            sampler, parameters,
            bounds=[uncertain_region],
            points=sparse_pts,
            objective=lambda row: objective(row[0])
        )

        uncertain_candidates = np.linspace(*uncertain_region, 20).reshape(-1, 1)
        certain_candidates = np.linspace(*certain_region, 20).reshape(-1, 1)
        sampler.candidates = np.vstack([uncertain_candidates, certain_candidates])

        samples = sampler.get_next_samples()
        xs = [s["c0"] for s in samples]

        n_uncertain = sum(uncertain_region[0] <= x <= uncertain_region[1] for x in xs)
        n_certain = sum(certain_region[0] <= x <= certain_region[1] for x in xs)

        assert n_uncertain > n_certain, (
            f"Failed for: {description}\n"
            f"Uncertain region {uncertain_region}: {n_uncertain} samples\n"
            f"Certain region {certain_region}: {n_certain} samples\n"
            "QS should prefer regions where the model struggles with the objective shape."
        )
    
    @pytest.mark.parametrize("objective,high_gradient_region,low_gradient_region,bounds,description", [
        (
            lambda x: -(x - 0.3) ** 2,
            (0.7, 1.0),   # far from optimum, steep gradient
            (0.25, 0.35), # near optimum, flat
            [(0.0, 1.0)],
            "quadratic: gradient highest far from optimum",
        ),
        (
            lambda x: np.sin(x),
            (0.0, 0.2),   # near zero crossing, steep
            (1.5, 1.6),   # near peak, flat
            [(0.0, 1.6)],
            "sinusoidal: gradient highest at zero crossings",
        ),
        (
            lambda x: np.exp(x),
            (1.5, 2.0),   # steep
            (0.0, 0.5),   # flat
            [(0.0, 2.0)],
            "exponential: gradient increases with x",
        ),
    ])
    def test_qs_queries_high_gradient_region_with_uniform_distribution(
        self, objective, high_gradient_region, low_gradient_region, bounds, description
    ):
        # When observations are uniform, high gradient regions should be queried next
        sampler, bounds, parameters = make_sampler(n_dims=1, batch_size=3, bounds=bounds)

        uniform_points = make_seeding_points(bounds=bounds, n_points=20, seed=1)
        seed_observations(
            sampler, parameters, bounds, uniform_points,
            objective=lambda row: objective(row[0])
        )

        high_grad_candidates = np.linspace(*high_gradient_region, 20).reshape(-1, 1)
        low_grad_candidates = np.linspace(*low_gradient_region, 20).reshape(-1, 1)
        sampler.candidates = np.vstack([high_grad_candidates, low_grad_candidates])

        samples = sampler.get_next_samples()
        xs = [s["c0"] for s in samples]

        n_high = sum(high_gradient_region[0] <= x <= high_gradient_region[1] for x in xs)
        n_low = sum(low_gradient_region[0] <= x <= low_gradient_region[1] for x in xs)

        assert n_high > n_low, (
            f"Failed for: {description}\n"
            f"High gradient region {high_gradient_region}: {n_high} samples\n"
            f"Low gradient region {low_gradient_region}: {n_low} samples\n"
            "GreedySamplingTarget should prefer high-gradient regions."
        )

class TestIterative:
    def test_smoke_many_rounds(self):
        # Run 50 times and see if it works
        sampler, _, _ = make_sampler(n_dims=2, batch_size=5)
        for _ in range(50):
            samples = sampler.get_next_samples()
            sampler.register_futures([(s, sum(s.values())) for s in samples])
        assert sampler.X_obs.shape[0] == 50 * 5
    
    def test_warmup_then_gradient_seeking(self):
        # testing if QS finds high gradient sections
        bounds = [(0.0,2.0)]
        sampler, _, _ = make_sampler(n_dims=1, batch_size=5, bounds=bounds)

        assert len(sampler.X_obs) == 0
        warmup_samples = sampler.get_next_samples()
        assert len(warmup_samples) == sampler.batch_size

        objective = lambda x: np.exp(x)
        sampler.register_futures([
            (s,objective(s["c0"])) for s in warmup_samples
        ])
        sampler.batch_size = 1

        high_gradient = np.linspace(1.0, 2.0, 20).reshape(-1, 1)  # steep exp(x)
        low_gradient = np.linspace(0.0, 1.0, 20).reshape(-1, 1)   # flat exp(x)
        sampler.candidates = np.vstack([high_gradient, low_gradient])

        active_samples = sampler.get_next_samples()
        xs = [s["c0"] for s in active_samples]

        n_high = sum(x >= 1.0 for x in xs)
        n_low = sum(x <= 1.0 for x in xs)

        assert n_high > n_low, (
            f"After warmup, QS should prefer high-gradient region [1.0, 2.0] "
            f"but got {n_high} high vs {n_low} low gradient samples.\n"
            f"Active xs: {xs}"
        )