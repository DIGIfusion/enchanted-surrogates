"""

"""
import numpy as np

from sklearn.ensemble import BaggingRegressor
from skactiveml.regressor import NICKernelRegressor, SklearnRegressor

# from skactiveml.utils import call_func, is_labeled, MISSING_LABEL
from scipy.stats import uniform

from enchanted_surrogates.samplers.base_sampler import Sampler
from enchanted_surrogates.utils.precise_imports import cached_import_external


class ActiveLearningSampler(Sampler):
    """

    ## Overview

    ActiveLearningSampler provides an active-learning driven sampler that
    proposes parameter configurations for evaluation using a surrogate
    regressor and a pool-based query strategy.

    > [!Note]
    > This sampler requires the `activelearning` optional dependency to function.
    > See installation guide for more details.

    ### Purpose
    - Maintain a pool of candidate parameter vectors and an internal
      dataset of observed (parameter, objective) pairs.
    - During a warmup period or when no observations exist, produce random
      samples within provided bounds.
    - After warmup, fit a surrogate regression model to observations and
      use a specified pool-based query strategy to select the most
      informative candidates to evaluate next.

    ### Key behavior
    - Samples continuous parameter spaces; each parameter has an
      independent [low, high] bound.
    - Produces samples in batches (controlled by batch_size).
    - Tracks how many samples have been generated via **self.submitted**
      and stops at **self.budget** if enforced externally.
    - Uses a surrogate regressor (default: `NICKernelRegressor` wrapped
      in `BaggingRegressor` and `SklearnRegressor`) to estimate objective
      values and uncertainties.
    - Uses a pool-based query strategy (loaded dynamically from a module)
      to select candidates from the current candidate pool.

    ### Configuration / Inputs
    - **bounds** (list of (low, high) tuples): per-parameter sampling ranges.
    - **budget** (int): total number of samples allowed.
    - **parameters** (list of str): names of parameters; order must match
      bounds.
    - **query_strategy** (str): import path or name used by
      `cached_import_external` to load a pool query strategy from
      `skactiveml.pool`.
    - Optional kwargs:
      - **batch_size** (int): number of samples returned per call
        (defaults to budget).

    ### Outputs
    - `get_next_samples()` returns a list of dicts mapping parameter names
      to sampled values for the next batch.
    - `register_future(future)` accepts either `(params_dict, y)` or
      `{"params": params_dict, "y": y}` and appends the observation to the
      internal dataset.

    ### Implementation notes
    - Candidates are sampled once at initialization from uniform
      distributions over bounds; batch selection is done from this pool.
    - Warmup behavior: while `self.submitted < self.warmup` or no
      observations exist, `get_next_samples()` returns random fallback
      samples.
    - The surrogate model and query strategy are dynamically
      instantiated; these should be made configurable (for example via
      external config) if different regressors or ensemble parameters are
      required.
    """

    def __init__(self, bounds, budget, parameters, query_strategy, **kwargs):
        """
        Initializes the ActiveLearningSampler.

        Args:
          bounds (list[tuple[float, float]]): Lower and upper bounds for each parameter.
          budget (int): Total number of samples that can be generated.
          parameters (list[str]): Names of the parameters to be sampled. The order must correspond to the order of bounds.
          batch_size (int, optional): Number of samples returned per call to `get_next_samples`. Defaults to the full sampling budget.
        """
        self.budget = budget
        self.bounds = bounds
        self.parameters = parameters
        self.batch_size = kwargs.get("batch_size", self.budget)

        # Parameter values
        n_candidates = self.batch_size  # or whatever pool size you want
        n_parameters = len(self.bounds)
        lows = np.array([b[0] for b in self.bounds])
        highs = np.array([b[1] for b in self.bounds])

        self.candidates = uniform.rvs(
            loc=lows, scale=highs - lows, size=(n_candidates, n_parameters)
        )

        self.X_obs = np.zeros((0, len(bounds)))
        self.y_obs = np.zeros((0,))

        self.submitted = 0
        self.warmup = 0

        # Regression

        # TODO: import from config instead
        nick_regressor = NICKernelRegressor()
        self.model = SklearnRegressor(BaggingRegressor(nick_regressor, n_estimators=5))

        # Query strategy
        self.qs = cached_import_external(query_strategy, "skactiveml.pool")()

    def get_next_samples(self) -> list[dict]:
        """
        Select the next batch of samples.

        If insufficient observations are available (warmup phase),
        random samples are returned.

        Otherwise, the surrogate model is trained and the query
        strategy selects the most informative candidates.

        Returns
        -------
        list[dict[str, float]]
            A batch of parameter dictionaries.
        """

        if self.submitted < self.warmup or len(self.X_obs) == 0:
            return self.get_fallback_samples()

        self.model.fit(self.X_obs, self.y_obs)

        query_indices = self.qs.query(
            X=self.X_obs,
            y=self.y_obs,
            reg=self.model,
            candidates=self.candidates,
            batch_size=self.batch_size,
        )

        selected = self.candidates[query_indices]
        self.submitted += len(selected)

        return [
            {key: value for key, value in zip(self.parameters, row)} for row in selected
        ]

    def get_fallback_samples(self) -> list[dict]:
        """
        Generate random samples within bounds.

        Used during warmup phase or when no observations
        are available.

        Returns
        -------
        list[dict[str, float]]
            Random parameter configurations.
        """
        list_param_dicts = []
        for _ in range(self.batch_size):
            params = [np.random.uniform(low, high) for low, high in self.bounds]
            param_dict = {key: value for key, value in zip(self.parameters, params)}
            list_param_dicts.append(param_dict)

        self.submitted += len(list_param_dicts)

        return list_param_dicts

    def register_future(self, future):
        """
        Register a completed evaluation.

        Parameters
        ----------
        future : a PD dataframe with output and params

        Adds the observation to the internal dataset.
        """
        X = future[self.parameters].to_numpy()
        y = future["output"].to_numpy()

        self.X_obs = np.vstack([self.X_obs, X])
        self.y_obs = np.append(self.y_obs, y)
