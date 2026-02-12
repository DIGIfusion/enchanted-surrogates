import numpy as np

from skactiveml.pool import (
    GreedySamplingX,
    GreedySamplingTarget,
    QueryByCommittee,
    KLDivergenceMaximization,
)
from sklearn.ensemble import BaggingRegressor
from skactiveml.regressor import NICKernelRegressor, SklearnRegressor

# from skactiveml.utils import call_func, is_labeled, MISSING_LABEL
from scipy.stats import uniform

from enchanted_surrogates.samplers.base_sampler import Sampler


class ActiveLearningSampler(Sampler):
    """

    ---
    TODO: Fix this
    ## Overview

    The random sampler generates samples randomly within the specified bounds
    for each parameter.

    ---

    ## Configuration

    To use the `RandomSampler`, specify it in the configuration file as in following example:

    ```yaml
      sampler:
        type: RandomSampler
        parameters: ['x', 'y']
        bounds: [[1, 10], [0, 1]]
        num_samples: 100
    ```

    Attributes:
        self.budget (int): Total number of samples that can be generated.
        self.bounds (list[tuple[float, float]]): Lower and upper bounds for each parameter.
        self.parameters (list[str]): Names of the parameters to be sampled.
        self.batch_size (int): Number of samples returned per batch (defaults to full budget).
        self.submitted (int): Counter tracking how many samples have been generated so far.
        BATCH_SAMPLE_SIZE (int): Class-level default batch size (currently 1).

    ---

    ## Assumptions and Notes

     - Parameter values are sampled independently using a uniform distribution
      (`np.random.uniform`) within the specified bounds.

     - The sampler generates samples in batches; the batch size is controlled
      by `batch_size`.

     - If `batch_size` is not provided, it defaults to the full sampling budget.
     - The sampler does not adapt based on previous evaluations.

     - The sampler maintains an internal counter (`self.submitted`) tracking
      the number of generated samples.

     - The current implementation assumes continuous parameter spaces.

    ---

    """

    BATCH_SAMPLE_SIZE = 1

    def __init__(self, bounds, budget, parameters, **kwargs):
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
        self.candidates = np.sort(
            np.concatenate(
                [
                    uniform.rvs(loc=min, scale=max - min, size=self.batch_size)
                    for min, max in bounds
                ]
            )
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
        self.qs = GreedySamplingTarget()  # TODO: import from config

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
        future : tuple or dict
            Either:
                (params_dict, y_value)
            or
                {"params": params_dict, "y": y_value}

        Adds the observation to the internal dataset.
        """
        if isinstance(future, dict):
            params = future["params"]
            y = future["y"]
        else:
            params, y = future

        arr = np.array([params[k] for k in self.parameters]).reshape(1, -1)

        self.X_obs = np.vstack([self.X_obs, arr])
        self.y_obs = np.append(self.y_obs, y)

    def register_futures(self, futures):
        """
        Register multiple completed evaluations.

        Parameters
        ----------
        futures : iterable
            Iterable of tuples or dicts accepted by `register_future`.
        """
        for f in futures:
            self.register_future(f)
