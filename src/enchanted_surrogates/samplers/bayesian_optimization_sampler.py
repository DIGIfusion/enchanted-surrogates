# samplers/bayesian_optimization_sampler.py
"""
---

## Overview

This sampler Class uses Bayesian Optimization techniques to data efficiently
sample through the search space to yield optimial information gain as
specified by the acquisition strategy.
Bayesian Optimization sampler using the BoTorch library.

---
"""

from enchanted_surrogates.utils.logger import get_logger
from enchanted_surrogates.utils.precise_imports import import_parser
from enchanted_surrogates.samplers.base_sampler import Sampler

from typing import Any, Iterable, Mapping, Sequence
import os

# import matplotlib as mpl
# import scienceplots
import numpy as np
import botorch, gpytorch, torch
from botorch.optim import optimize_acqf
from botorch.utils.transforms import standardize, normalize, unnormalize

log = get_logger(__name__)


class BayesianOptimizationSampler(Sampler):
    """
    ## Configuration

    To use the `BayesianOptimizationSampler`, specify it in the configuration file as follows:

    ```yaml
    sampler:
        type: BayesianOptimizationSampler
        budget: 50
        initial_samples: 20
        acquisition_batch_size: 10
        acquisition_function: qEI
        random_fraction: 0.2
        bounds: [[0.0, 1.0], [1.0, 5.0]]
        parameters: ['x', 'y']
        observations: ['distance']
        base_run_dir: ./runs
        fully_bayesian: false
        async_samp: false
        failure_prob_filter: false
        ucb_beta: 2.0
        parser: Parser
        parser_config:
        key: value
    ```

    Attributes:
        initial_samples (int): Number of initial samples required.
        verbose (bool): Whether to print verbose output.
        fully_bayesian (bool): Whether to use fully Bayesian models.
        acquisition_batch_size (int): Number of samples in each acquisition batch.
        observations (list): List of observations.
        bounds (list): Bounds for the search space.
        acquisition_function (str): Acquisition function to use.
        random_fraction (float): Fraction of random samples.
        failure_prob_filter (bool): Whether to filter based on failure probability.
        ucb_beta (float): Beta parameter for UCB acquisition function.
        async_samp (bool): Whether to use asynchronous sampling.
        parameters (list): List of parameter names.
        parser (type): Parser type for collecting sample information.
        parser_config: Parser kwargs

    ---

    ## Assumptions and notes

     - The sampler assumes continuous numeric parameters and bounded search spaces.

     - Bayesian optimization relies on existing evaluation results stored in base_run_dir.

     - Sampling proceeds in two phases:
          Random sampling until initial_samples are collected.
          Model-based sampling using a Gaussian Process surrogate.

    ---
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        """
        Bayesian optimization sampler using BoTorch.

        The sampler follows the same lifecycle as the active-learning sampler:
        it owns its observation state and only exposes the three external methods
        expected by the orchestration layer: `__init__`, `get_next_samples`, and
        `register_future`.

        Configuration
        -------------
        bounds : list[tuple[float, float]]
            Per-parameter lower and upper bounds.
        parameters : list[str]
            Parameter names in the same order as `bounds`.
        budget : int, optional
            Total number of samples allowed. Stored for bookkeeping.
        initial_samples : int, optional
            Number of random samples to return before surrogate-driven sampling.
        acquisition_batch_size : int, optional
            Number of candidates returned per call once the surrogate is active.
        acquisition_function : str, optional
            One of: `qLEI`, `qUCB`, `qEI`, `qPI`, `EI`, `LEI`.
        random_fraction : float, optional
            Fraction of each batch reserved for random exploration after warm-up.
        async_samp : bool, optional
            If true, a call may return a single random point with probability
            `random_fraction`, otherwise it behaves like synchronous acquisition.
        failure_prob_filter : bool, optional
            If true and failure data are available, proposed candidates are filtered
            by a secondary failure model.
        parser : type or str, optional
            Parser used to reconstruct old run-directory results when `base_run_dir`
            is supplied.
        parser_config : dict, optional
            Keyword arguments forwarded to parser construction.
        observations : list[str], optional
            Observation field names. Multiple fields are summed into the scalar
            target used by the GP, matching the legacy behavior.

        """

        log.info("INITIALISING BAYESIAN OPTIMIZATION SAMPLER")

        self.base_run_dir = kwargs.get("base_run_dir", "")
        self._budget = kwargs.get("budget", 20)
        self.bounds = kwargs.get("bounds", [None])
        self.parameters = kwargs.get("parameters", [])

        self.observations = kwargs.get("observations", [None])
        self.parser_type = kwargs.get("parser", None)
        self.parser_config = kwargs.get("parser_config", {}) or {}

        self.initial_samples = kwargs.get("initial_samples", 50)
        self.acq_batch_size = kwargs.get("acquisition_batch_size", 20)
        self.acq_function = kwargs.get("acquisition_function", "qLEI")
        self.random_fraction = kwargs.get("random_fraction", 0.2)
        self.fail_p_filter = kwargs.get("failure_prob_filter", False)
        self.ucb_beta = kwargs.get("ucb_beta", 2.0)
        self.async_samp = kwargs.get("async_samp", False)

        self.fully_bayesian = kwargs.get("fully_bayesian", False)
        self.covar = kwargs.get("covar_kernel", "Matern-3/2")

        self.verbose = kwargs.get("verbose", False)

        # Legacy flags are kept as no-ops for compatibility.
        self.plot_GPR_flag = kwargs.get("plot_GPR", False)
        self.plot_GPR_file = kwargs.get("plot_file", False)
        self.plot_frequency = kwargs.get("plot_frequency", 1)
        self.plot_debug = kwargs.get("plot_debug", False)
        self.plot_progress = kwargs.get("plot_progress", False)
        self.plot_labels = kwargs.get("plot_labels", None)
        self.GPR_plot_dim = kwargs.get("GPR_plot_dim", [0])

        self.submitted = 0
        self.futures = []

        self.parser = None
        if self.parser_type is not None:
            self.parser = import_parser(self.parser_type, self.parser_config)

        # Internal state used by the new workflow.
        self.X_obs = np.zeros((0, len(self.bounds)), dtype=float)
        self.y_obs = np.zeros((0,), dtype=float)
        self.X_failed = np.zeros((0, len(self.bounds)), dtype=float)
        self.y_failed = np.zeros((0,), dtype=float)

        # Backward-compatible state used by the old GP pipeline.
        self.result_dictionary = [None]
        self.result_dictionary_failed = [None]
        self.model = None
        self.model_failed = None
        self.best_f = None
        self.best_f_loc = None

        self.seen_run_dirs: set[str] = set()

    def get_next_samples(self):
        """
        Return the next batch of candidate parameter dictionaries.

        The sampler first refreshes its internal state from any completed futures
        and, if configured, from the legacy run directory parser path. It then
        either:
        - returns random warm-up points until `initial_samples` observations are
          available, or
        - fits the surrogate and acquires new candidates with BoTorch.

        Returns
        -------
        list[dict[str, float]]
            Batch of parameter dictionaries in the order expected by the
            orchestration layer.
        """
        self.sync_from_run_directory()

        if self._n_observations() < self.initial_samples:
            return self.get_random_samples(self.acq_batch_size)

        self.train_surrogate()
        acq = self.build_acquisition()

        batch_samples: list[dict[str, float]] = []

        # Synchronous mode keeps a small random exploration fraction in the batch.
        if self.async_samp:
            if torch.rand(1).item() < self.random_fraction:
                batch_samples = self.get_random_samples(1)
                self.submitted += len(batch_samples)
                return batch_samples
            qval = 1
        else:
            random_count = int(self.random_fraction * self.acq_batch_size)
            model_count = max(int((1 - self.random_fraction) * self.acq_batch_size), 1)
            batch_samples.extend(self.get_random_samples(random_count))
            qval = model_count

        candidates = self.optimize_candidates(acq, qval=qval)
        if self.fail_p_filter and self.model_failed is not None:
            candidates = self.apply_failure_filter(candidates, acq)

        for row in candidates:
            batch_samples.append({p: float(v) for p, v in zip(self.parameters, row)})

        self.submitted += len(batch_samples)
        return batch_samples

    def register_future(self, future):
        """
        Register a completed evaluation.

        Accepted inputs
        ---------------
        - mapping with keys `params` and `y`
        - mapping containing parameter columns directly plus one or more
          observation columns
        - dataframe-like object with parameter columns and observation columns
        - a list/tuple of the above, which will be ingested item by item

        Notes
        -----
        The sampler stores observations internally and also keeps a small
        compatibility layer that mirrors the legacy `result_dictionary` fields.
        If a `failure` field is present and evaluates truthy, the record is routed
        to the failure model dataset as well.
        """
        if not future:
            return
        self.futures.append(future)
        self.ingest_future(future)
        self.refresh_result_dictionaries()

    def sync_from_run_directory(self):
        """
        Rebuild the internal dataset from the legacy run directory path.

        This keeps the old parser-based workflow usable without making it a
        public method. It only runs when both a parser and `base_run_dir` are
        available.
        """
        if self.parser is None or not self.base_run_dir:
            return
        if not os.path.isdir(self.base_run_dir):
            return

        try:
            dirlist = os.listdir(self.base_run_dir)
        except OSError:
            return

        skiplist = [
            "yaml",
            "worker_out",
            "FINISHED",
            ".pkl",
            ".csv",
            "_RUN",
            "GPR",
            "Fig",
        ]

        changed = False
        for dirname in dirlist:
            if any(tag in dirname for tag in skiplist):
                continue

            run_dir = os.path.join(self.base_run_dir, dirname)
            if run_dir in self.seen_run_dirs:
                continue

            try:
                sample_dict = self.parser.collect_sample_information(
                    run_dir,
                    self.observations,
                )
            except Exception as exc:  # pragma: no cover - defensive guard
                log.warning("Could not parse run directory %s: %s", run_dir, exc)
                continue

            self.seen_run_dirs.add(run_dir)
            self.ingest_sample_dict(sample_dict, run_dir=run_dir)
            changed = True

        if changed:
            self.refresh_result_dictionaries()

    def ingest_future(self, future: Any):
        """
        Convert one completed result payload into internal arrays.

        This method accepts several lightweight shapes so the orchestration
        layer can pass whatever it already has available.
        """
        if isinstance(future, (list, tuple)):
            for item in future:
                self.ingest_future(item)
            return

        if hasattr(future, "to_dict") and not isinstance(future, dict):
            # Works for pandas Series/DataFrame-like rows.
            try:
                future = future.to_dict()
            except Exception:
                future = future

        if isinstance(future, dict):
            if "params" in future and "y" in future:
                params = future["params"]
                y_val = future["y"]
                failure = future.get("failure", 0)
                self.append_observation(params, y_val, failure=failure)
                return

            if all(param in future for param in self.parameters):
                params = {k: future[k] for k in self.parameters}
                y_val = self.extract_target_from_mapping(future)
                failure = future.get("failure", 0)
                self.append_observation(params, y_val, failure=failure)
                return

        raise TypeError(
            "Unsupported future payload. Expected a mapping with params/y, "
            "parameter columns plus observation fields, or an iterable of those."
        )

    def ingest_sample_dict(
        self, sample_dict: Mapping[str, Any], run_dir: str | None = None
    ):
        """
        Ingest a sample dictionary returned by the legacy parser.

        The parser path usually returns keys like `inputs`, `distances`, `failure`,
        and `run_dir`. The observation values are reduced to a scalar by summing
        across the distance vector, which preserves the old sampler behavior.
        """
        if sample_dict is None:
            return

        if sample_dict.get("failure", 0) not in (0, 0.0, False, None):
            x = sample_dict.get("inputs", None)
            y = sample_dict.get("distances", None)
            if x is None or y is None:
                return
            x_arr = np.asarray(x, dtype=float)
            if x_arr.ndim == 1:
                x_arr = x_arr.reshape(1, -1)
            y_arr = np.asarray(y, dtype=float)
            y_scalar = float(np.sum(y_arr))
            self.X_failed = np.vstack([self.X_failed, x_arr.reshape(1, -1)])
            self.y_failed = np.append(self.y_failed, y_scalar)
            return

        x = sample_dict.get("inputs", None)
        y = sample_dict.get("distances", None)
        if x is None or y is None:
            return

        x_arr = np.asarray(x, dtype=float)
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(1, -1)

        y_arr = np.asarray(y, dtype=float)
        y_scalar = float(np.sum(y_arr))

        self.X_obs = np.vstack([self.X_obs, x_arr.reshape(1, -1)])
        self.y_obs = np.append(self.y_obs, y_scalar)

    def append_observation(self, params: Mapping[str, Any], y: Any, failure: Any = 0):
        """Append a single observation to the in-memory dataset."""
        x = np.array([float(params[p]) for p in self.parameters], dtype=float).reshape(
            1, -1
        )
        y_scalar = float(np.asarray(y, dtype=float).squeeze())

        if failure not in (0, 0.0, False, None):
            self.X_failed = np.vstack([self.X_failed, x])
            self.y_failed = np.append(self.y_failed, y_scalar)
        else:
            self.X_obs = np.vstack([self.X_obs, x])
            self.y_obs = np.append(self.y_obs, y_scalar)

    def extract_target_from_mapping(self, payload: Mapping[str, Any]) -> float:
        """
        Reduce a mapping to a scalar target value.

        The legacy sampler used `distances` and summed over their components.
        The new workflow keeps that convention.
        """
        if len(self.observations) == 1 and self.observations[0] in payload:
            return float(
                np.asarray(payload[self.observations[0]], dtype=float).squeeze()
            )

        values = []
        for key in self.observations:
            if key in payload:
                values.append(np.asarray(payload[key], dtype=float))
        if values:
            return float(np.sum(values))
        if "output" in payload:
            return float(np.asarray(payload["output"], dtype=float).squeeze())
        raise KeyError(
            "Could not determine target value from payload. Provide an `y` key "
            "or one or more observation fields named in `observations`."
        )

    def refresh_result_dictionaries(self):
        """
        Keep legacy result dictionary attributes in sync with the internal arrays.

        This allows any downstream code that still inspects these attributes to
        continue working, even though the public workflow no longer calls the
        original builder directly.
        """
        if len(self.X_obs) > 0:
            self.result_dictionary = {
                "inputs": self.X_obs.tolist(),
                "distances": [[float(v)] for v in self.y_obs.tolist()],
                "failure": [0 for _ in range(len(self.y_obs))],
            }
        else:
            self.result_dictionary = [None]

        if len(self.X_failed) > 0:
            self.result_dictionary_failed = {
                "inputs": self.X_failed.tolist(),
                "distances": [[float(v)] for v in self.y_failed.tolist()],
                "failure": [1 for _ in range(len(self.y_failed))],
            }
        else:
            self.result_dictionary_failed = [None]

    # Fitting the GPR.
    def train_surrogate(self):
        if self.verbose:
            log.info("FITTING THE GPR")
        # Presently implemented as single objective model. Therefore,
        # sum over the distances and norm
        distances = torch.tensor(self.result_dictionary["distances"][:])
        distances = torch.sum(distances, axis=1)
        distances = standardize(distances)

        inputs = torch.tensor(self.result_dictionary["inputs"][:])
        bounds = torch.tensor(self.bounds)
        input_vector = normalize(inputs, bounds.T)

        # Check if normzalized inputs are below 0 or larger than 1.
        # dummy = torch.abs(input_vector - 0.5)
        # dummy = torch.max(dummy, axis=1).values < 0.5
        # idx = torch.where(dummy)
        # input_vector = input_vector[idx]
        # distances = distances[idx]

        distances = distances.unsqueeze(distances.ndim)
        # Multiply by -1 the task to a maximization problem.
        distances = -distances

        # Default covar module
        covar_module = gpytorch.kernels.MaternKernel(nu=1.5)

        if self.covar == "Matern-5/2":
            covar_module = gpytorch.kernels.MaternKernel(nu=2.5)
        if self.covar == "Matern-3/2":
            covar_module = gpytorch.kernels.MaternKernel(nu=1.5)
        if self.covar == "Matern-1/2":
            covar_module = gpytorch.kernels.MaternKernel(nu=0.5)
        if self.covar == "RBF":
            covar_module = gpytorch.kernels.RBFKernel(ard_num_dims=len(bounds.T))
        if self.fully_bayesian:
            if self.verbose:
                log.info("SaasFullyBayesianSingleTaskGP")
            # Default kernel is Matern-5/2
            gp = botorch.models.fully_bayesian.SaasFullyBayesianSingleTaskGP(
                input_vector, distances
            )
            gp.covar_module = covar_module
            botorch.fit.fit_fully_bayesian_model_nuts(gp)
        else:
            if self.verbose:
                log.info("SingleTaskGP")
            gp = botorch.models.SingleTaskGP(
                input_vector, distances, covar_module=covar_module
            )
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(
                likelihood=gp.likelihood, model=gp
            )
            botorch.fit.fit_gpytorch_mll(mll)

        self.model = gp
        self.best_f, idx = torch.max(distances, 0)
        self.best_f_loc = input_vector[idx, :]
        if self.result_dictionary_failed != [None]:
            inp = torch.tensor(self.result_dictionary["inputs"])
            inp_f = torch.tensor(self.result_dictionary_failed["inputs"])
            f_0 = torch.tensor(self.result_dictionary["failure"], dtype=torch.float64)
            f_1 = torch.tensor(
                self.result_dictionary_failed["failure"], dtype=torch.float64
            )

            gp_failed = botorch.models.SingleTaskGP(
                normalize(torch.cat([inp, inp_f]), bounds.T),
                torch.cat([f_0, f_1]).unsqueeze(0).T,
            )
            mll_gp_failed = gpytorch.mlls.ExactMarginalLogLikelihood(
                likelihood=gp_failed.likelihood, model=gp_failed
            )

            botorch.fit.fit_gpytorch_mll(mll_gp_failed)
            self.model_failed = gp_failed

    def build_acquisition(self):
        """
        Build the configured ByTorch acquisition function from the trained model
        """
        if self.model is None or self.best_f is None:
            raise RuntimeError("Surroage model has not been trained")

        match self.acq_function:
            case "qLEI":
                return botorch.acquisition.qLogExpectedImprovement(
                    model=self.model, best_f=self.best_f
                )
            case "qUCB":
                return botorch.acquisition.qUpperConfidenceBound(
                    model=self.model, beta=self.ucb_beta
                )
            case "qEI":
                return botorch.acquisition.qExpectedImprovement(
                    model=self.model, best_f=self.best_f
                )
            case "qPI":
                return botorch.acquisition.qProbabilityOfImprovement(
                    model=self.model, best_f=self.best_f
                )
            case "EI":
                return botorch.acquisition.ExpectedImprovement(
                    model=self.model, best_f=self.best_f
                )
            case "LEI":
                return botorch.acquisition.LogExpectedImprovement(
                    model=self.model, best_f=self.best_f
                )
            case _:
                raise ValueError(
                    "Unsupported acquisition function:", f"{self.acq_function}"
                )

    def get_random_samples(self, n: int) -> list[dict[str, float]]:
        """Generate `n` random samples uniformly within the configured bounds."""
        if n <= 0:
            return []

        samples = []

        for _ in range(n):
            params = [
                torch.distributions.Uniform(lb, ub).sample().item()
                for (lb, ub) in self.bounds
            ]
            param_dict = dict(zip(self.parameters, params))
            samples.append(param_dict)

        return samples

    def optimize_candidates(self, acq, qval: int) -> torch.Tensor:
        """
        Optimize the acquisition function over the bounded domain.

        Returns
        -------
        torch.Tensor
            Candidate tensor in the original parameter space.
        """
        if qval <= 0:
            return torch.empty((0, len(self.bounds)), dtype=torch.float64)

        lower_bound = torch.zeros(len(self.bounds), dtype=torch.float64).unsqueeze(0)
        upper_bound = torch.ones(len(self.bounds), dtype=torch.float64).unsqueeze(0)
        boundtensor = torch.cat((lower_bound, upper_bound))

        candidates, _ = optimize_acqf(
            acq,
            bounds=boundtensor,
            sequential=False,
            q=qval,
            num_restarts=10,
            raw_samples=1024,
        )

        bounds = torch.tensor(self.bounds, dtype=torch.float64)
        return unnormalize(candidates, bounds.T)

    def apply_failure_filter(self, candidates: torch.Tensor, acq) -> torch.Tensor:
        """
        Filter candidates using the failure-probability model.

        This preserves the legacy idea of rejecting points likely to fail while
        keeping the logic private to the sampler.
        """
        if self.model_failed is None or len(candidates) == 0:
            return candidates

        bounds = torch.tensor(self.bounds, dtype=torch.float64)
        accepted = []
        target_len = candidates.size(dim=0)

        norm_inp = normalize(candidates, bounds.T)
        pred = self.model_failed(norm_inp).mean.squeeze(-1)

        for i in range(len(pred)):
            if torch.rand(1).item() > float(pred[i]):
                accepted.append(candidates[i, :].detach().cpu().numpy())

        if len(accepted) < target_len:
            refill = self.optimize_candidates(acq, qval=target_len - len(accepted))
            accepted_tensor = (
                torch.tensor(accepted, dtype=torch.float64)
                if accepted
                else torch.empty((0, len(self.bounds)), dtype=torch.float64)
            )
            if len(refill) > 0:
                return torch.cat([accepted_tensor, refill], dim=0)
            return accepted_tensor

        return torch.tensor(accepted, dtype=torch.float64)

    def skip(self, index):
        return

    def _n_observations(self) -> int:
        """Return the number of non-failed observations currently available."""
        return int(len(self.y_obs))

    def _has_failure_data(self) -> bool:
        """Return True when failure examples are available for filtering."""
        return len(self.X_failed) > 0
