"""
Time-aware sparse-grid density-estimation (SGDE) boundary-uncertainty active
sampler.

Same overall structure as svm_time_aware_active_sampler.py /
extra_trees_time_aware_active_sampler.py, but the classifier is a set of
sparse-grid density estimators (one per class, via SG++'s pysgpp.LearnerSGDE)
instead of an SVM or a random forest. This follows the classification
approach used in:

  - The SG++ LearnerSGDE example/documentation:
    https://sgpp.github.io/SGpp/example_learnerSGDETest_py.html
  - Peherstorfer, Pflueger & Bungartz, "Density Estimation with Adaptive
    Sparse Grids for Large Data Sets", 2014 (and the earlier
    classification-via-density-estimation approach applied to MNIST digit
    recognition): fit one adaptively-refined sparse-grid density p_c(x) per
    class c from the training points of that class, then classify a new
    point x by the class maximizing the posterior

        posterior_c(x) = prior_c * p_c(x) / sum_c'(prior_c' * p_c'(x))

    with prior_c estimated as the empirical class frequency (this differs
    from a naive equal-prior vote, which would bias predictions on
    imbalanced classes toward whichever class's density happens to spike,
    e.g. from a small training count).
  - Bohn, Garcke & Griebel, "A sparse grid based method for generative
    dimensionality reduction of high-dimensional data" and related SGDE
    literature (https://linkinghub.elsevier.com/retrieve/pii/S0885064X10000257)
    for the surplus-based adaptive refinement strategy used to refine each
    class's grid during training.

A second ExtraTreesRegressor side-model predicts simulation cost (e.g. GENE
wallclock/simtime), exactly as in the other time-aware samplers: the
boundary sub-batch is ranked by a weighted combination of classification
uncertainty and predicted cost, while the exploration sub-batch stays purely
random and unbiased by predicted cost (coverage, not exploitation).

Two output variables are required: a categorical class label and a
continuous cost/time value, both produced by the same simulation.

Two acquisition modes are available (see _uncertainty), both based on the
posterior (i.e. label ambiguity, not grid/density resolution -- see note
below on why a surplus-based acquisition function was considered and
rejected for classification):
    - "margin":  gap between the top-2 classes' normalized posterior
      probabilities. Smallest gap = most uncertain.
    - "entropy": Shannon entropy of the normalized posterior over all
      classes. Highest entropy = most uncertain.

Why not a surplus-based acquisition function (unlike a regression SGDE
sampler, where the per-point hierarchical surplus directly measures
remaining approximation error in the predicted value and so is a natural
acquisition signal): here each class's surplus measures how well-resolved
*that class's density* is, not how ambiguous the *classification decision*
is at that point -- two different quantities. A point deep inside one
class's territory can have a large, unresolved density surplus (the density
there is just a complicated shape, unrelated to the class boundary) while
being nowhere near ambiguous; conversely a genuine decision boundary can
pass through a region where both classes' densities are already smooth and
well-resolved (low surplus for both) but the posterior is still ~50/50.
Surplus-based scoring would misprioritize both cases relative to
margin/entropy, so it isn't offered as an acquisition mode here.

Caveat specific to SGDE (unlike the SVM/ExtraTrees classifiers): each
per-class density is fit independently from only that class's points, so a
class with very few training points yields an unreliable (sometimes
near-degenerate) density -- see _fit_model's minimum-points-per-class
handling below. Also, SG++'s linear-basis SGDE solve can produce small
negative density values away from the training data (a known regularization
artifact); these are clipped to zero before use as posteriors.
"""

import os

import numpy as np
import pandas as pd
import pysgpp

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support

from enchanted_surrogates.samplers.parent_active_sampler import ParentActiveSampler
from enchanted_surrogates.samplers.svm_active_sampler import _in_hull
from enchanted_surrogates.utils.logger import get_logger

log = get_logger(__name__)


def _make_learner_sgde(dim, grid_kwargs, adaptivity_kwargs, solver_kwargs, regularization_kwargs):
    """
    Builds a fresh, untrained pysgpp.LearnerSGDE from plain-Python kwarg
    dicts, translating friendly string values (grid/regularization type
    names) to the corresponding pysgpp enum constants. A fresh instance is
    built on every fit (mirroring how the other samplers re-instantiate
    their sklearn models in _fit_model) rather than reusing/retraining one,
    since LearnerSGDE has no documented "reset and refit from scratch"
    entry point of its own.
    """
    grid = pysgpp.RegularGridConfiguration()
    grid.dim_ = dim
    grid.level_ = int(grid_kwargs.get("level", 3))
    grid_type_name = grid_kwargs.get("type", "Linear")
    grid.type_ = getattr(pysgpp, f"GridType_{grid_type_name}")

    adapt = pysgpp.AdaptivityConfiguration()
    adapt.numRefinements_ = int(adaptivity_kwargs.get("numRefinements", 3))
    adapt.numRefinementPoints_ = int(adaptivity_kwargs.get("numRefinementPoints", 5))
    adapt.refinementThreshold_ = float(adaptivity_kwargs.get("refinementThreshold", 0.0))
    adapt.percent_ = float(adaptivity_kwargs.get("percent", 0.1))
    adapt.coarsenInitialPoints_ = bool(adaptivity_kwargs.get("coarsenInitialPoints", False))

    solver = pysgpp.SLESolverConfiguration()
    solver.maxIterations_ = int(solver_kwargs.get("maxIterations", 1000))
    solver.eps_ = float(solver_kwargs.get("eps", 1e-10))
    solver.threshold_ = float(solver_kwargs.get("threshold", 1e-10))

    reg = pysgpp.RegularizationConfiguration()
    reg_type_name = regularization_kwargs.get("type", "Laplace")
    reg.type_ = getattr(pysgpp, f"RegularizationType_{reg_type_name}")
    reg.lambda_ = float(regularization_kwargs.get("lambda", 1e-6))

    cv = pysgpp.CrossvalidationConfiguration()
    cv.enable_ = False

    return pysgpp.LearnerSGDE(grid, adapt, solver, reg, cv)


def _sgde_pdf(learner, X):
    """
    Evaluates a fitted LearnerSGDE's density at every row of X (N, dim),
    returning a length-N numpy array clipped to >= 0 (see module docstring:
    SG++'s regularized linear-basis solve can yield small negative density
    values away from the training data, which are meaningless as
    probabilities/posteriors).
    """
    res = pysgpp.DataVector(X.shape[0])
    learner.pdf(pysgpp.DataMatrix(np.ascontiguousarray(X, dtype=float)), res)
    return np.clip(res.array(), 0.0, None)


def _margin_from_posterior(posterior):
    """Top1-vs-top2 gap of normalized posterior probabilities; smaller = more uncertain."""
    sorted_p = np.sort(posterior, axis=1)
    return sorted_p[:, -1] - sorted_p[:, -2]


def _entropy_from_posterior(posterior):
    """Shannon entropy (nats) of normalized posterior probabilities; larger = more uncertain."""
    p = np.clip(posterior, 1e-12, 1.0)
    return -np.sum(p * np.log(p), axis=1)


class SgdeTimeAwareActiveSampler(ParentActiveSampler):
    """
    Sparse-grid density-estimation (SGDE) classifier -- one adaptively
    refined sparse grid density per class, classified by posterior -- plus
    an ExtraTrees simulation-cost regressor, jointly driving hull-restricted
    boundary sampling.

    Configuration (in addition to ParentActiveSampler's):
        class_output_variable : str
            Column name of the categorical class label.
        time_output_variable : str
            Column name of the continuous simulation cost/time.
        acquisition_mode : str, optional
            Posterior-uncertainty score used to rank the boundary sub-batch:
            "margin" (top1-vs-top2 posterior gap, default) or "entropy"
            (Shannon entropy of the posterior). See the module docstring for
            why a surplus/grid-resolution-based mode is not offered here.
        min_points_per_class : int, optional
            Minimum number of training points a class must have before an
            SGDE density is fit for it. Classes below this count are
            excluded from the posterior (treated as zero-probability) until
            enough points are observed -- an SGDE density fit from a
            handful of points is unreliable and, worse, its coarse-grid
            normalization can make it spuriously dominate the posterior
            over classes with proper densities. Default: max(10, dim + 1).
        grid_kwargs : dict, optional
            Per-class sparse grid settings: ``level`` (int, default 3) and
            ``type`` (str, one of pysgpp's GridType_* suffixes, e.g.
            "Linear", "ModLinear"; default "Linear").
        adaptivity_kwargs : dict, optional
            Per-class refinement settings forwarded to
            pysgpp.AdaptivityConfiguration: ``numRefinements`` (default 3),
            ``numRefinementPoints`` (default 5), ``refinementThreshold``
            (default 0.0), ``percent`` (default 0.1),
            ``coarsenInitialPoints`` (default False).
        solver_kwargs : dict, optional
            Per-class CG solver settings forwarded to
            pysgpp.SLESolverConfiguration: ``maxIterations`` (default 1000),
            ``eps`` (default 1e-10), ``threshold`` (default 1e-10).
        regularization_kwargs : dict, optional
            Per-class regularization settings: ``type`` (str, one of
            pysgpp's RegularizationType_* suffixes, e.g. "Laplace",
            "Identity"; default "Laplace") and ``lambda`` (default 1e-6).
        time_regressor_kwargs : dict, optional
            Forwarded to sklearn.ensemble.ExtraTreesRegressor.
        exploration_per_batch : int, optional
            Unrestricted random exploration points drawn each batch (after
            the initial batch). Default: max(1, batch_size // 5).
        boundary_pool_fraction : float, optional
            Fraction of the hull-restricted remaining pool to rank by the
            combined uncertainty/cost score before drawing the boundary
            sub-batch. Default: 0.10.
        cost_lambda : float, optional
            Weight of predicted cost in the combined boundary score:
            ``score = uncertainty_rank - cost_lambda * predicted_time_rank``,
            both ranks in [0, 1] (1 = most uncertain / most expensive).
            Default: 0.5. Larger values favor cheaper points more strongly.
        max_predicted_time : float, optional
            Hard cutoff: pool points whose predicted simulation time exceeds
            this value are excluded from both the exploration and boundary
            sub-batches. Applies from the second batch onward, once a time
            model has been fit. If fewer than the requested number of points
            remain under the limit, a warning is logged and the sub-batch is
            filled with the cheapest available points regardless of the
            limit. Default: None (no filtering).
        unclassified_label : str or None, optional
            A class_output_variable value that carries no class information
            and should be excluded from training rather than learned as a
            real class. Rows with this label are dropped in register_future
            (the point is still "spent": the simulation ran, it just didn't
            yield a usable label). Set to None to disable this filtering and
            treat it as a real class instead. Default: "Unclassified".
        test_data_csv : str, optional
            Path to a held-out labeled CSV used for evaluation instead of
            K-fold CV. Inherited from ParentActiveSampler; see _load_test_set.
        test_set_accuracy_target : float, optional
            If set (requires test_data_csv), get_next_samples stops the run
            early as soon as an evaluate_model() call reports test-set
            accuracy below this threshold. Default: None.
        cpuh_budget : float, optional
            Total CPU-hours (core-hours) allowed across the whole run,
            computed from per-stage wallclock columns (gene_runtime_variable,
            helena_runtime_variable) times the corresponding per-run core
            counts. get_next_samples stops the run once the running total
            meets or exceeds this value. Default: None.
        gene_cores_per_run : int, optional
            Cores used by one GENE run. Required if cpuh_budget is set.
        helena_cores_per_run : int, optional
            Cores used by one HELENA run. Required if cpuh_budget is set.
        gene_runtime_variable : str, optional
            Column name of GENE's per-run wallclock time in seconds.
            Default: "runtime_sec_gene".
        helena_runtime_variable : str, optional
            Column name of HELENA's per-run wallclock time in seconds.
            Default: "runtime_sec_helena".
    """

    DEFAULT_GRID_KWARGS = dict(level=3, type="Linear")
    DEFAULT_ADAPTIVITY_KWARGS = dict(
        numRefinements=3, numRefinementPoints=5, refinementThreshold=0.0,
        percent=0.1, coarsenInitialPoints=False,
    )
    DEFAULT_SOLVER_KWARGS = dict(maxIterations=1000, eps=1e-10, threshold=1e-10)
    DEFAULT_REGULARIZATION_KWARGS = dict(type="Laplace", **{"lambda": 1e-6})
    DEFAULT_TIME_REGRESSOR_KWARGS = dict(n_estimators=200, bootstrap=False)
    VALID_ACQUISITION_MODES = ("margin", "entropy")

    def __init__(self, **kwargs):
        self.class_output_variable = kwargs.get("class_output_variable")
        self.time_output_variable = kwargs.get("time_output_variable")
        if not self.class_output_variable or not self.time_output_variable:
            raise ValueError(
                "SgdeTimeAwareActiveSampler requires both class_output_variable "
                "and time_output_variable in sampler_config."
            )
        # ParentActiveSampler.__init__ uses output_variables for row-filtering
        # (drop rows with NaN in either output) and output_dim bookkeeping.
        kwargs = dict(kwargs)
        kwargs["output_variables"] = [self.class_output_variable, self.time_output_variable]

        super().__init__(**kwargs)

        self.unclassified_label = kwargs.get("unclassified_label", "Unclassified")

        self.acquisition_mode = kwargs.get("acquisition_mode", "margin")
        if self.acquisition_mode not in self.VALID_ACQUISITION_MODES:
            raise ValueError(
                f"acquisition_mode must be one of {self.VALID_ACQUISITION_MODES}, "
                f"got {self.acquisition_mode!r}."
            )

        # _test_y (if a test set was loaded) has columns [class, time] since
        # output_variables was set to both above; split it apart and drop
        # unclassified_label rows the same way register_future does.
        if self._test_X is not None:
            test_y_class = self._test_y[:, 0]
            test_y_time = self._test_y[:, 1].astype(float)
            if self.unclassified_label is not None:
                keep = test_y_class != self.unclassified_label
                self._test_X = self._test_X[keep]
                test_y_class = test_y_class[keep]
                test_y_time = test_y_time[keep]
            self._test_y_class = test_y_class
            self._test_y_time = test_y_time
        else:
            self._test_y_class = None
            self._test_y_time = None

        # Narrow train_y into two separate arrays now that the task shape is
        # known (ParentActiveSampler initializes a single generic train_y).
        self.train_x = np.empty((0, self.input_dim), dtype=float)
        self.train_y_class = np.empty((0,), dtype=object)
        self.train_y_time = np.empty((0,), dtype=float)

        self.min_points_per_class = int(
            kwargs.get("min_points_per_class", max(10, self.input_dim + 1))
        )
        self.grid_kwargs = kwargs.get("grid_kwargs", None) or dict(self.DEFAULT_GRID_KWARGS)
        self.adaptivity_kwargs = kwargs.get("adaptivity_kwargs", None) or dict(self.DEFAULT_ADAPTIVITY_KWARGS)
        self.solver_kwargs = kwargs.get("solver_kwargs", None) or dict(self.DEFAULT_SOLVER_KWARGS)
        self.regularization_kwargs = kwargs.get("regularization_kwargs", None) or \
            dict(self.DEFAULT_REGULARIZATION_KWARGS)
        self.time_regressor_kwargs = kwargs.get("time_regressor_kwargs", None) or \
            dict(self.DEFAULT_TIME_REGRESSOR_KWARGS)

        self.exploration_per_batch = int(
            kwargs.get("exploration_per_batch", max(1, self.batch_size // 5))
        )
        if self.exploration_per_batch >= self.batch_size:
            raise ValueError(
                "exploration_per_batch must be smaller than batch_size "
                f"(got {self.exploration_per_batch} >= {self.batch_size})."
            )
        self.boundary_pool_fraction = float(kwargs.get("boundary_pool_fraction", 0.10))
        self.cost_lambda = float(kwargs.get("cost_lambda", 0.5))
        max_predicted_time = kwargs.get("max_predicted_time", None)
        self.max_predicted_time = float(max_predicted_time) if max_predicted_time is not None else None

        test_set_accuracy_target = kwargs.get("test_set_accuracy_target", None)
        self.test_set_accuracy_target = float(test_set_accuracy_target) \
            if test_set_accuracy_target is not None else None
        if self.test_set_accuracy_target is not None and self._test_X is None:
            raise ValueError(
                "test_set_accuracy_target requires test_data_csv to also be set."
            )
        self._stop_early = False

        cpuh_budget = kwargs.get("cpuh_budget", None)
        self.cpuh_budget = float(cpuh_budget) if cpuh_budget is not None else None
        self.gene_cores_per_run = kwargs.get("gene_cores_per_run", None)
        self.helena_cores_per_run = kwargs.get("helena_cores_per_run", None)
        self.gene_runtime_variable = kwargs.get("gene_runtime_variable", "runtime_sec_gene")
        self.helena_runtime_variable = kwargs.get("helena_runtime_variable", "runtime_sec_helena")
        if self.cpuh_budget is not None and (
            self.gene_cores_per_run is None or self.helena_cores_per_run is None
        ):
            raise ValueError(
                "cpuh_budget requires both gene_cores_per_run and helena_cores_per_run "
                "to also be set."
            )
        self.cpuh_used = 0.0

        # {class_label: (LearnerSGDE, prior_probability)} for classes with
        # enough points to fit; see _fit_model / min_points_per_class.
        self.class_models = {}
        self.classes_ = None
        self.time_model = None

    # ------------------------------------------------------------
    # MODEL FITTING / PREDICTION
    # ------------------------------------------------------------
    def _has_multiple_classes(self):
        """
        True once at least two distinct classes have enough points to fit
        an SGDE density each (see min_points_per_class). Mirrors the other
        time-aware samplers' _has_multiple_classes, but additionally
        requires the per-class point-count minimum SGDE needs to produce a
        non-degenerate density.
        """
        counts = pd.Series(self.train_y_class).value_counts()
        return (counts >= self.min_points_per_class).sum() >= 2

    def _fit_model(self):
        self.time_model = ExtraTreesRegressor(**self.time_regressor_kwargs)
        self.time_model.fit(self.train_x, self.train_y_time)

        if not self._has_multiple_classes():
            self.class_models = {}
            self.classes_ = None
            log.warning(
                "Fewer than 2 classes have >= min_points_per_class=%d training "
                "point(s) so far; skipping SGDE fit and drawing a fully random "
                "batch instead until a second class qualifies.",
                self.min_points_per_class,
            )
            return

        counts = pd.Series(self.train_y_class).value_counts()
        n_total = len(self.train_y_class)
        eligible_classes = counts[counts >= self.min_points_per_class].index.tolist()

        skipped = set(counts.index) - set(eligible_classes)
        if skipped:
            log.warning(
                "Class(es) %s have fewer than min_points_per_class=%d training "
                "points; excluding them from the SGDE posterior until they have "
                "enough data.",
                sorted(skipped), self.min_points_per_class,
            )

        self.class_models = {}
        for label in eligible_classes:
            X_class = self.train_x[self.train_y_class == label]
            learner = _make_learner_sgde(
                self.input_dim, self.grid_kwargs, self.adaptivity_kwargs,
                self.solver_kwargs, self.regularization_kwargs,
            )
            learner.initialize(pysgpp.DataMatrix(X_class))
            learner.train()
            prior = len(X_class) / n_total
            self.class_models[label] = (learner, prior)

        self.classes_ = sorted(self.class_models.keys())

    def _posterior(self, X_unit):
        """
        Normalized posterior probability matrix (N, n_eligible_classes) in
        self.classes_ order: posterior_c(x) = prior_c * p_c(x) / Z(x), with
        Z(x) the sum over eligible classes (renormalizing away the excluded,
        too-few-points classes rather than assigning them 0 probability
        mass that skews the remaining classes' posteriors).

        If every eligible class's density evaluates to (clipped) zero at a
        given point -- e.g. a point far outside all classes' training
        regions -- Z(x) is zero there; the posterior for such rows falls
        back to the prior distribution rather than dividing by zero, so
        those points still get *a* defined (maximally ambiguous-looking)
        posterior instead of NaN propagating into ranking.
        """
        densities = np.stack(
            [_sgde_pdf(learner, X_unit) for learner, _ in self.class_models.values()],
            axis=1,
        )  # (N, C)
        priors = np.array([prior for _, prior in self.class_models.values()])  # (C,)

        weighted = densities * priors[None, :]
        Z = weighted.sum(axis=1)

        posterior = np.empty_like(weighted)
        nonzero = Z > 0
        posterior[nonzero] = weighted[nonzero] / Z[nonzero, None]
        if not nonzero.all():
            posterior[~nonzero] = priors / priors.sum()

        return posterior

    def _predict_labels(self, X_unit):
        posterior = self._posterior(X_unit)
        best = np.argmax(posterior, axis=1)
        classes = np.array(self.classes_, dtype=object)
        return classes[best]

    def _predict_time(self, X_unit):
        return self.time_model.predict(X_unit)

    def _uncertainty(self, X_unit):
        """
        Posterior-uncertainty score for X_unit, per self.acquisition_mode.
        Larger = more uncertain in both modes (entropy is "larger = more
        uncertain" natively; margin is negated so the same convention holds
        across modes, matching the ExtraTrees sampler's _uncertainty).
        """
        posterior = self._posterior(X_unit)
        if self.acquisition_mode == "margin":
            return -_margin_from_posterior(posterior)
        if self.acquisition_mode == "entropy":
            return _entropy_from_posterior(posterior)
        raise ValueError(f"Unknown acquisition_mode: {self.acquisition_mode!r}")

    def _compute_acquisition_unchunked(self, X_unit):
        # Not used directly (see get_next_samples' explicit explore/boundary
        # split), but kept for interface parity / potential reuse by
        # ParentActiveSampler's generic batch-selection helpers.
        return self._uncertainty(X_unit)

    # ------------------------------------------------------------
    # EVALUATION
    # ------------------------------------------------------------
    def evaluate_model(self, do_write_batch_info=False, do_plot_residuals=False):
        """
        Evaluates against the held-out test set (if test_data_csv was
        configured) or, failing that, stratified K-fold CV on the training
        set. Also enforces test_set_accuracy_target: if the test-set
        accuracy drops below it, self._stop_early is set so get_next_samples
        ends the run on its next check.
        """
        if self._test_X is not None:
            metrics = self.compute_testset_metrics()
        else:
            metrics = self.compute_kfold_metrics()

        if metrics is None:
            return None

        if do_write_batch_info:
            self.write_batch_info(metrics)

        if self.test_set_accuracy_target is not None and self._test_X is not None:
            if metrics["accuracy"] < self.test_set_accuracy_target:
                log.warning(
                    "Test-set accuracy %.3f fell below test_set_accuracy_target=%.3f; "
                    "stopping the run.",
                    metrics["accuracy"], self.test_set_accuracy_target,
                )
                self._stop_early = True

        return metrics

    @staticmethod
    def _per_class_metrics(y_true, y_pred, labels):
        """
        Flat {class}_precision / {class}_recall / {class}_f1 columns for each
        label, so a rare class being learned poorly is visible in
        batch_info.csv even when it's masked by a healthy macro F1.
        """
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average=None, zero_division=0
        )
        out = {}
        for label, p, r, f in zip(labels, precision, recall, f1):
            out[f"{label}_precision"] = float(p)
            out[f"{label}_recall"] = float(r)
            out[f"{label}_f1"] = float(f)
        return out

    def _fit_class_models_on(self, X, y_class):
        """
        Fits one LearnerSGDE per eligible class on an arbitrary (X, y_class)
        subset, returning {label: (learner, prior)} -- the K-fold-CV
        counterpart of _fit_model's class_models construction, kept separate
        so K-fold CV never touches self.class_models (the "official" model
        used for boundary sampling).
        """
        counts = pd.Series(y_class).value_counts()
        eligible_classes = counts[counts >= self.min_points_per_class].index.tolist()
        n_total = len(y_class)

        models = {}
        for label in eligible_classes:
            X_class = X[y_class == label]
            learner = _make_learner_sgde(
                self.input_dim, self.grid_kwargs, self.adaptivity_kwargs,
                self.solver_kwargs, self.regularization_kwargs,
            )
            learner.initialize(pysgpp.DataMatrix(X_class))
            learner.train()
            models[label] = (learner, len(X_class) / n_total)

        return models

    @staticmethod
    def _predict_labels_from_models(models, X_unit):
        if len(models) < 2:
            return None
        classes = sorted(models.keys())
        densities = np.stack([_sgde_pdf(models[c][0], X_unit) for c in classes], axis=1)
        priors = np.array([models[c][1] for c in classes])
        weighted = densities * priors[None, :]
        Z = weighted.sum(axis=1)
        posterior = np.empty_like(weighted)
        nonzero = Z > 0
        posterior[nonzero] = weighted[nonzero] / Z[nonzero, None]
        if not nonzero.all():
            posterior[~nonzero] = priors / priors.sum()
        best = np.argmax(posterior, axis=1)
        return np.array(classes, dtype=object)[best]

    def compute_kfold_metrics(self):
        X = self.train_x
        y_class = self.train_y_class
        y_time = self.train_y_time

        if not self._has_multiple_classes():
            return None

        class_counts = pd.Series(y_class).value_counts()
        if len(y_class) < self.num_folds or class_counts.min() < self.num_folds:
            return None

        skf = StratifiedKFold(n_splits=self.num_folds, shuffle=True, random_state=self.seed)
        all_y_true, all_y_pred = [], []
        for train_idx, val_idx in skf.split(X, y_class):
            models = self._fit_class_models_on(X[train_idx], y_class[train_idx])
            y_pred = self._predict_labels_from_models(models, X[val_idx])
            if y_pred is None:
                continue
            all_y_true.append(y_class[val_idx])
            all_y_pred.append(y_pred)

        if not all_y_true:
            return None

        y_true_all = np.concatenate(all_y_true)
        y_pred_all = np.concatenate(all_y_pred)
        labels = sorted(pd.unique(np.concatenate([y_true_all, y_pred_all])))

        kf = KFold(n_splits=self.num_folds, shuffle=True, random_state=self.seed)
        time_rmses = []
        for train_idx, val_idx in kf.split(X):
            reg = ExtraTreesRegressor(**self.time_regressor_kwargs)
            reg.fit(X[train_idx], y_time[train_idx])
            y_pred_time = reg.predict(X[val_idx])
            time_rmses.append(np.sqrt(np.mean((y_pred_time - y_time[val_idx]) ** 2)))

        return {
            "eval_mode": "kfold",
            "accuracy": float(accuracy_score(y_true_all, y_pred_all)),
            "f1_macro": float(f1_score(y_true_all, y_pred_all, average="macro", zero_division=0)),
            "time_rmse": float(np.mean(time_rmses)),
            **self._per_class_metrics(y_true_all, y_pred_all, labels),
            "labels": labels,
            "confusion_matrix": confusion_matrix(y_true_all, y_pred_all, labels=labels),
        }

    def compute_testset_metrics(self):
        """
        Fits on all current training data and evaluates against the held-out
        test set (self._test_X / self._test_y_class / self._test_y_time).
        """
        if len(self.train_y_class) == 0:
            return None

        self._fit_model()

        if not self.class_models or len(self.class_models) < 2:
            # Not enough class diversity yet to evaluate classification
            # performance (see _fit_model); skip this evaluation cycle.
            return None

        y_pred_class = self._predict_labels(self._test_X)
        y_pred_time = self.time_model.predict(self._test_X)

        accuracy = accuracy_score(self._test_y_class, y_pred_class)
        f1_macro = f1_score(self._test_y_class, y_pred_class, average="macro", zero_division=0)
        time_rmse = np.sqrt(np.mean((y_pred_time - self._test_y_time) ** 2))
        labels = sorted(pd.unique(np.concatenate([self._test_y_class, y_pred_class])))

        return {
            "eval_mode": "test_set",
            "accuracy": float(accuracy),
            "f1_macro": float(f1_macro),
            "time_rmse": float(time_rmse),
            **self._per_class_metrics(self._test_y_class, y_pred_class, labels),
            "labels": labels,
            "confusion_matrix": confusion_matrix(self._test_y_class, y_pred_class, labels=labels),
        }

    def write_batch_info(self, metrics):
        """
        Appends one row to batch_info.csv. Per-class columns (e.g.
        MTM_precision) only exist once that class has been observed, so
        later rows can introduce columns earlier rows never had. A plain
        mode="a", header=False append would silently misalign those new
        columns under the stale header instead of adding them, so when the
        new row's columns aren't a subset of the existing header, the whole
        file is rewritten with the union of old + new columns (missing
        values become blank/NaN for rows that predate a class).
        """
        scalar_metrics = {k: v for k, v in metrics.items() if k not in ("labels", "confusion_matrix")}
        row = {"num_train_samples": self.train_x.shape[0], **scalar_metrics}
        df = pd.DataFrame([row])

        csv_path = os.path.join(self.base_run_dir, 'batch_info.csv')
        if not os.path.exists(csv_path):
            df.to_csv(csv_path, index=False)
            return

        existing = pd.read_csv(csv_path)
        if set(df.columns).issubset(existing.columns):
            df = df.reindex(columns=existing.columns)
            df.to_csv(csv_path, mode="a", header=False, index=False)
        else:
            combined = pd.concat([existing, df], ignore_index=True, sort=False)
            combined.to_csv(csv_path, index=False)

    # ------------------------------------------------------------
    # TRAINING SET BOOKKEEPING
    # ------------------------------------------------------------
    def register_future(self, future_df):
        if self.cpuh_budget is not None:
            self._accumulate_cpuh(future_df)

        future_df = future_df[future_df['success']]
        future_df = self._apply_row_filters(future_df)

        if self.unclassified_label is not None:
            n_before = len(future_df)
            future_df = future_df[future_df[self.class_output_variable] != self.unclassified_label]
            n_dropped = n_before - len(future_df)
            if n_dropped:
                log.debug(
                    f"Dropped {n_dropped} row(s) labeled {self.unclassified_label!r} "
                    "(carries no class information; not added to training set)."
                )

        if future_df.empty:
            return

        X_real = future_df[self.parameters].to_numpy(dtype=float)
        y_class = future_df[self.class_output_variable].to_numpy()
        y_time = future_df[self.time_output_variable].to_numpy(dtype=float)

        X_unit = self.to_unit_numpy(X_real)

        self.train_x = np.vstack([self.train_x, X_unit])
        self.train_y_class = np.concatenate([self.train_y_class, y_class])
        self.train_y_time = np.concatenate([self.train_y_time, y_time])

        log.debug(
            f"future_df rows: {len(future_df)}\n"
            f"train_x new shape: {self.train_x.shape}"
        )

    def _accumulate_cpuh(self, future_df):
        """
        Adds this batch's CPU-hour cost to self.cpuh_used, from every row
        submitted (successful or not -- a failed run still burns CPU-hours),
        using gene_runtime_variable/helena_runtime_variable * the configured
        per-run core counts. Missing/NaN runtime values contribute 0 (rather
        than raising), so a run missing one stage's timing column doesn't
        block CPU-hour tracking for the rest.
        """
        gene_hours = 0.0
        if self.gene_runtime_variable in future_df.columns:
            gene_sec = future_df[self.gene_runtime_variable].fillna(0).to_numpy(dtype=float)
            gene_hours = float(gene_sec.sum()) / 3600.0 * self.gene_cores_per_run

        helena_hours = 0.0
        if self.helena_runtime_variable in future_df.columns:
            helena_sec = future_df[self.helena_runtime_variable].fillna(0).to_numpy(dtype=float)
            helena_hours = float(helena_sec.sum()) / 3600.0 * self.helena_cores_per_run

        batch_cpuh = gene_hours + helena_hours
        self.cpuh_used += batch_cpuh
        log.debug(
            f"Batch CPU-hours: {batch_cpuh:.2f} (gene={gene_hours:.2f}, "
            f"helena={helena_hours:.2f}); running total: {self.cpuh_used:.2f}"
        )

    # ------------------------------------------------------------
    # MAIN ENTRY: GET NEXT SAMPLES
    # ------------------------------------------------------------
    def get_next_samples(self):
        if self.batch_number == 0:
            initial_pool_indices = self._get_initial_batch()
            real_selected_samples = self._get_samples_from_pool(initial_pool_indices)
            self._remove_from_pool(initial_pool_indices)
        else:
            self._fit_model()
            self.evaluate_model(
                do_write_batch_info=self._should_trigger(self.write_batch_info_every),
            )

            if self._stop_early:
                log.warning(
                    "Stopping early: test-set accuracy fell below "
                    "test_set_accuracy_target (submitted %d/%d of budget).",
                    self.submitted, self.budget,
                )
                self._light_post_process()
                return None

            if self.cpuh_budget is not None and self.cpuh_used >= self.cpuh_budget:
                log.warning(
                    "Stopping: CPU-hour budget reached (%.2f/%.2f CPU-hours used, "
                    "submitted %d/%d of sample budget).",
                    self.cpuh_used, self.cpuh_budget, self.submitted, self.budget,
                )
                self._light_post_process()
                return None

            if not self.class_models or len(self.class_models) < 2:
                # Not enough class diversity yet to fit a posterior boundary
                # (see _fit_model): draw a fully random batch instead, same
                # as the initial batch, until a second class qualifies.
                selected_indices = self._get_initial_batch_n(self.batch_size)
            else:
                explore_indices = self._get_initial_batch_n(
                    self.exploration_per_batch, filter_by_time=True
                )
                boundary_indices = self._compute_boundary_candidates(
                    self.batch_size - self.exploration_per_batch, exclude=explore_indices
                )
                selected_indices = np.concatenate([explore_indices, boundary_indices]).astype(int)

            real_selected_samples = self._get_samples_from_pool(selected_indices)
            self._remove_from_pool(selected_indices)

        self.batch_number += 1
        self.submitted += len(real_selected_samples)
        params_dict = self.samples_to_params_dict(real_selected_samples)

        # Always return this batch, even if submitting it exhausts (or
        # crosses) the budget -- the caller's own has_budget/submitted>budget
        # checks stop the loop on the *next* call. Returning None here
        # instead would silently discard the batch just selected and
        # computed, wasting it (see supervisor.py's `if samples is None:
        # break` -- it never runs the batch in that case).
        if not self.has_budget:
            self._light_post_process()

        return params_dict

    def samples_to_params_dict(self, samples):
        """
        Same as ParentActiveSampler.samples_to_params_dict, but tags each
        sample with batch_num (self.batch_number at submission time). This
        column is not a real simulation parameter and is safely ignored by
        the GENE/HELENA parsers -- it just flows through to
        enchanted_dataset.csv, which is enough to fully reconstruct any
        checkpoint later: refit the sampler's model on the subset of rows
        with batch_num <= N using the same sampler config.
        """
        params_dict = super().samples_to_params_dict(samples)
        for p in params_dict:
            p["batch_num"] = self.batch_number
        return params_dict

    def _get_initial_batch_n(self, n, filter_by_time=False):
        """
        Unrestricted random draw of size n. If filter_by_time and
        max_predicted_time are set (and a time model has been fit, i.e. this
        isn't the very first batch), points predicted to exceed
        max_predicted_time are excluded before the random draw; if too few
        qualify, a warning is logged and the draw falls back to the n
        cheapest-predicted points in the pool regardless of the limit.
        """
        use_time_filter = filter_by_time and self.max_predicted_time is not None and \
            self.time_model is not None

        cand_scores = np.array([], float)
        cand_indices = np.array([], int)
        under_limit_indices = np.array([], int)

        cheapest_times = np.array([], float)
        cheapest_indices = np.array([], int)

        self._reset_iterator()
        while True:
            X_chunk_unit, _, chunk_indices = self.get_next_pool_chunk()
            if X_chunk_unit is None:
                break
            chunk_indices = np.asarray(chunk_indices)

            if use_time_filter:
                pred_time = self._predict_time(X_chunk_unit)
                under_limit_indices = np.concatenate(
                    [under_limit_indices, chunk_indices[pred_time <= self.max_predicted_time]]
                )
                # Keep a running set of the cheapest-seen points as a fallback
                # in case not enough points end up under the limit.
                cheapest_times = np.concatenate([cheapest_times, pred_time])
                cheapest_indices = np.concatenate([cheapest_indices, chunk_indices])
                if len(cheapest_times) > n:
                    keep = np.argsort(cheapest_times)[:n]
                    cheapest_times = cheapest_times[keep]
                    cheapest_indices = cheapest_indices[keep]

            scores = self.rng.random(len(chunk_indices))
            combined_scores = np.concatenate([cand_scores, scores])
            combined_indices = np.concatenate([cand_indices, chunk_indices])
            if len(combined_scores) > n:
                top_m = np.argpartition(combined_scores, -n)[-n:]
                cand_scores = combined_scores[top_m]
                cand_indices = combined_indices[top_m]
            else:
                cand_scores = combined_scores
                cand_indices = combined_indices

        if not use_time_filter:
            return cand_indices.astype(int)

        if len(under_limit_indices) < n:
            log.warning(
                "Only %d/%d pool points are predicted under max_predicted_time=%s; "
                "falling back to the %d cheapest-predicted points for the exploration sub-batch.",
                len(under_limit_indices), n, self.max_predicted_time, n,
            )
            return cheapest_indices.astype(int)

        chosen = self.rng.choice(under_limit_indices, size=n, replace=False)
        return chosen.astype(int)

    def _compute_boundary_candidates(self, n, exclude=()):
        """
        Streams the remaining pool, restricts to the convex hull of the
        points sampled so far, drops any point whose predicted time exceeds
        max_predicted_time (if set), and ranks the survivors by a weighted
        combination of posterior uncertainty (see self.acquisition_mode /
        _uncertainty) and predicted simulation cost:

            score = uncertainty_rank - cost_lambda * predicted_time_rank

        with both ranks normalized to [0, 1] over the candidate pool (1 =
        most uncertain / most expensive), so cost_lambda directly trades off
        informativeness against cost regardless of each quantity's raw
        scale. The boundary sub-batch is then drawn uniformly at random from
        the top boundary_pool_fraction slice by this combined score.

        If the time limit leaves fewer than n hull-restricted candidates, a
        warning is logged and the limit is relaxed (falling back to the
        cheapest-predicted hull-restricted candidates) so the boundary
        sub-batch is still filled.
        """
        hull_points = self.train_x
        exclude = set(int(i) for i in exclude)

        all_uncertainty = []
        all_times = []
        all_indices = []

        self._reset_iterator()
        while True:
            X_chunk_unit, _, chunk_indices = self.get_next_pool_chunk()
            if X_chunk_unit is None:
                break

            chunk_indices = np.asarray(chunk_indices)
            keep = np.array([idx not in exclude for idx in chunk_indices])
            if not keep.any():
                continue
            X_chunk_unit = X_chunk_unit[keep]
            chunk_indices = chunk_indices[keep]

            inside = _in_hull(X_chunk_unit, hull_points)
            if inside.sum() > 0:
                u = self._uncertainty(X_chunk_unit[inside])
                t = self._predict_time(X_chunk_unit[inside])
                all_uncertainty.append(u)
                all_times.append(t)
                all_indices.append(chunk_indices[inside])

        if not all_indices or sum(len(idx) for idx in all_indices) < n:
            log.warning(
                "Hull-restricted pool has fewer than %d classifiable candidates; "
                "falling back to an unrestricted random draw for the boundary sub-batch.",
                n,
            )
            fallback = self._get_initial_batch_n(n + len(exclude), filter_by_time=True)
            fallback = fallback[~np.isin(fallback, list(exclude))]
            return fallback[:n]

        uncertainty = np.concatenate(all_uncertainty)
        times = np.concatenate(all_times)
        indices = np.concatenate(all_indices)

        if self.max_predicted_time is not None:
            under_limit = times <= self.max_predicted_time
            if under_limit.sum() >= n:
                uncertainty = uncertainty[under_limit]
                times = times[under_limit]
                indices = indices[under_limit]
            else:
                log.warning(
                    "Only %d/%d hull-restricted candidates are predicted under "
                    "max_predicted_time=%s; relaxing the limit and using the "
                    "cheapest-predicted hull-restricted candidates instead.",
                    under_limit.sum(), n, self.max_predicted_time,
                )
                pool_size = max(n, int(len(times) * self.boundary_pool_fraction))
                cheapest = np.argsort(times)[:min(pool_size, len(times))]
                uncertainty = uncertainty[cheapest]
                times = times[cheapest]
                indices = indices[cheapest]

        # Larger uncertainty = more uncertain -> rank 1 is most uncertain
        # (see _uncertainty's docstring for the shared sign convention).
        uncertainty_rank = pd.Series(uncertainty).rank(pct=True).to_numpy()
        time_rank = pd.Series(times).rank(pct=True).to_numpy()
        combined_score = uncertainty_rank - self.cost_lambda * time_rank

        pool_size = max(n, int(len(indices) * self.boundary_pool_fraction))
        pool_size = min(pool_size, len(indices))
        boundary_pool = indices[np.argsort(combined_score)[-pool_size:]]

        chosen = self.rng.choice(boundary_pool, size=n, replace=False)
        return chosen.astype(int)
