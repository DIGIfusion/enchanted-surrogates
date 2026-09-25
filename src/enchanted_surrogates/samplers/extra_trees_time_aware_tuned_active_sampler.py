"""
ExtraTreesTimeAwareActiveSampler + per-cycle hyperparameter re-tuning.

Same rationale as svm_time_aware_tuned_active_sampler.py: no sampler in this
package re-tunes its model's hyperparameters automatically, so this subclass
adds a fast holdout search (80/20 random split of the current training set)
over the two hyperparameters that matter most for ExtraTrees overfitting
control on small active-learning training sets -- max_depth and
max_features -- re-run every cycle before _fit_model() constructs the
classifier. n_estimators stays fixed (compute/variance tradeoff, not a major
over/underfitting lever) per classifier_kwargs/DEFAULT_CLASSIFIER_KWARGS.

Ported from the hand-rolled study's tune_et() (see
RT-01_JET_data/ES_data/256_sobol_hel_60_grid_gene_97481_ped_width_ne/
svm_et_3class_acquisition_study/run_study.py).

Also overrides _compute_boundary_candidates to swap the base class's exact
convex-hull filter (scipy Delaunay triangulation, via _build_hull/_in_hull)
for the approximate PCA-whitened-ellipsoid one in approx_hull.py -- see
svm_time_aware_tuned_active_sampler.py's module docstring for the full
rationale (confirmed via the mem_issue_debug branch's RSS/timing logs that
exact Delaunay is combinatorially infeasible on this study's 10-D parameter
space, getting more expensive every cycle purely from train_x growing).
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.metrics import f1_score

from enchanted_surrogates.samplers.extra_trees_time_aware_active_sampler import (
    ExtraTreesTimeAwareActiveSampler,
)
from enchanted_surrogates.samplers.svm_active_sampler import _build_hull, _in_hull
from enchanted_surrogates.samplers.approx_hull import build_approx_hull, approx_in_hull
from enchanted_surrogates.utils.logger import get_logger

log = get_logger(__name__)

VALID_HULL_METHODS = ("approx", "exact")


class ExtraTreesTimeAwareTunedActiveSampler(ExtraTreesTimeAwareActiveSampler):
    """
    Configuration (in addition to ExtraTreesTimeAwareActiveSampler's):
        tune_max_depth_grid : list, optional
            Candidate max_depth values (None allowed) swept every cycle.
            Default: [None, 10, 20].
        tune_max_features_grid : list, optional
            Candidate max_features values (None allowed) swept every cycle.
            Default: ["sqrt", "log2", None].
        tune_val_fraction : float, optional
            Fraction of the current training set held out for tuning.
            Default: 0.2.
        tune_min_val_rows : int, optional
            Minimum holdout rows required to attempt tuning; below this,
            tuning is skipped and the previous (max_depth, max_features) is
            kept. Default: 4.
        hull_method : {"approx", "exact"}, optional
            "approx" (default): PCA-whitened ellipsoid boundary test
            (approx_hull.py) -- O(train points x dim^2) to fit, O(1) per
            candidate, stays cheap regardless of dimension or training-set
            size. "exact": the base class's scipy Delaunay triangulation
            (_build_hull/_in_hull) -- exact convex hull, but combinatorially
            expensive past ~6-8 input dimensions; confirmed to blow up in
            both memory and time well before 100 training points on this
            study's 10-D parameter space. Default is "approx" because that's
            what actually completes on this study; "exact" is offered for
            lower-dimensional studies or direct comparison.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hull_method = kwargs.get("hull_method", "approx")
        if self.hull_method not in VALID_HULL_METHODS:
            raise ValueError(
                f"hull_method must be one of {VALID_HULL_METHODS}, got {self.hull_method!r}."
            )
        self.tune_max_depth_grid = kwargs.get("tune_max_depth_grid", [None, 10, 20])
        self.tune_max_features_grid = kwargs.get("tune_max_features_grid", ["sqrt", "log2", None])
        self.tune_val_fraction = float(kwargs.get("tune_val_fraction", 0.2))
        self.tune_min_val_rows = int(kwargs.get("tune_min_val_rows", 4))
        self.tune_rng = np.random.default_rng(
            (self.seed if self.seed is not None else 0) + 10_000
        )
        self.selected_max_depth = self.classifier_kwargs.get("max_depth", None)
        self.selected_max_features = self.classifier_kwargs.get("max_features", "sqrt")

    def _holdout_split(self, n):
        n_val = max(self.tune_min_val_rows, int(round(self.tune_val_fraction * n)))
        n_val = min(n_val, n - 2)
        if n_val < 1 or n - n_val < 2:
            return None, None
        val_idx = self.tune_rng.choice(n, size=n_val, replace=False)
        mask = np.ones(n, dtype=bool)
        mask[val_idx] = False
        return np.nonzero(mask)[0], val_idx

    def _tune(self):
        n = len(self.train_y_class)
        if len(np.unique(self.train_y_class)) < 2:
            return
        fit_idx, val_idx = self._holdout_split(n)
        if fit_idx is None or len(np.unique(self.train_y_class[fit_idx])) < 2:
            return

        Xf, Xv = self.train_x[fit_idx], self.train_x[val_idx]
        yf, yv = self.train_y_class[fit_idx], self.train_y_class[val_idx]

        best_score = -1.0
        best_params = (self.selected_max_depth, self.selected_max_features)
        for max_depth in self.tune_max_depth_grid:
            for max_features in self.tune_max_features_grid:
                kwargs = dict(self.classifier_kwargs, max_depth=max_depth, max_features=max_features)
                kwargs.setdefault("random_state", 0)
                clf = ExtraTreesClassifier(**kwargs)
                clf.fit(Xf, yf)
                score = f1_score(yv, clf.predict(Xv), average="macro", zero_division=0)
                if score > best_score:
                    best_score, best_params = score, (max_depth, max_features)

        self.selected_max_depth, self.selected_max_features = best_params
        log.debug(
            "Tuned ExtraTrees: selected max_depth=%s max_features=%s "
            "(holdout macro-F1=%.3f, n_train=%d)",
            self.selected_max_depth, self.selected_max_features, best_score, n,
        )

    def _fit_model(self):
        self._tune()

        self.time_model = ExtraTreesRegressor(**self.time_regressor_kwargs)
        self.time_model.fit(self.train_x, self.train_y_time)

        if not self._has_multiple_classes():
            self.classifier_model = None
            log.warning(
                "Only one class (%r) seen in %d training point(s) so far; "
                "skipping classifier fit and drawing a fully random batch instead "
                "until a second class is observed.",
                self.train_y_class[0] if len(self.train_y_class) else None,
                len(self.train_y_class),
            )
            return

        kwargs = dict(
            self.classifier_kwargs,
            max_depth=self.selected_max_depth,
            max_features=self.selected_max_features,
        )
        self.classifier_model = ExtraTreesClassifier(**kwargs)
        self.classifier_model.fit(self.train_x, self.train_y_class)

    def evaluate_model(self, do_write_batch_info=False, do_plot_residuals=False):
        metrics = super().evaluate_model(
            do_write_batch_info=False, do_plot_residuals=do_plot_residuals
        )
        if metrics is None:
            return None
        metrics = dict(metrics)
        metrics["selected_max_depth"] = self.selected_max_depth
        metrics["selected_max_features"] = self.selected_max_features
        if do_write_batch_info:
            self.write_batch_info(metrics)
        return metrics

    def samples_to_params_dict(self, samples):
        """
        Drops the batch_num tag ExtraTreesTimeAwareActiveSampler.
        samples_to_params_dict adds (used for checkpoint reconstruction from
        a real simulation dataset -- not needed here). CsvRunner.
        single_code_run validates every key in params against the pool CSV's
        columns and requires an exact/tolerance match on all of them, unlike
        the GENE/HELENA parsers which silently ignore unrecognized keys; an
        offline CsvRunner-backed study has no simulation run to reconstruct,
        so batch_num would only ever cause spurious lookup failures here.
        """
        params_dict = super().samples_to_params_dict(samples)
        for p in params_dict:
            p.pop("batch_num", None)
        return params_dict

    def _compute_boundary_candidates(self, n, exclude=()):
        """
        Identical to ExtraTreesTimeAwareActiveSampler._compute_boundary_candidates
        (see that method's docstring for the scoring/fallback logic), except
        the hull is built/tested via self.hull_method: "approx"
        (approx_hull.py's PCA-whitened ellipsoid, default) or "exact"
        (svm_active_sampler.py's Delaunay triangulation, the base class's
        original behavior) -- see this module's docstring for the tradeoff.
        """
        if self.hull_method == "exact":
            build_hull_fn, in_hull_fn = _build_hull, _in_hull
        else:
            build_hull_fn, in_hull_fn = build_approx_hull, approx_in_hull

        hull_points = self.train_x
        hull = build_hull_fn(hull_points)
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

            inside = in_hull_fn(X_chunk_unit, hull)
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

        uncertainty_rank = pd.Series(uncertainty).rank(pct=True).to_numpy()
        time_rank = pd.Series(times).rank(pct=True).to_numpy()
        combined_score = uncertainty_rank - self.cost_lambda * time_rank

        pool_size = max(n, int(len(indices) * self.boundary_pool_fraction))
        pool_size = min(pool_size, len(indices))
        boundary_pool = indices[np.argsort(combined_score)[-pool_size:]]

        chosen = self.rng.choice(boundary_pool, size=n, replace=False)
        return chosen.astype(int)
