"""
SvmTimeAwareActiveSampler + per-cycle hyperparameter re-tuning.

SvmTimeAwareActiveSampler (and every other sampler in this package) fits its
model with fixed, caller-supplied hyperparameters every cycle -- there is no
built-in re-tuning hook (the only precedent, ExtraTreesActiveSampler's
tune_n_estimators, is a standalone opt-in method nothing calls
automatically). This subclass adds one: before each _fit_model() call, it
re-selects (C, gamma) via a fast holdout search (an 80/20 random split of the
*current* training set, not K-fold CV -- deliberately cheap since this runs
every single acquisition cycle) over a coarse grid, then fits with the
winning pair. All other SVC kwargs (kernel, class_weight) stay at whatever
svc_kwargs/DEFAULT_SVC_KWARGS already provides.

Ported from the hand-rolled study's tune_svm()/_holdout_split() (see
RT-01_JET_data/ES_data/256_sobol_hel_60_grid_gene_97481_ped_width_ne/
svm_et_3class_acquisition_study/run_study.py) -- same holdout-fraction, same
grid, same "too few rows -> skip tuning, use last/default" fallback -- just
wired into this sampler's _fit_model() instead of a hand-rolled loop.

Also overrides _compute_boundary_candidates to swap the base class's exact
convex-hull filter (scipy Delaunay triangulation, via _build_hull/_in_hull)
for the approximate PCA-whitened-ellipsoid one in approx_hull.py. Confirmed
via the mem_issue_debug branch's RSS/timing instrumentation that exact
Delaunay is combinatorially infeasible on this study's 10-D parameter space:
725MB/31s at 60 training points, 1.4GB/125s at 80, both climbing every cycle
purely from the triangulation itself getting more expensive as train_x
grows -- not a bug fixable by computing the hull less often (already tried;
see _ChunkBuffer's docstring), a fundamental limit of exact-hull computation
past ~6-8 dimensions. See approx_hull.py's module docstring for the method.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from enchanted_surrogates.samplers.svm_time_aware_active_sampler import (
    SvmTimeAwareActiveSampler,
    _entropy_from_probs,
)
from enchanted_surrogates.samplers.svm_active_sampler import _margin, _build_hull, _in_hull
from enchanted_surrogates.samplers.approx_hull import build_approx_hull, approx_in_hull
from enchanted_surrogates.utils.logger import get_logger

log = get_logger(__name__)

VALID_HULL_METHODS = ("approx", "exact")


class SvmTimeAwareTunedActiveSampler(SvmTimeAwareActiveSampler):
    """
    Configuration (in addition to SvmTimeAwareActiveSampler's):
        tune_C_grid : list of float, optional
            Candidate C values swept every cycle. Default: [1, 10, 100].
        tune_gamma_grid : list, optional
            Candidate gamma values swept every cycle (mix of "scale" and
            floats is fine). Default: ["scale", 0.01, 0.1].
        tune_val_fraction : float, optional
            Fraction of the current training set held out for tuning.
            Default: 0.2.
        tune_min_val_rows : int, optional
            Minimum holdout rows required to attempt tuning; below this,
            tuning is skipped and the previous (C, gamma) is kept. Default: 4.
        hull_method : {"approx", "exact"}, optional
            "approx" (default): PCA-whitened ellipsoid boundary test
            (approx_hull.py) -- O(train points x dim^2) to fit, O(1) per
            candidate, stays cheap regardless of dimension or training-set
            size. "exact": the base class's scipy Delaunay triangulation
            (_build_hull/_in_hull) -- exact convex hull, but combinatorially
            expensive past ~6-8 input dimensions and confirmed to blow up in
            both memory and time well before 100 training points on this
            study's 10-D parameter space (see this module's docstring).
            Default is "approx" because that's what actually completes on
            this study; "exact" is offered for lower-dimensional studies or
            direct comparison against the base class's behavior.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hull_method = kwargs.get("hull_method", "approx")
        if self.hull_method not in VALID_HULL_METHODS:
            raise ValueError(
                f"hull_method must be one of {VALID_HULL_METHODS}, got {self.hull_method!r}."
            )
        self.tune_C_grid = kwargs.get("tune_C_grid", [1, 10, 100])
        self.tune_gamma_grid = kwargs.get("tune_gamma_grid", ["scale", 0.01, 0.1])
        self.tune_val_fraction = float(kwargs.get("tune_val_fraction", 0.2))
        self.tune_min_val_rows = int(kwargs.get("tune_min_val_rows", 4))
        self.tune_rng = np.random.default_rng(
            (self.seed if self.seed is not None else 0) + 10_000
        )
        # Seeded from svc_kwargs so tuning is a no-op (falls straight through
        # to the configured/default C,gamma) until the first successful tune.
        self.selected_C = self.svc_kwargs.get("C", self.DEFAULT_SVC_KWARGS["C"])
        self.selected_gamma = self.svc_kwargs.get("gamma", self.DEFAULT_SVC_KWARGS["gamma"])

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

        scaler = StandardScaler().fit(self.train_x[fit_idx])
        Xf = scaler.transform(self.train_x[fit_idx])
        Xv = scaler.transform(self.train_x[val_idx])
        yf, yv = self.train_y_class[fit_idx], self.train_y_class[val_idx]

        best_score, best_params = -1.0, (self.selected_C, self.selected_gamma)
        for C in self.tune_C_grid:
            for gamma in self.tune_gamma_grid:
                kwargs = dict(self.svc_kwargs, C=C, gamma=gamma)
                if self.acquisition_mode == "entropy":
                    kwargs["probability"] = True
                try:
                    clf = SVC(**kwargs)
                    clf.fit(Xf, yf)
                    score = f1_score(yv, clf.predict(Xv), average="macro", zero_division=0)
                except ValueError:
                    continue
                if score > best_score:
                    best_score, best_params = score, (C, gamma)

        self.selected_C, self.selected_gamma = best_params
        log.debug(
            "Tuned SVM: selected C=%s gamma=%s (holdout macro-F1=%.3f, n_train=%d)",
            self.selected_C, self.selected_gamma, best_score, n,
        )

    def _fit_model(self):
        self._tune()

        self.scaler = StandardScaler().fit(self.train_x)

        from sklearn.ensemble import ExtraTreesRegressor
        self.time_model = ExtraTreesRegressor(**self.time_regressor_kwargs)
        self.time_model.fit(self.train_x, self.train_y_time)

        if not self._has_multiple_classes():
            self.svm_model = None
            log.warning(
                "Only one class (%r) seen in %d training point(s) so far; "
                "skipping SVM fit and drawing a fully random batch instead "
                "until a second class is observed.",
                self.train_y_class[0] if len(self.train_y_class) else None,
                len(self.train_y_class),
            )
            return

        X_scaled = self.scaler.transform(self.train_x)
        svc_kwargs = dict(self.svc_kwargs, C=self.selected_C, gamma=self.selected_gamma)
        if self.acquisition_mode == "entropy" and "probability" not in svc_kwargs:
            svc_kwargs["probability"] = True
        self.svm_model = SVC(**svc_kwargs)
        self.svm_model.fit(X_scaled, self.train_y_class)

    def evaluate_model(self, do_write_batch_info=False, do_plot_residuals=False):
        metrics = super().evaluate_model(
            do_write_batch_info=False, do_plot_residuals=do_plot_residuals
        )
        if metrics is None:
            return None
        metrics = dict(metrics)
        metrics["selected_C"] = self.selected_C
        metrics["selected_gamma"] = self.selected_gamma
        if do_write_batch_info:
            self.write_batch_info(metrics)
        return metrics

    def samples_to_params_dict(self, samples):
        """
        Drops the batch_num tag that SvmTimeAwareActiveSampler.
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
        Identical to SvmTimeAwareActiveSampler._compute_boundary_candidates
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

        hull_points = self.scaler.transform(self.train_x)
        hull = build_hull_fn(hull_points)
        exclude = set(int(i) for i in exclude)

        all_uncertainty = []
        all_times = []
        all_indices = []

        candidate_buffer = self._ChunkBuffer(self.pool_chunk_size)

        def _score_and_absorb(X_batch, idx_batch):
            all_uncertainty.append(self._uncertainty(X_batch))
            all_times.append(self._predict_time(X_batch))
            all_indices.append(idx_batch)

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

            X_chunk_scaled = self.scaler.transform(X_chunk_unit)
            inside = in_hull_fn(X_chunk_scaled, hull)
            if inside.sum() == 0:
                continue

            candidate_buffer.add(X_chunk_unit[inside], chunk_indices[inside])
            for X_batch, idx_batch in candidate_buffer.ready_batches():
                _score_and_absorb(X_batch, idx_batch)

        for X_batch, idx_batch in candidate_buffer.flush():
            _score_and_absorb(X_batch, idx_batch)

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
