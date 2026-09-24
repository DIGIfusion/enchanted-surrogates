"""
SVM boundary-uncertainty active sampler.

Ports the active-learning strategy demonstrated in
``active_sampling_boundaries/active_sampling_svm.py`` (RBF-kernel SVM +
convex-hull-restricted margin uncertainty) onto the enchanted-surrogates
classification active-sampler engine (ParentActiveSamplerClassification).

Each batch after the first is split into two fixed-size parts:
  - a small unrestricted random exploration sub-batch, so the sampled
    region's convex hull keeps growing outward every batch instead of being
    locked to the initial random draw;
  - a boundary sub-batch drawn from the most-uncertain (smallest SVM margin)
    points, restricted to the convex hull of all points sampled so far.

The hull restriction matters: outside the sampled region, RBF kernel
similarity to every training point decays toward zero, so *all* classes'
decision_function values shrink toward zero together -- that looks like
"most uncertain" to a naive margin score even though it's just unexplored
extrapolation. Without hull restriction the boundary sub-batch chases the
domain's edges/corners instead of real class boundaries. Without the
exploration sub-batch, the hull can never grow past whatever the initial
random draw happened to span, permanently starving any region it missed of
training data.
"""

import numpy as np
from scipy.spatial import Delaunay, QhullError
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from enchanted_surrogates.samplers.parent_active_sampler_classification import (
    ParentActiveSamplerClassification,
)
from enchanted_surrogates.utils.logger import get_logger

log = get_logger(__name__)


def _build_hull(hull_points):
    """
    Builds the Delaunay triangulation of hull_points once, for reuse across
    many _in_hull calls (e.g. once per pool chunk in a streaming loop).
    Returns None if the hull is degenerate (fewer than 3 non-collinear hull
    points, which can happen early on) -- _in_hull treats None as "keep
    everything".

    Building this is expensive (worse than linear in point count, and
    scipy/Qhull's Delaunay triangulation scales particularly poorly past
    ~6-8 dimensions) and does not depend on the points being tested against
    it, only on hull_points -- so callers that test many batches of points
    against the *same* hull_points (e.g. a chunked pool stream) should build
    it once here and pass the result to every _in_hull call, rather than
    rebuilding it per chunk.
    """
    try:
        return Delaunay(hull_points)
    except QhullError:
        return None


def _in_hull(points, hull):
    """
    Boolean mask of which rows of ``points`` fall inside the convex hull
    represented by ``hull``, a value returned by _build_hull (or None, which
    means "keep everything" -- see _build_hull's docstring).
    """
    if hull is None:
        return np.ones(len(points), dtype=bool)
    return hull.find_simplex(points) >= 0


def _margin(clf, X):
    """
    Per-sample distance-to-decision-boundary proxy: smaller = more uncertain.
    Uses the multiclass OVR decision function's top1-vs-top2 gap, or the raw
    signed OVO distance when only two classes have been seen so far.
    """
    d = clf.decision_function(X)
    if d.ndim == 1:
        return np.abs(d)
    sorted_d = np.sort(d, axis=1)
    return sorted_d[:, -1] - sorted_d[:, -2]


class SvmActiveSampler(ParentActiveSamplerClassification):
    """
    RBF-kernel SVM active sampler using convex-hull-restricted margin
    uncertainty, for classification tasks (e.g. GENE instability regime
    classification).

    Configuration (in addition to ParentActiveSamplerClassification's):
        svc_kwargs : dict, optional
            Keyword arguments forwarded to sklearn.svm.SVC. Defaults to
            ``kernel="rbf", C=10.0, gamma="scale", class_weight="balanced"``.
        exploration_per_batch : int, optional
            Number of unrestricted random exploration points drawn each
            batch (after the initial batch). Default: max(1, batch_size // 5).
        boundary_pool_fraction : float, optional
            Fraction of the (hull-restricted) remaining pool to treat as the
            "most uncertain" candidate pool, from which the boundary
            sub-batch is drawn uniformly at random. Default: 0.10.
    """

    DEFAULT_SVC_KWARGS = dict(kernel="rbf", C=10.0, gamma="scale", class_weight="balanced")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.svc_kwargs = kwargs.get("svc_kwargs", None) or dict(self.DEFAULT_SVC_KWARGS)
        self.exploration_per_batch = int(
            kwargs.get("exploration_per_batch", max(1, self.batch_size // 5))
        )
        if self.exploration_per_batch >= self.batch_size:
            raise ValueError(
                "exploration_per_batch must be smaller than batch_size "
                f"(got {self.exploration_per_batch} >= {self.batch_size})."
            )
        self.boundary_pool_fraction = float(kwargs.get("boundary_pool_fraction", 0.10))

        self.svm_model = None
        self.scaler = None

    # ------------------------------------------------------------
    # MODEL FITTING / PREDICTION
    # ------------------------------------------------------------
    def _fit_model(self):
        self.scaler = StandardScaler().fit(self.train_x)
        self.svm_model = SVC(**self.svc_kwargs)
        self.svm_model.fit(self.scaler.transform(self.train_x), self.train_y)

    def _predict_labels(self, X_unit):
        return self.svm_model.predict(self.scaler.transform(X_unit))

    def _fit_model_fold(self, X_tr, y_tr):
        self._fold_scaler = StandardScaler().fit(X_tr)
        self._fold_model = SVC(**self.svc_kwargs)
        self._fold_model.fit(self._fold_scaler.transform(X_tr), y_tr)

    def _predict_fold(self, X_val):
        return self._fold_model.predict(self._fold_scaler.transform(X_val))

    # ------------------------------------------------------------
    # ACQUISITION (used only for the boundary sub-batch's scoring; see
    # get_next_samples for the exploration/boundary split itself)
    # ------------------------------------------------------------
    def _compute_acquisition_unchunked(self, X_unit):
        return -_margin(self.svm_model, self.scaler.transform(X_unit))

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
                do_plot_residuals=self._should_trigger(self.plot_residuals_every),
            )

            explore_indices = self._get_initial_batch_n(self.exploration_per_batch)
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
        column is not a real simulation parameter -- gene_parser.py's
        parameter_nml_map / write_input_file only forward keys it recognizes
        (dropping unknown ones with a warning) and helena_parser.py's
        write_input_file_noKBMconstraint only reads named keys, so an unknown
        batch_num key is safely ignored by both and simply flows through to
        enchanted_dataset.csv. Recording it there (rather than saving model
        snapshots) is enough to fully reconstruct any checkpoint later: refit
        the sampler's model on the subset of rows with batch_num <= N using
        the same sampler config.
        """
        params_dict = super().samples_to_params_dict(samples)
        for p in params_dict:
            p["batch_num"] = self.batch_number
        return params_dict

    def _get_initial_batch_n(self, n):
        """
        Like ParentActiveSampler._get_initial_batch, but for an arbitrary
        batch size n instead of self.initial_batch_size. Used for the fixed
        per-batch exploration sub-batch.
        """
        cand_scores = np.array([], float)
        cand_indices = np.array([], int)
        self._reset_iterator()
        while True:
            X_chunk_unit, _, chunk_indices = self.get_next_pool_chunk()
            if X_chunk_unit is None:
                break
            scores = self.rng.random(len(X_chunk_unit))

            combined_scores = np.concatenate([cand_scores, scores])
            combined_indices = np.concatenate([cand_indices, chunk_indices])

            if len(combined_scores) > n:
                top_m = np.argpartition(combined_scores, -n)[-n:]
                cand_scores = combined_scores[top_m]
                cand_indices = combined_indices[top_m]
            else:
                cand_scores = combined_scores
                cand_indices = combined_indices

        return cand_indices.astype(int)

    def _compute_boundary_candidates(self, n, exclude=()):
        """
        Streams the remaining pool, restricts to the convex hull of the
        points sampled so far (see _in_hull), and draws n points uniformly
        at random from the most-uncertain (smallest-margin) slice of that
        hull-restricted pool -- mirroring active_sampling_svm.py's boundary
        draw exactly, but streaming-friendly.

        ``exclude`` is a set of pool indices to skip (e.g. this batch's
        exploration draws), so they aren't double-selected before either has
        been removed from the pool.
        """
        hull_points = self.scaler.transform(self.train_x)
        hull = _build_hull(hull_points)
        exclude = set(int(i) for i in exclude)

        all_scores = []
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

            X_chunk_scaled = self.scaler.transform(X_chunk_unit)
            inside = _in_hull(X_chunk_scaled, hull)
            if inside.sum() > 0:
                m = _margin(self.svm_model, X_chunk_scaled[inside])
                all_scores.append(m)
                all_indices.append(chunk_indices[inside])

        if not all_indices or sum(len(idx) for idx in all_indices) < n:
            log.warning(
                "Hull-restricted pool has fewer than %d classifiable candidates; "
                "falling back to an unrestricted random draw for the boundary sub-batch.",
                n,
            )
            fallback = self._get_initial_batch_n(n + len(exclude))
            fallback = fallback[~np.isin(fallback, list(exclude))]
            return fallback[:n]

        scores = np.concatenate(all_scores)
        indices = np.concatenate(all_indices)

        pool_size = max(n, int(len(indices) * self.boundary_pool_fraction))
        pool_size = min(pool_size, len(indices))
        # Smallest margin = most uncertain.
        boundary_pool = indices[np.argsort(scores)[:pool_size]]

        chosen = self.rng.choice(boundary_pool, size=n, replace=False)
        return chosen.astype(int)
