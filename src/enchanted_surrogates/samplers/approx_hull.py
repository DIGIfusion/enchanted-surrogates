"""
PCA-whitened ellipsoid approximation of "inside the convex hull of the
points sampled so far", as a drop-in swap for svm_active_sampler.py's
_build_hull/_in_hull (exact Delaunay triangulation) in the *_tuned active
samplers.

Why this exists: scipy/Qhull's Delaunay triangulation is combinatorially
expensive past ~6-8 input dimensions (see _build_hull's own docstring) --
confirmed via the mem_issue_debug branch's RSS/timing logs on a 10-D active-
learning run: 725MB/31s at 60 training points, 1.4GB/125s at 80, both still
climbing with no problem-size increase, purely from Delaunay(train_x) getting
more expensive every single cycle as train_x grows. An ellipsoid boundary
test is not exact (it can admit some points outside the true convex hull, in
concave "pockets" of the sampled region, and can also exclude some extreme
points near thin, elongated regions of the actual hull), but it is O(train
points x dim^2) to fit and O(1) per candidate point to test, with no
dependence on the point count blowing up test cost -- it stays cheap and
memory-bounded regardless of dimension or training-set size, which is what
this dataset's 10-D parameter space needs.

Method: whiten train_x via PCA (handles correlated/differently-scaled
dimensions, unlike an axis-aligned bounding box), then treat a candidate as
"inside" if its whitened Euclidean norm from the training-set centroid is
within (1 + margin) times the maximum whitened norm among the training
points themselves -- i.e. an ellipsoid just large enough to contain every
training point, inflated by a small margin so points near the true hull
boundary (which the training points approximate, not exactly trace) aren't
spuriously excluded.
"""
import numpy as np


class ApproxHull:
    """
    Built once per cycle from hull_points (mirrors _build_hull's contract),
    reused across many is_inside(...) calls (one per streamed pool chunk).
    """

    def __init__(self, hull_points, margin=0.15):
        hull_points = np.asarray(hull_points, dtype=float)
        self._degenerate = hull_points.shape[0] < 3
        if self._degenerate:
            return

        self.mean = hull_points.mean(axis=0)
        centered = hull_points - self.mean

        # PCA via SVD: centered = U @ diag(S) @ Vt : columns of Vt.T are the
        # principal directions, S/sqrt(n) are their standard deviations.
        n = centered.shape[0]
        _, S, Vt = np.linalg.svd(centered, full_matrices=False)
        std = S / np.sqrt(max(n - 1, 1))
        # Directions with ~zero spread (e.g. a constant input dimension, or
        # fewer distinct points than dimensions) would divide-by-zero in the
        # whitening transform below; floor them so those directions
        # contribute ~0 to the whitened distance instead of blowing up.
        std_floor = np.maximum(std, 1e-8 * (std.max() if std.max() > 0 else 1.0))

        self.components = Vt  # (d, d), rows are principal directions
        self.inv_std = 1.0 / std_floor

        whitened_train = (centered @ self.components.T) * self.inv_std
        train_radii = np.linalg.norm(whitened_train, axis=1)
        self.radius = train_radii.max() * (1.0 + margin) if len(train_radii) else 0.0

    def is_inside(self, points):
        points = np.asarray(points, dtype=float)
        if self._degenerate:
            return np.ones(len(points), dtype=bool)
        centered = points - self.mean
        whitened = (centered @ self.components.T) * self.inv_std
        radii = np.linalg.norm(whitened, axis=1)
        return radii <= self.radius


def build_approx_hull(hull_points, margin=0.15):
    """
    Functional-style wrapper matching _build_hull's call signature, so
    callers can swap `from .svm_active_sampler import _build_hull, _in_hull`
    for `from .approx_hull import build_approx_hull as _build_hull,
    approx_in_hull as _in_hull` with no other code changes.
    """
    return ApproxHull(hull_points, margin=margin)


def approx_in_hull(points, hull):
    """Matches _in_hull's call signature (hull may be None -> keep everything)."""
    if hull is None:
        return np.ones(len(points), dtype=bool)
    return hull.is_inside(points)
