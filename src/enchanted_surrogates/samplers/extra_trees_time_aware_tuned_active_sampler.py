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
"""

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.metrics import f1_score

from enchanted_surrogates.samplers.extra_trees_time_aware_active_sampler import (
    ExtraTreesTimeAwareActiveSampler,
)
from enchanted_surrogates.utils.logger import get_logger

log = get_logger(__name__)


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
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
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
                clf = ExtraTreesClassifier(**kwargs, random_state=0)
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
