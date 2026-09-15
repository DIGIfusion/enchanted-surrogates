import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support

from enchanted_surrogates.samplers.parent_active_sampler import ParentActiveSampler
from enchanted_surrogates.utils.logger import get_logger

log = get_logger(__name__)


class ParentActiveSamplerClassification(ParentActiveSampler):
    """
    Classification-flavoured active learning engine. Adds accuracy/macro-F1
    evaluation (stratified K-fold CV or held-out test set) and categorical
    training-set bookkeeping on top of ParentActiveSampler's generic pool
    streaming and batch acquisition machinery.

    Only a single categorical output variable is supported (multi-output
    classification is out of scope).

    Configuration (in addition to ParentActiveSampler's):
        unclassified_label : str or None, optional
            A label value that carries no class information (e.g. GENE runs
            where no known instability regime matched) and should be
            excluded from training rather than learned as a real class --
            matching the reference active_sampling_svm.py demo script's
            treatment of "Unclassified" points. Rows with this label are
            dropped in register_future (the point is still "spent": GENE
            ran, it just didn't yield a usable label). Set to None to
            disable this filtering and treat it as a real class instead.
            Default: "Unclassified".

    Child classes must implement:
        - _fit_model()
        - _predict_labels(X_unit) -> array of predicted class labels
        - _compute_acquisition_unchunked(X_unit)
        - _fit_model_fold(X_tr, y_tr)
        - _predict_fold(X_val) -> array of predicted class labels
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.output_dim != 1:
            raise ValueError(
                "ParentActiveSamplerClassification supports exactly one "
                f"categorical output_variable, got {self.output_variables!r}."
            )
        self.unclassified_label = kwargs.get("unclassified_label", "Unclassified")
        # Narrow train_y to hold categorical labels (object dtype), now that
        # the task is known to be classification.
        self.train_x = np.empty((0, self.input_dim), dtype=float)
        self.train_y = np.empty((0,), dtype=object)

    # ------------------------------------------------------------
    # EVALUATION
    # ------------------------------------------------------------
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

    def compute_kfold_metrics(self):
        """
        Computes accuracy, macro-F1 and a confusion matrix using stratified
        K-fold CV. Returns a dict with metrics and predictions, or None if
        there isn't enough training data yet.
        """
        X = self.train_x
        y = self.train_y

        class_counts = pd.Series(y).value_counts()
        if len(y) < self.num_folds or class_counts.min() < self.num_folds:
            return None

        skf = StratifiedKFold(n_splits=self.num_folds, shuffle=True, random_state=self.seed)

        all_y_true = []
        all_y_pred = []
        accuracies = []
        f1s = []

        for train_idx, val_idx in skf.split(X, y):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            self._fit_model_fold(X_tr, y_tr)
            y_pred = self._predict_fold(X_val)

            all_y_true.append(y_val)
            all_y_pred.append(y_pred)
            accuracies.append(accuracy_score(y_val, y_pred))
            f1s.append(f1_score(y_val, y_pred, average="macro", zero_division=0))

        y_true_all = np.concatenate(all_y_true)
        y_pred_all = np.concatenate(all_y_pred)
        labels = sorted(pd.unique(np.concatenate([y_true_all, y_pred_all])))

        return {
            "accuracy": float(np.mean(accuracies)),
            "f1_macro": float(np.mean(f1s)),
            **self._per_class_metrics(y_true_all, y_pred_all, labels),
            "y_true": y_true_all,
            "y_pred": y_pred_all,
            "labels": labels,
            "confusion_matrix": confusion_matrix(y_true_all, y_pred_all, labels=labels),
        }

    def compute_testset_metrics(self):
        """
        If a test set is provided, compute accuracy/macro-F1 on it.
        """
        if self._test_X is None or self._test_y is None:
            return None

        self._fit_model()

        y_true = np.asarray(self._test_y).reshape(-1)
        y_pred = np.asarray(self._predict_labels(self._test_X)).reshape(-1)
        labels = sorted(pd.unique(np.concatenate([y_true, y_pred])))

        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            **self._per_class_metrics(y_true, y_pred, labels),
            "y_true": y_true,
            "y_pred": y_pred,
            "labels": labels,
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels),
        }

    def plot_confusion_matrix(self, metrics, name='', out_dir=None):
        """
        Plots the confusion matrix from a compute_kfold_metrics /
        compute_testset_metrics result dict.
        """
        if out_dir is None:
            out_dir = os.path.join(self.base_run_dir, 'confusion_matrix_plots')
        os.makedirs(out_dir, exist_ok=True)

        labels = metrics["labels"]
        cm = metrics["confusion_matrix"]

        fig, ax = plt.subplots(figsize=(1.2 * len(labels) + 2, 1.2 * len(labels) + 2))
        im = ax.imshow(cm, cmap="viridis")
        fig.colorbar(im, ax=ax).set_label("Count")

        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(
            f"Confusion matrix (acc={metrics['accuracy']:.1%}, "
            f"F1-macro={metrics['f1_macro']:.3f})"
        )

        thresh = cm.max() / 2.0 if cm.size else 0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                         color="white" if cm[i, j] > thresh else "black")

        fig.tight_layout()
        file_name = f"{name}_confusion_matrix.png" if name else "confusion_matrix.png"
        save_path = os.path.join(out_dir, file_name)
        log.debug(f'Saving confusion matrix at: {save_path}')
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    def plot_batch_info(self):
        df = pd.read_csv(os.path.join(self.base_run_dir, "batch_info.csv"))

        save_dir = os.path.join(self.base_run_dir, "performance_data_efficiency_plots")
        os.makedirs(save_dir, exist_ok=True)

        for metric, title in [("accuracy", "Accuracy"), ("f1_macro", "F1 (macro)")]:
            if metric not in df.columns:
                continue
            fig, ax = plt.subplots(figsize=(3.5, 2.5))
            ax.plot(df["num_train_samples"], df[metric], linewidth=2)
            ax.set_title(title)
            ax.set_xlabel("num_train_samples")
            ax.set_ylabel(title)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(save_dir, f"performance_{metric}.png"), dpi=300)
            plt.close(fig)

    def evaluate_model(self, do_write_batch_info=False, do_plot_residuals=False):
        """
        Runs either test-set evaluation or stratified K-fold CV. Writes
        metrics and plots the confusion matrix if requested. The
        `do_plot_residuals` name is kept for interface parity with the
        regression sampler; here it triggers a confusion-matrix plot.
        """
        if self._test_X is not None:
            plot_name = f'test-{len(self._test_X)}_train-{len(self.train_x)}'
            metrics = self.compute_testset_metrics()
        else:
            plot_name = f'Nfold-{self.num_folds}_train-{len(self.train_x)}'
            metrics = self.compute_kfold_metrics()

        if metrics is None:
            return None

        if do_write_batch_info:
            self.write_batch_info(metrics)

        if do_plot_residuals:
            self.plot_confusion_matrix(metrics, name=plot_name)

        return metrics

    def write_batch_info(self, metrics):
        """
        Writes accuracy, macro-F1, per-class precision/recall/F1, and
        training sample count to CSV.
        """
        scalar_metrics = {
            k: v for k, v in metrics.items()
            if k not in ("labels", "confusion_matrix", "y_true", "y_pred")
        }
        row = {"num_train_samples": self.train_x.shape[0], **scalar_metrics}

        df = pd.DataFrame([row])

        csv_path = os.path.join(self.base_run_dir, 'batch_info.csv')
        if not os.path.exists(csv_path):
            df.to_csv(csv_path, index=False)
        else:
            df.to_csv(csv_path, mode="a", header=False, index=False)

        self.plot_batch_info()

    # ------------------------------------------------------------
    # TRAINING SET BOOKKEEPING
    # ------------------------------------------------------------
    def register_future(self, future_df):
        """
        Register a completed evaluation.

        Parameters
        ----------
        future_df : pandas.DataFrame
            Must contain columns for all parameters and the single
            categorical output variable.

        Adds the observation(s) to the internal dataset.
        """
        future_df = future_df[future_df['success']]

        future_df = self._apply_row_filters(future_df)

        output_col = self.output_variables if isinstance(self.output_variables, str) \
            else self.output_variables[0]

        if self.unclassified_label is not None:
            n_before = len(future_df)
            future_df = future_df[future_df[output_col] != self.unclassified_label]
            n_dropped = n_before - len(future_df)
            if n_dropped:
                log.debug(
                    f"Dropped {n_dropped} row(s) labeled {self.unclassified_label!r} "
                    "(carries no class information; not added to training set)."
                )

        if future_df.empty:
            return

        X_real = future_df[self.parameters].to_numpy(dtype=float)
        y = future_df[output_col].to_numpy()

        X_unit = self.to_unit_numpy(X_real)

        self.train_x = np.vstack([self.train_x, X_unit])
        self.train_y = np.concatenate([self.train_y, y])

        msg = (
            f"future_df rows: {len(future_df)}\n"
            f"X_real shape: {X_real.shape}\n"
            f"X_unit shape: {X_unit.shape}\n"
            f"train_x new shape: {self.train_x.shape}"
        )
        log.debug(msg)
