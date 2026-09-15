import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.model_selection import KFold

from enchanted_surrogates.samplers.parent_active_sampler import ParentActiveSampler, safe_filename
from enchanted_surrogates.utils.logger import get_logger

log = get_logger(__name__)

# -----------------------------
# Scientific plotting style
# -----------------------------
mpl.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 18,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "axes.linewidth": 1.2,
    "lines.linewidth": 2.2,
    "text.usetex": False,  # Set to True if LaTeX is installed
    "font.family": "serif",
})

# Okabe–Ito colorblind-safe palette
OKABE_ITO = [
    "#000000", "#E69F00", "#56B4E9", "#009E73",
    "#F0E442", "#0072B2", "#D55E00", "#CC79A7"
]


class ParentActiveSamplerRegression(ParentActiveSampler):
    """
    Regression-flavoured active learning engine. Adds RMSE/MAPE/R2 evaluation
    (K-fold CV or held-out test set), residual plotting, and continuous-target
    training-set bookkeeping on top of ParentActiveSampler's generic pool
    streaming and batch acquisition machinery.

    Child classes must implement:
        - _fit_model()
        - _predict_mean_var(X_unit)
        - _compute_acquisition_unchunked(X_unit)
        - _fit_model_fold(X_tr, Y_tr)
        - _predict_fold(X_val)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Narrow train_y back to a float array now that the task is known to
        # be regression (ParentActiveSampler initializes it as object dtype
        # to stay task-agnostic).
        self.train_x = np.empty((0, self.input_dim), dtype=float)
        self.train_y = np.empty((0, self.output_dim), dtype=float)

    def compute_kfold_metrics(self):
        """
        Computes RMSE, MAPE, R2 using K-fold CV.
        Returns a dict with metrics and residuals.
        """

        X = self.train_x
        Y = self.train_y  # shape (N, M)

        if X.shape[0] < self.num_folds:
            return None

        kf = KFold(n_splits=self.num_folds, shuffle=True, random_state=self.seed)

        all_y_true = []
        all_y_pred = []
        rmses = []
        mapes = []
        r2s = []

        for train_idx, val_idx in kf.split(X):
            X_tr, X_val = X[train_idx], X[val_idx]
            Y_tr, Y_val = Y[train_idx], Y[val_idx]

            # Fit model on fold
            self._fit_model_fold(X_tr, Y_tr)

            # Predict
            Y_pred = self._predict_fold(X_val)  # child class must implement

            Y_val = self._ensure_2d(Y_val)
            Y_pred = self._ensure_2d(Y_pred)

            # Residuals
            residuals = Y_val - Y_pred
            all_y_true.append(Y_val)
            all_y_pred.append(Y_pred)

            self._warn_if_mape_unreliable(Y_val)

            # Metrics per output dimension
            rmse = np.sqrt(np.mean((residuals)**2, axis=0))
            mape = np.mean(np.abs(residuals * 100 / (Y_val + 1e-12)), axis=0)
            ss_res = np.sum((residuals)**2, axis=0)
            ss_tot = np.sum((Y_val - Y_val.mean(axis=0))**2, axis=0)
            r2 = 1 - ss_res / (ss_tot + 1e-12)

            rmses.append(rmse)
            mapes.append(mape)
            r2s.append(r2)

        # Stack lists of arrays
        y_pred_stacked = np.vstack(all_y_pred)
        y_true_stacked = np.vstack(all_y_true)

        # Ensure 2D shape
        y_pred_stacked = self._ensure_2d(y_pred_stacked)
        y_true_stacked = self._ensure_2d(y_true_stacked)

        return {
            "rmse": np.mean(rmses, axis=0),
            "mape": np.mean(mapes, axis=0),
            "r2": np.mean(r2s, axis=0),
            "y_true": y_true_stacked,
            "y_pred": y_pred_stacked,
        }

    def compute_testset_metrics(self):
        """
        If a test set is provided, compute RMSE, MAPE, R2 on it.
        """

        if self._test_X is None or self._test_y is None:
            return None

        # Fit on full training data
        self._fit_model()

        Y_pred, Y_var = self._predict_mean_var(self._test_X)

        # Ensure 2D BEFORE computing residuals
        Y_pred = self._ensure_2d(Y_pred)
        Y_true = self._ensure_2d(self._test_y)
        residuals = self._test_y - Y_pred

        self._warn_if_mape_unreliable(self._test_y)

        rmse = np.sqrt(np.mean(residuals**2, axis=0))
        mape = np.mean(np.abs(residuals * 100 / (self._test_y + 1e-12)), axis=0)
        ss_res = np.sum(residuals**2, axis=0)
        ss_tot = np.sum((self._test_y - self._test_y.mean(axis=0))**2, axis=0)
        r2 = 1 - ss_res / (ss_tot + 1e-12)

        Y_pred = self._ensure_2d(Y_pred)
        self._test_y = self._ensure_2d(self._test_y)

        return {
            "rmse": rmse,
            "mape": mape,
            "r2": r2,
            "y_pred": Y_pred,
            "y_true": self._test_y
        }

    def _warn_if_mape_unreliable(self, Y):
        """
        Warns if MAPE cannot be trusted because Y contains values near zero.
        """
        eps = 1e-6
        if np.any(np.abs(Y) < eps):
            log.warning(
                f"MAPE may be unreliable because some true values "
                f"are near zero (|y| < {eps}). "
                "MAPE divides by y_true, so values close to zero cause "
                "artificially huge percentages."
            )

    def plot_regression_residuals(self, y_true, y_pred, name='', out_dir=None):
        """
        Creates regression-style residual plots for each output variable:
        1. Residuals hexbin (y_true vs residual)
        2. True vs Predicted hexbin (with diagonal)
        """

        if out_dir is None:
            out_dir = os.path.join(self.base_run_dir, 'residuals_plots')
        os.makedirs(out_dir, exist_ok=True)

        y_pred = self._ensure_2d(y_pred)
        y_true = self._ensure_2d(y_true)

        residuals = y_pred - y_true
        M = y_true.shape[1]
        for m in range(M):
            yt = y_true[:, m]
            yp = y_pred[:, m]  # giving an error if only 1 output
            res = residuals[:, m]

            # -----------------------------
            # Metrics
            # -----------------------------
            mse = np.mean((yp - yt)**2)
            rmse = np.sqrt(mse)
            mape = np.mean(np.abs((yp - yt) / (yt + 1e-12))) * 100

            ss_res = np.sum((yp - yt)**2)
            ss_tot = np.sum((yt - yt.mean())**2)
            r2 = 1 - ss_res / (ss_tot + 1e-12)

            # -----------------------------
            # Plotting
            # -----------------------------
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))

            # 1. Residuals hexbin
            ax = axes[0]
            hb = ax.hexbin(yt, res, gridsize=40, cmap="viridis", mincnt=1)
            fig.colorbar(hb, ax=ax).set_label("Count")
            ax.set_xlabel("y_true")
            ax.set_ylabel("Residual (y_pred - y_true)")
            ax.set_title(f"Residuals ({self.output_variables[m]})")

            # 2. True vs Predicted
            ax2 = axes[1]
            hb2 = ax2.hexbin(yt, yp, gridsize=40, cmap="viridis", mincnt=1)
            fig.colorbar(hb2, ax=ax2).set_label("Count")
            ax2.set_xlabel("y_true")
            ax2.set_ylabel("y_pred")
            ax2.set_title(f"True vs Predicted ({self.output_variables[m]})")

            # Diagonal reference
            lo = min(yt.min(), yp.min())
            hi = max(yt.max(), yp.max())
            ax2.plot([lo, hi], [lo, hi], 'r--', linewidth=1)

            # Metrics text
            fig.text(
                0.5, 0.98,
                f"RMSE={rmse:.4f} | R²={r2:.4f}",  # | MAPE={mape:.2f}%
                ha='center', va='top', fontsize=9,
            )

            fig.tight_layout(rect=[0, 0, 1, 0.95])

            # Save
            file_name = f"{name}_residuals_{self.output_variables[m]}.png"
            file_name = safe_filename(file_name)
            save_path = os.path.join(out_dir, file_name)
            log.debug(f'Saving Residuals at: {save_path}')
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
            plt.close(fig)

    def plot_batch_info(self):
        # Load CSV
        df = pd.read_csv(os.path.join(self.base_run_dir, "batch_info.csv"))

        metric_types = ["rmse", "mape", "r2"]
        titles = {"rmse": "RMSE", "mape": "MAPE", "r2": "R²"}

        # Group columns by metric type
        groups = {m: {} for m in metric_types}
        for col in df.columns:
            for m in metric_types:
                if col.endswith("_" + m):
                    base = col[: -(len(m) + 1)]
                    groups[m][base] = col

        # Consistent colors
        base_vars = sorted({b for g in groups.values() for b in g.keys()})
        color_map = {base: OKABE_ITO[i % len(OKABE_ITO)] for i, base in enumerate(base_vars)}

        save_dir = os.path.join(self.base_run_dir, "performance_data_efficiency_plots")
        os.makedirs(save_dir, exist_ok=True)

        # Create one figure per metric
        for metric in metric_types:
            fig, ax = plt.subplots(figsize=(3.5, 2.5))

            for base, col in groups[metric].items():
                ax.plot(
                    df["num_train_samples"],
                    df[col],
                    label=base,
                    color=color_map[base],
                    linewidth=2
                )

            ax.set_title(titles[metric])
            ax.set_xlabel("num_train_samples")
            ax.set_ylabel(metric.upper())
            ax.grid(True, alpha=0.3)

            # Single legend per figure
            ax.legend(title="Quantity", frameon=False)

            fig.tight_layout()
            fig.savefig(os.path.join(save_dir, f"performance_{metric}.png"), dpi=300)
            plt.close(fig)

    def evaluate_model(self, do_write_batch_info=False, do_plot_residuals=False):
        """
        Runs either test-set evaluation or K-fold CV.
        Writes metrics and plots if requested.
        """
        log.debug(f'test_X : {self._test_X}')
        if self._test_X is not None:
            plot_name = f'test-{len(self._test_X)}_train-{len(self.train_x)}'
            metrics = self.compute_testset_metrics()
        else:
            plot_name = f'Nfold-{self.num_folds}_train-{len(self.train_x)}'
            metrics = self.compute_kfold_metrics()

        if metrics is None:
            return None

        # Write batch info
        if do_write_batch_info:
            self.write_batch_info(metrics)

        # Plot residuals
        if do_plot_residuals:
            self.plot_regression_residuals(metrics['y_true'], metrics['y_pred'], name=plot_name)

        return metrics

    def write_batch_info(self, metrics):
        """
        Writes RMSE, MAPE, R2 and training sample count to CSV.
        """

        row = {
            "num_train_samples": self.train_x.shape[0],
        }

        for i, name in enumerate(self.output_variables):
            row[f"{name}_rmse"] = metrics["rmse"][i]
            row[f"{name}_mape"] = metrics["mape"][i]
            row[f"{name}_r2"] = metrics["r2"][i]

        df = pd.DataFrame([row])

        csv_path = os.path.join(self.base_run_dir, 'batch_info.csv')
        if not os.path.exists(csv_path):
            df.to_csv(csv_path, index=False)
        else:
            df.to_csv(csv_path, mode="a", header=False, index=False)

        self.plot_batch_info()

    def register_future(self, future_df):
        """
        Register a completed evaluation.

        Parameters
        ----------
        future_df : pandas.DataFrame
            Must contain columns for all parameters and all output variables.

        Adds the observation(s) to the internal dataset.
        Supports multiple output variables.
        """
        # only add succedded outputs to the training set
        future_df = future_df[future_df['success']]

        future_df = self._apply_row_filters(future_df)

        if future_df.empty:
            return

        # Extract X and Y
        X_real = future_df[self.parameters].to_numpy(dtype=float)
        Y = future_df[self.output_variables].to_numpy(dtype=float)
        Y = self._ensure_2d(Y)

        # Convert X to unit space
        X_unit = self.to_unit_numpy(X_real)

        # Append to training data
        self.train_x = np.vstack([self.train_x, X_unit])
        self.train_y = np.vstack([self.train_y, Y])

        msg = (
            f"future_df rows: {len(future_df)}\n"
            f"X_real shape: {X_real.shape}\n"
            f"X_unit shape: {X_unit.shape}\n"
            f"train_x new shape: {self.train_x.shape}"
        )

        log.debug(msg)
