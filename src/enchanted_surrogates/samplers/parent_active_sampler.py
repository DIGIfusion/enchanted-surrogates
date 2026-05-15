import numpy as np
import os
from numpy.lib.format import open_memmap

import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

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

import math
from sklearn.model_selection import KFold
from sklearn.cluster import AgglomerativeClustering

from enchanted_surrogates.samplers.base_sampler import Sampler


from enchanted_surrogates.utils.logger import get_logger

log = get_logger(__name__)


class ParentActiveSampler(Sampler):
    """
    Model-agnostic active learning engine.
    Child classes must implement:
        - _fit_model()
        - _predict_mean_var_discrepancy(X_unit)
        - _compute_acquisition_unchunked(X_unit)
    """

    # ------------------------------------------------------------
    # INITIALISATION
    # ------------------------------------------------------------
    def __init__(self, **kwargs):

        # Core config
        self.parameters = kwargs.get("parameters")
        self.input_dim = len(self.parameters)

        self.bounds = kwargs.get("bounds")
        self.determine_bounds_from_pool = bool(kwargs.get("determine_bounds_from_pool", False))

        self.batch_size = int(kwargs.get("batch_size", 1))
        self.initial_batch_size = kwargs.get('initial_batch_size', self.batch_size)
        self.budget = kwargs.get('budget', self.batch_size)
        
        self.acquisition_batch_mode = kwargs.get("acquisition_batch_mode", "best_score")
        self.base_run_dir = kwargs.get('base_run_dir', None)
        assert self.base_run_dir is not None
        self.seed = kwargs.get('seed', None)
        self.output_variables = kwargs.get("output_variables", None)
        if self.output_variables is None:
            raise ValueError('Must set ouptu_variables in sampler_config. It is a list of strings naming the output variables used for training and active learning.')
        self.output_dim = 1 if isinstance(self.output_variables, str) else len(self.output_variables)
        self.residual_plot_save_dir = os.path.join(self.base_run_dir, 'residual_plots')
        
        self.plot_residuals_every = int(kwargs.get("plot_residuals_every", 0))
        self.write_batch_info_every = int(kwargs.get("write_batch_info_every", 0))
        
        # Bookkeeping
        self.removed_indices = set()
        self.batch_number = 0
        self.submitted = 0
        self.rng = np.random.default_rng(self.seed)
         
        # Diversity parameters
        self.dpp_sigma = kwargs.get("dpp_sigma", 0.1)
        self.dpp_lambda = kwargs.get("dpp_lambda", 1.0)
        self.dpp_M_alpha = kwargs.get("dpp_M_alpha", 5)

        # Pool streaming
        self.pool_chunk_size = int(kwargs.get("pool_chunk_size", None))
        if self.pool_chunk_size is None:
            raise ValueError('pool_chunk_size is missing from the sampler_config')
        tps = kwargs.get("total_pool_size", None)
        self.total_pool_size = int(tps) if tps is not None else None

        self.pool_csv_path = kwargs.get("pool_csv_path", None)
        self.pool_npy_path = kwargs.get("pool_npy_path", None)
        self.allowed_pool_values = kwargs.get("allowed_pool_values", None)
        self.clean_npy_pool_file = kwargs.get('clean_npy_pool_file', False)
        
        # Initialise pool
        self._init_pool_stream()
        
        if self.determine_bounds_from_pool and (self.pool_csv_path or self.pool_npy_path):
            self._compute_bounds_from_pool_stream()
        else:
            # Bounds → unit transforms
            if self.bounds is None:
                raise ValueError('bounds is missing from sampler config')
            self._lb = np.array([b[0] for b in self.bounds], dtype=float)
            self._ub = np.array([b[1] for b in self.bounds], dtype=float)
            self._range = self._ub - self._lb
            
        
        # Training data
        self.train_x = np.empty((0, self.input_dim), dtype=float)
        self.train_y = np.empty((0, self.output_dim), dtype=float)

        # Test set (optional)
        self.test_data_csv = kwargs.get("test_data_csv", None)
        self._test_X = None
        self._test_y = None
        if self.test_data_csv:
            self._load_test_set(self.test_data_csv)
        else:
            self.num_folds = kwargs.get('num_folds', 5)

    # ------------------------------------------------------------
    # UNIT / REAL TRANSFORMS
    # ------------------------------------------------------------
    def to_unit_numpy(self, X_real):
        return (X_real - self._lb) / self._range

    def from_unit_numpy(self, X_unit):
        return self._lb + X_unit * self._range

    # ------------------------------------------------------------
    # TEST SET LOADING
    # ------------------------------------------------------------
    def _load_test_set(self, csv_path):
        assert self.output_variables
        df = pd.read_csv(csv_path)
        X_real = df[self.parameters].to_numpy()
        y = df[self.output_variables].to_numpy()
        self._test_X = self.to_unit_numpy(X_real)
        
        self._test_y = self._ensure_2d(y)

    def _should_trigger(self, every_n):
        """
        Returns True if an action should trigger based on number of training samples.
        """
        if every_n <= 0:
            return False
        return (self.train_x.shape[0] % every_n) == 0
    
    def _init_pool_stream(self):
        """
        Initialise pool streaming from:
        1. NPY file
        2. CSV file
        3. Auto-generated random pool stored on disk (fallback)
        """

        # ---------------------------------------------------------
        # 1. NPY file
        # ---------------------------------------------------------
        if self.pool_npy_path is not None:
            self.X_pool = np.load(self.pool_npy_path, mmap_mode="r")
            self.y_pool = None
            self.total_pool_size = self.X_pool.shape[0]
            self._next_row_index = 0
            return

        # ---------------------------------------------------------
        # 2. CSV file
        # ---------------------------------------------------------
        if self.pool_csv_path is not None:

            # Determine total pool size WITHOUT loading full CSV
            with open(self.pool_csv_path, "r") as f:
                # subtract 1 for header
                self.total_pool_size = sum(1 for _ in f) - 1

            # Now create the streaming iterator
            self._csv_iter = pd.read_csv(self.pool_csv_path, chunksize=self.pool_chunk_size)
            self._next_row_index = 0
            return


        # ---------------------------------------------------------
        # 3. AUTO-GENERATE RANDOM POOL STORED ON DISK
        # ---------------------------------------------------------
        if self.bounds is None:
            raise ValueError('To auto generate the random pool the bounds need to be specified in sampler_config.')
        if self.total_pool_size is None:
            raise ValueError('total_pool_size is missing from the sampler_config')
        
        log.info("No pool source provided — generating random pool on disk")

        # Create tmp directory next to base_run_dir
        parent_dir = os.path.dirname(self.base_run_dir)
        tmp_dir = os.path.join(parent_dir, "tmp")
        os.makedirs(tmp_dir, exist_ok=True)

        pool_path = os.path.join(tmp_dir, "random_pool_X.npy")
        self.pool_npy_path = pool_path

        rng = np.random.default_rng(self.seed)
        N = self.total_pool_size
        D = self.input_dim

        # Create memmap file
        pool_memmap = open_memmap(
            pool_path,
            mode="w+",
            dtype=np.float32,
            shape=(N, D),
        )

        # Fill in chunks to avoid large RAM usage
        chunk = self.pool_chunk_size
        for start in range(0, N, chunk):
            end = min(start + chunk, N)
            size = end - start

            block = np.zeros((size, D), dtype=np.float32)

            for d, p in enumerate(self.parameters):
                lb, ub = self.bounds[d]

                if self.allowed_pool_values is not None and p in self.allowed_pool_values:
                    block[:, d] = rng.choice(self.allowed_pool_values[p], size=size)
                else:
                    block[:, d] = rng.uniform(lb, ub, size=size)

            pool_memmap[start:end] = block

        # Load as memmap for streaming
        self.X_pool = np.load(pool_path, mmap_mode="r")
        self.y_pool = None
        self.total_pool_size = self.X_pool.shape[0]
        self._next_row_index = 0

    def _delete_npy_pool(self):
        if self.pool_npy_path is not None and os.path.exists(self.pool_npy_path):
            os.remove(self.pool_npy_path)
    
    def _reset_iterator(self):
        self._next_row_index = 0
        if hasattr(self, "X_pool"):
            return
        if self.pool_csv_path is not None:
            self._csv_iter = pd.read_csv(self.pool_csv_path, chunksize=self.pool_chunk_size)

    def _compute_bounds_from_pool_stream(self):
        """
        Streams the pool and computes min/max bounds for each parameter.
        Works for CSV, NPY (mmap), or random pools.
        """
        lb = np.full(self.input_dim, np.inf, dtype=float)
        ub = np.full(self.input_dim, -np.inf, dtype=float)

        # Reset iterator so we stream from the beginning
        self._reset_iterator()

        while True:
            X_chunk_real, _, _ = self.get_next_pool_chunk(in_unit_space=False)
            if X_chunk_real is None:
                break
            lb = np.minimum(lb, X_chunk_real.min(axis=0))
            ub = np.maximum(ub, X_chunk_real.max(axis=0))

        # Update internal bounds
        self._lb = lb
        self._ub = ub
        self._range = ub - lb
        
        # ALSO update self.bounds in the canonical list-of-tuples format
        self.bounds = [(float(lb[i]), float(ub[i])) for i in range(self.input_dim)]

        # Reset iterator again for normal use
        self._reset_iterator()
    
    def get_next_pool_chunk(self, in_unit_space=True):
        if hasattr(self, "X_pool"):
            start = self._next_row_index
            if start >= self.total_pool_size:
                return None, None, None
            end = min(start + self.pool_chunk_size, self.total_pool_size)
            chunk = self.X_pool[start:end]
            indices = list(range(start, end))
            self._next_row_index = end

            mask = [i not in self.removed_indices for i in indices]
            if not any(mask):
                return self.get_next_pool_chunk()

            chunk = chunk[mask]
            indices = [i for i, keep in zip(indices, mask) if keep]
            return self.to_unit_numpy(chunk), None, indices

        # CSV mode
        try:
            df = next(self._csv_iter)
        except StopIteration:
            return None, None, None

        start = self._next_row_index
        end = start + len(df)
        indices = list(range(start, end))
        self._next_row_index = end

        mask = [i not in self.removed_indices for i in indices]
        df = df[mask]
        indices = [i for i, keep in zip(indices, mask) if keep]

        if df.empty:
            return self.get_next_pool_chunk()

        y_pool = None
        X_real = df[self.parameters].to_numpy()
        if not in_unit_space:
            return X_real, y_pool, indices            
        else:
            X_unit = self.to_unit_numpy(X_real)
            return X_unit.astype(np.float32), y_pool, indices

    # ------------------------------------------------------------
    # STREAM SCORES
    # ------------------------------------------------------------
    def _stream_scores(self):
        self._reset_iterator()
        while True:
            X_chunk_unit, _, chunk_indices = self.get_next_pool_chunk()
            if X_chunk_unit is None:
                break
            scores = self._compute_acquisition_unchunked(X_chunk_unit)
            yield scores, chunk_indices, X_chunk_unit

    # ------------------------------------------------------------
    # BATCH SELECTION
    # ------------------------------------------------------------
    def _compute_acquisition_candidates(self):
        K = self.batch_size
        N = self.total_pool_size
        D = self.input_dim

        M = int(self.dpp_M_alpha * K * (1 + math.log10(max(1, N/K))) * math.sqrt(D))
        M = min(M, N, self.pool_chunk_size)

        cand_scores = np.array([], float)
        cand_indices = np.array([], int)
        cand_x = np.empty((0, D), float)

        for scores, pool_indices, X_chunk_unit in self._stream_scores():
            combined_scores = np.concatenate([cand_scores, scores])
            combined_indices = np.concatenate([cand_indices, pool_indices])
            combined_x = np.vstack([cand_x, X_chunk_unit])

            if len(combined_scores) > M:
                top_m = np.argpartition(combined_scores, -M)[-M:]
                cand_scores = combined_scores[top_m]
                cand_indices = combined_indices[top_m]
                cand_x = combined_x[top_m]
            else:
                cand_scores = combined_scores
                cand_indices = combined_indices
                cand_x = combined_x

        mode = self.acquisition_batch_mode

        if mode == "best_score":
            top_k = np.argpartition(cand_scores, -K)[-K:]
            return cand_indices[top_k]

        if mode == "distance_penalisation":
            return self._distance_penalisation(cand_scores, cand_indices, cand_x, K)

        if mode == "approx_dpp_dynamic_clusters":
            return self._dynamic_clusters(cand_scores, cand_indices, cand_x, K)

        top_k = np.argpartition(cand_scores, -K)[-K:]
        return cand_indices[top_k]

    # ------------------------------------------------------------
    # Initial Batch SELECTION
    # ------------------------------------------------------------
    def _get_initial_batch(self):
        '''
        Randomly sample from the pool with streaming
        '''
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
        
            if len(combined_scores) > self.initial_batch_size:
                top_m = np.argpartition(combined_scores, -self.initial_batch_size)[-self.initial_batch_size:]
                cand_scores = combined_scores[top_m]
                cand_indices = combined_indices[top_m]
            else:
                cand_scores = combined_scores
                cand_indices = combined_indices
        
        return cand_indices

    def _distance_penalisation(self, scores, indices, X, K):
        selected = []
        remaining = np.arange(len(scores))

        first = np.argmax(scores)
        selected.append(first)
        remaining = remaining[remaining != first]

        while len(selected) < K and len(remaining) > 0:
            sel_x = X[selected]
            rem_x = X[remaining]

            dists = np.linalg.norm(rem_x[:, None, :] - sel_x[None, :, :], axis=-1)
            min_d = dists.min(axis=1)

            penalised = scores[remaining] - self.dpp_lambda * np.exp(-(min_d**2)/(2*self.dpp_sigma**2))
            best = remaining[np.argmax(penalised)]

            selected.append(best)
            remaining = remaining[remaining != best]

        return indices[selected]

    def _dynamic_clusters(self, scores, indices, X, K):
        clusterer = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=self.dpp_sigma,
            linkage="complete",
        )
        labels = clusterer.fit_predict(X)
        n_clusters = clusterer.n_clusters_

        selected = []
        for cid in range(n_clusters):
            mask = (labels == cid)
            best = np.argmax(scores[mask])
            selected.append(indices[mask][best])

        if len(selected) > K:
            sel_scores = [scores[indices == s][0] for s in selected]
            top = np.argsort(sel_scores)[-K:]
            selected = [selected[i] for i in top]

        return np.array(selected, int)

    # ------------------------------------------------------------
    # MAIN ENTRY: GET NEXT SAMPLES
    # ------------------------------------------------------------
    def get_next_samples(self):
        
        if self.batch_number == 0:
            initial_pool_indicies = self._get_initial_batch()
            real_selected_samples = self._get_samples_from_pool(initial_pool_indicies)        
        else:            
            self._fit_model() # defined in child class
            log.debug(f'Plot residuals every: {self.plot_residuals_every} | len(train): {len(self.train_x)}')
            log.debug(f'Should Trigger Plot Residuals: {self._should_trigger(self.plot_residuals_every)}')
            self.evaluate_model(do_write_batch_info=self._should_trigger(self.write_batch_info_every),
                                do_plot_residuals=self._should_trigger(self.plot_residuals_every))

            selected_indices = self._compute_acquisition_candidates()
            real_selected_samples = self._get_samples_from_pool(selected_indices)
            self._remove_from_pool(selected_indices)
        self.batch_number += 1
        self.submitted += len(real_selected_samples)
        params_dict = self.samples_to_params_dict(real_selected_samples)
        
        if not self.has_budget:
            self._light_post_process()
            return None
        
        return params_dict
    
    def samples_to_params_dict(self, samples):
        params = [{key: value for key, value in zip(self.parameters, params)} for params in samples]
        return params
    
    def _remove_from_pool(self, global_indices):
        arr = np.asarray(global_indices).astype(int).ravel()
        self.removed_indices.update(arr)
    
    def _get_samples_from_pool(self, global_indices, in_unit_space=False):
        selected_set = set(int(i) for i in global_indices)
        chosen_unit = []
        chosen_idx = []

        self._reset_iterator()
        while True:
            X_chunk_unit, _, chunk_indices = self.get_next_pool_chunk()
            if X_chunk_unit is None:
                break

            mask = [idx in selected_set for idx in chunk_indices]
            if any(mask):
                chosen_unit.append(X_chunk_unit[mask])
                chosen_idx.extend([idx for idx, keep in zip(chunk_indices, mask) if keep])

                if len(chosen_idx) >= len(global_indices):
                    break
        chosen_unit = np.vstack(chosen_unit)
        if in_unit_space:
            return chosen_unit
        else:
            return self.from_unit_numpy(chosen_unit)

        
    
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

        import matplotlib.pyplot as plt
        import numpy as np
        import os

        if out_dir is None:
            out_dir = os.path.join(self.base_run_dir, 'residuals_plots')
        os.makedirs(out_dir, exist_ok=True)

        y_pred = self._ensure_2d(y_pred)
        y_true = self._ensure_2d(y_true)

        residuals = y_pred - y_true
        M = y_true.shape[1]
        for m in range(M):
            yt = y_true[:, m]
            yp = y_pred[:, m] # giving an error if only 1 output
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
                f"RMSE={rmse:.4f} | R²={r2:.4f}", #| MAPE={mape:.2f}%
                ha='center', va='top', fontsize=9,
            )

            fig.tight_layout(rect=[0, 0, 1, 0.95])

            # Save
            file_name = f"{name}_residuals_{self.output_variables[m]}.png"
            file_name = safe_filename(file_name)
            save_path = os.path.join(
                out_dir, file_name                
            )
            log.debug(f'Saving Residuals at: {save_path}')
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
            plt.close(fig)

    def _ensure_2d(self, arr):
        """
        Ensures arr is a 2D array of shape (N, M).

        - (N,)        -> (N, 1)
        - (1, N)      -> (N, 1)
        - (N, 1)      -> unchanged
        - (N, M)      -> unchanged
        - lists/tuples -> converted to ndarray
        """
        arr = np.asarray(arr)

        # Already 2D
        if arr.ndim == 2:
            return arr

        # 1D -> make column vector
        if arr.ndim == 1:
            return arr.reshape(-1, 1)

        raise ValueError(f"Cannot convert array of shape {arr.shape} to 2D")

    
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
    
    def _light_post_process(self):
        if self.clean_npy_pool_file:
            self._delete_npy_pool()
    
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

def safe_filename(s, replacement="_"):
    """
    Convert an arbitrary string into a filesystem‑safe filename.

    - Removes or replaces characters illegal on Windows/macOS/Linux
    - Collapses repeated separators
    - Strips leading/trailing separators
    """

    # Characters forbidden on Windows:  \ / : * ? " < > |
    # Also remove control chars and anything non-printable
    s = re.sub(r'[\\/:*?"<>|\x00-\x1F]', replacement, s)

    # Replace spaces with underscore (optional)
    s = re.sub(r'\s+', replacement, s)

    # Collapse multiple replacements into one
    rep = re.escape(replacement)
    s = re.sub(rf'{rep}+', replacement, s)

    # Strip leading/trailing separators
    s = s.strip(replacement)

    # Fallback if string becomes empty
    return s or "untitled"
