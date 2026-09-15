import numpy as np
import os
from numpy.lib.format import open_memmap

import re
import pandas as pd

import math
from sklearn.cluster import AgglomerativeClustering

from enchanted_surrogates.samplers.base_sampler import Sampler


from enchanted_surrogates.utils.logger import get_logger

log = get_logger(__name__)


class ParentActiveSampler(Sampler):
    """
    Task-agnostic active learning engine: owns pool streaming (CSV/NPY/random),
    unit-space transforms, batch acquisition selection (best_score/DPP-style),
    and the get_next_samples/register_future main loop.

    Not specific to regression or classification. Child classes must implement:
        - _fit_model()
        - _compute_acquisition_unchunked(X_unit)
        - evaluate_model(do_write_batch_info, do_plot_residuals) (task-specific
          CV/metrics; see ParentActiveSamplerRegression / ParentActiveSamplerClassification)
        - register_future(future_df) (task-specific label/target handling)
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
        self.reject_if_false = kwargs.get("reject_if_false", [])
        self.reject_if_true = kwargs.get("reject_if_true", [])
        self.criteria = kwargs.get("criteria", [])
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
        self._last_triggered_count = {}
         
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
            
        
        # Training data. train_y dtype is object here since labels may be
        # categorical (classification) or continuous (regression); task-specific
        # subclasses may re-initialize it with a narrower dtype.
        self.train_x = np.empty((0, self.input_dim), dtype=float)
        self.train_y = np.empty((0, self.output_dim), dtype=object)

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
    @staticmethod
    def _parse_criterion(expr):
        """
        Parse a human-readable criterion string.

        Accepted formats:
            reject -5 <= gamma <= 0
            reject -inf <= gamma <= 0
            accept 0 < paramA < 5
            reject gamma <= 0          (single right-sided)
            reject 0 <= gamma          (single left-sided)

        Returns (action, col, lo, op_lo, hi, op_hi) where:
            action  : 'reject' or 'accept'
            col     : column name (str)
            lo/hi   : float bound or None (None = unbounded on that side)
            op_lo/op_hi : '<' or '<=' or None
        """
        expr = expr.strip()
        m = re.match(r'^(reject|accept)\s+(.+)$', expr, re.IGNORECASE)
        if not m:
            raise ValueError(f"Criterion must start with 'reject' or 'accept': {expr!r}")

        action = m.group(1).lower()
        rest = m.group(2).strip()

        num = r'([+-]?inf|[+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)'
        op  = r'(<=|<)'
        col = r'([^\s<>=]+)'

        def parse_num(s):
            f = float(s)
            return None if (f == float('inf') or f == float('-inf')) else f

        # Two-sided: num op col op num
        two = re.fullmatch(rf'{num}\s*{op}\s*{col}\s*{op}\s*{num}', rest)
        if two:
            lo_s, op_lo, c, op_hi, hi_s = two.groups()
            lo, hi = parse_num(lo_s), parse_num(hi_s)
            return action, c, lo, (op_lo if lo is not None else None), hi, (op_hi if hi is not None else None)

        # Right-sided: col op num
        right = re.fullmatch(rf'{col}\s*{op}\s*{num}', rest)
        if right:
            c, op_hi, hi_s = right.groups()
            return action, c, None, None, parse_num(hi_s), op_hi

        # Left-sided: num op col
        left = re.fullmatch(rf'{num}\s*{op}\s*{col}', rest)
        if left:
            lo_s, op_lo, c = left.groups()
            return action, c, parse_num(lo_s), op_lo, None, None

        raise ValueError(f"Cannot parse criterion expression: {rest!r}")

    @staticmethod
    def _apply_criterion(df, action, col, lo, op_lo, hi, op_hi):
        if col not in df.columns:
            return df
        vals = df[col].astype(float)
        cond = pd.Series(True, index=df.index)
        if lo is not None:
            cond &= (vals > lo) if op_lo == '<' else (vals >= lo)
        if hi is not None:
            cond &= (vals < hi) if op_hi == '<' else (vals <= hi)
        return df[~cond] if action == 'reject' else df[cond]

    def _apply_row_filters(self, df):
        """
        Drop rows excluded from both training and test sets:
          - NaN in any output variable
          - falsy value in any reject_if_false column
          - truthy value in any reject_if_true column
          - any criterion in criteria list, e.g.:
              "reject -inf <= gamma <= 0"
              "accept 0 < paramA < 5"
        """
        output_cols = [self.output_variables] if isinstance(self.output_variables, str) \
            else self.output_variables
        df = df[df[output_cols].notna().all(axis=1)]
        for col in self.reject_if_false:
            if col in df.columns:
                df = df[df[col].astype(float).astype(bool)]
        for col in self.reject_if_true:
            if col in df.columns:
                df = df[~df[col].astype(float).astype(bool)]
        for expr in self.criteria:
            df = self._apply_criterion(df, *self._parse_criterion(expr))
        return df

    def _load_test_set(self, csv_path):
        assert self.output_variables
        df = pd.read_csv(csv_path)
        df = self._apply_row_filters(df)
        X_real = df[self.parameters].to_numpy()
        y = df[self.output_variables].to_numpy()
        self._test_X = self.to_unit_numpy(X_real)
        self._test_y = self._ensure_2d(y)

    def _should_trigger(self, every_n):
        """
        Returns True if an action should trigger based on number of training
        samples. Fires whenever train_x.shape[0] has crossed (not just
        landed exactly on) a multiple of every_n since the last time this
        every_n was checked -- growth in train_x is not guaranteed to land
        on exact multiples (e.g. classification samplers drop Unclassified
        rows before adding to train_x), so an exact-modulo check can permanently
        miss every trigger for a whole run. Tracked per distinct every_n value
        so plot_residuals_every and write_batch_info_every (which may differ)
        don't interfere with each other.
        """
        if every_n <= 0:
            return False
        current = self.train_x.shape[0]
        last = self._last_triggered_count.get(every_n, 0)
        if (current // every_n) > (last // every_n):
            self._last_triggered_count[every_n] = current
            return True
        return False
    
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

        # Drop rows where any parameter column is NaN (e.g. failed upstream runs)
        nan_valid = df[self.parameters].notna().all(axis=1).to_numpy()
        if not nan_valid.all():
            df = df[nan_valid]
            indices = [idx for idx, keep in zip(indices, nan_valid) if keep]

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

        if arr.ndim == 2:
            return arr
        if arr.ndim == 1:
            return arr.reshape(-1, 1)

        raise ValueError(f"Cannot convert array of shape {arr.shape} to 2D")

    def _light_post_process(self):
        if self.clean_npy_pool_file:
            self._delete_npy_pool()

    # ------------------------------------------------------------
    # TASK-SPECIFIC HOOKS (see ParentActiveSamplerRegression /
    # ParentActiveSamplerClassification for implementations)
    # ------------------------------------------------------------
    def evaluate_model(self, do_write_batch_info=False, do_plot_residuals=False):
        """
        Runs CV/test-set evaluation appropriate to the task, writes metrics
        and plots if requested. Must be implemented by a task-specific
        subclass (regression or classification).
        """
        raise NotImplementedError(
            "evaluate_model must be implemented by a task-specific subclass "
            "(e.g. ParentActiveSamplerRegression, ParentActiveSamplerClassification)."
        )

    def register_future(self, future_df):
        """
        Register a completed evaluation, appending it to the training set.
        Must be implemented by a task-specific subclass since label/target
        handling differs between regression (continuous targets) and
        classification (categorical labels).

        Parameters
        ----------
        future_df : pandas.DataFrame
            Must contain columns for all parameters and all output variables.
        """
        raise NotImplementedError(
            "register_future must be implemented by a task-specific subclass "
            "(e.g. ParentActiveSamplerRegression, ParentActiveSamplerClassification)."
        )


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
