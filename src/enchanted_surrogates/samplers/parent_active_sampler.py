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
        # Class-discovery bookkeeping (classification samplers): number of
        # evaluated samples before each class was first seen.
        self._n_evaluated = 0
        self.class_first_found = {}
        
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
            
        
        # Optional class grouping for classification samplers (see
        # _group_labels). None = every label is its own class.
        self.classes = kwargs.get("classes", None)
        self.other_label = kwargs.get("other_label", "OTHER")
        if self.classes is not None:
            self.classes = list(self.classes)
            if len(set(self.classes)) != len(self.classes):
                raise ValueError(f"classes contains duplicates: {self.classes!r}")

        # Training data. train_y dtype is object here since labels may be
        # categorical (classification) or continuous (regression); task-specific
        # subclasses may re-initialize it with a narrower dtype.
        self.train_x = np.empty((0, self.input_dim), dtype=float)
        self.train_y = np.empty((0, self.output_dim), dtype=object)

        self._init_mcc_convergence(kwargs)

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

    # ------------------------------------------------------------
    # MCC CONVERGENCE STOPPING
    # ------------------------------------------------------------
    def _init_mcc_convergence(self, kwargs):
        """
        Optional stopping rule for classification samplers: stop once the
        class-vs-rest MCC of the monitored class(es) has stayed within a narrow
        band over the last few batches. Config (all in sampler_config):

            mcc_convergence_class   : str or list of str, optional
                Class label(s) to monitor, e.g. "KBM". None (default) disables
                the rule. With several classes, all must have converged.
            mcc_convergence_batches : int, default 5
                Number of most recent batch evaluations in the window.
            mcc_convergence_range   : float, default 0.02
                Converged when max - min of the class MCC over the window is
                <= this value. A window whose MCC is 0 throughout (class not
                learned yet) never counts as converged.

        The MCC comes from the confusion matrix of each evaluate_model() call
        (test set if test_data_csv is set, otherwise K-fold CV), so it needs no
        extra model fits. Child classes call _check_mcc_convergence(metrics)
        from evaluate_model(); it sets self._stop_early when converged.
        """
        cls = kwargs.get("mcc_convergence_class", None)
        self.mcc_convergence_classes = [] if cls is None else ([cls] if isinstance(cls, str) else list(cls))
        self.mcc_convergence_batches = int(kwargs.get("mcc_convergence_batches", 5))
        self.mcc_convergence_range = float(kwargs.get("mcc_convergence_range", 0.02))
        if self.mcc_convergence_batches < 2:
            raise ValueError("mcc_convergence_batches must be >= 2")
        if self.mcc_convergence_range < 0:
            raise ValueError("mcc_convergence_range must be >= 0")
        self._mcc_history = {c: [] for c in self.mcc_convergence_classes}
        self._mcc_history_n_train = -1

    @staticmethod
    def _class_vs_rest_mcc(cm, labels, label):
        """Class-vs-rest Matthews correlation of `label` from a confusion matrix (0.0 if undefined)."""
        labels = list(labels)
        if label not in labels:
            return 0.0
        i = labels.index(label)
        tp = float(cm[i, i]); fn = float(cm[i].sum()) - tp; fp = float(cm[:, i].sum()) - tp
        tn = float(cm.sum()) - tp - fn - fp
        den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        return float((tp * tn - fp * fn) / den) if den > 0 else 0.0

    def _check_mcc_convergence(self, metrics):
        """
        Record this batch's class-vs-rest MCC for each monitored class and set
        self._stop_early once every one has a full window whose range is within
        mcc_convergence_range. No-op if the rule is disabled or metrics is None.
        """
        if not self.mcc_convergence_classes or metrics is None or "confusion_matrix" not in metrics:
            return
        n_train = len(self.train_x)
        if n_train == self._mcc_history_n_train:  # repeated evaluation of the same training set
            return
        self._mcc_history_n_train = n_train
        k, tol = self.mcc_convergence_batches, self.mcc_convergence_range
        converged = True
        for c in self.mcc_convergence_classes:
            hist = self._mcc_history[c]
            hist.append(self._class_vs_rest_mcc(metrics["confusion_matrix"], metrics["labels"], c))
            win = hist[-k:]
            if len(win) < k or max(win) <= 0.0 or max(win) - min(win) > tol:
                converged = False
        if converged:
            log.warning(
                "MCC converged: %s changed by <= %.3f over the last %d batches (%d training samples); stopping the run.",
                ", ".join(f"{c} MCC={self._mcc_history[c][-1]:.3f}" for c in self.mcc_convergence_classes),
                tol, k, n_train,
            )
            self._stop_early = True

    # ------------------------------------------------------------
    # CPU-HOUR BUDGET
    # ------------------------------------------------------------
    def _init_cpuh_budget(self, kwargs):
        """
        Set up CPU-hour tracking from sampler config (see the samplers'
        docstrings for the user-facing description):

            cpuh_budget : float          total core-hours allowed
            cpuh_stages : list of {runtime_variable, cores_per_run}
            cores_per_run : int          shorthand for a single stage timed by
                                         self.time_output_variable

        The legacy gene_/helena_ runtime_variable / cores_per_run keys are
        still accepted and translated into two stages (deprecated).
        """
        budget = kwargs.get("cpuh_budget", None)
        self.cpuh_budget = float(budget) if budget is not None else None
        self.cpuh_used = 0.0
        self.cpuh_stages = []
        if self.cpuh_budget is None:
            return

        stages = kwargs.get("cpuh_stages", None)
        cores = kwargs.get("cores_per_run", None)
        legacy = {
            "gene": kwargs.get("gene_cores_per_run", None),
            "helena": kwargs.get("helena_cores_per_run", None),
        }
        if stages:
            self.cpuh_stages = [
                (st["runtime_variable"], int(st["cores_per_run"])) for st in stages
            ]
        elif cores is not None:
            tov = getattr(self, "time_output_variable", None)
            if not tov:
                raise ValueError(
                    "cores_per_run needs time_output_variable to be set; "
                    "otherwise use cpuh_stages."
                )
            self.cpuh_stages = [(tov, int(cores))]
        elif any(v is not None for v in legacy.values()):
            if any(v is None for v in legacy.values()):
                raise ValueError(
                    "cpuh_budget requires both gene_cores_per_run and "
                    "helena_cores_per_run (or use cpuh_stages)."
                )
            log.warning(
                "gene_/helena_ runtime_variable and cores_per_run are deprecated; "
                "use cpuh_stages or cores_per_run."
            )
            self.cpuh_stages = [
                (kwargs.get("gene_runtime_variable", "runtime_sec_gene"), int(legacy["gene"])),
                (kwargs.get("helena_runtime_variable", "runtime_sec_helena"), int(legacy["helena"])),
            ]
        else:
            raise ValueError(
                "cpuh_budget requires cores_per_run (single stage timed by "
                "time_output_variable) or cpuh_stages."
            )

    def _accumulate_cpuh(self, future_df):
        """
        Adds this batch's CPU-hour cost to self.cpuh_used, from every row
        submitted (successful or not -- a failed run still burns CPU-hours):
        sum over stages of runtime_variable (seconds) * cores_per_run / 3600.
        Missing columns or NaN runtimes contribute 0 (rather than raising), so
        a run missing one stage's timing doesn't block tracking for the rest.
        """
        parts = []
        for runtime_variable, cores in self.cpuh_stages:
            if runtime_variable in future_df.columns:
                sec = future_df[runtime_variable].fillna(0).to_numpy(dtype=float)
                parts.append(float(sec.sum()) / 3600.0 * cores)
            else:
                parts.append(0.0)
        batch_cpuh = sum(parts)
        self.cpuh_used += batch_cpuh
        detail = ", ".join(
            f"{rv}={h:.2f}" for (rv, _), h in zip(self.cpuh_stages, parts)
        )
        log.debug(
            f"Batch CPU-hours: {batch_cpuh:.2f} ({detail}); "
            f"running total: {self.cpuh_used:.2f}"
        )

    def _tag_eval_order(self, future_df):
        """
        Number the (successful, row-filtered) rows of a completed batch in
        evaluation order, continuing the running count of evaluated samples.
        Includes rows later dropped as unclassified -- they cost budget too.
        Adds an `_eval_n` column (1-based) used by _record_class_discovery.
        """
        future_df = future_df.assign(
            _eval_n=np.arange(len(future_df)) + self._n_evaluated + 1
        )
        self._n_evaluated += len(future_df)
        return future_df

    def _record_class_discovery(self, future_df, y_class):
        """
        Record, for each class not yet seen, how many samples had been
        evaluated (`n_evaluated`, counting unclassified rows) and how many
        labelled training points existed (`n_train`) when it first appeared.
        Call before appending the batch to the training set, with future_df
        carrying `_eval_n` (see _tag_eval_order) aligned row-for-row with
        y_class (the grouped labels). Writes class_discovery.csv.
        """
        n_train_before = len(self.train_x)
        new = False
        for i, label in enumerate(y_class):
            if label in self.class_first_found:
                continue
            self.class_first_found[label] = {
                "n_evaluated": int(future_df["_eval_n"].iloc[i]),
                "n_train": n_train_before + i + 1,
            }
            new = True
            log.info(
                f"Class {label!r} first found after "
                f"{self.class_first_found[label]['n_evaluated']} evaluated samples "
                f"({self.class_first_found[label]['n_train']} training samples)."
            )
        if new:
            rows = [{"class": k, **v} for k, v in self.class_first_found.items()]
            pd.DataFrame(rows).to_csv(
                os.path.join(self.base_run_dir, "class_discovery.csv"), index=False
            )

    def _group_labels(self, labels):
        """
        Map raw class labels onto the configured `classes` list.

        Labels listed in `classes` are kept as-is. Any other label is
        relabelled to `other_label` (default "OTHER") if that is one of the
        configured classes, otherwise it is dropped. With `classes=None`
        labels pass through unchanged.

        Returns (mapped_labels, keep_mask) as numpy arrays; callers must
        apply keep_mask to X (and any other per-row arrays) alongside labels.
        """
        labels = np.asarray(labels, dtype=object)
        keep = np.ones(len(labels), dtype=bool)
        if self.classes is None:
            return labels, keep
        known = set(self.classes)
        in_list = np.array([l in known for l in labels], dtype=bool)
        if self.other_label in known:
            mapped = np.where(in_list, labels, self.other_label).astype(object)
        else:
            mapped = labels
            keep = in_list
        return mapped, keep

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
