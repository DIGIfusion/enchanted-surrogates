from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from enchanted_surrogates.samplers.parent_active_sampler import ParentActiveSampler
class ExtraTreesActiveSampler(ParentActiveSampler):
    """
    Extra Trees surrogate plugged into the model-agnostic parent sampler.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.n_estimators = int(kwargs.get("n_estimators", 200))
        self.max_depth = kwargs.get("max_depth", None)
        self.min_samples_leaf = int(kwargs.get("min_samples_leaf", 1))
        self.n_jobs = int(kwargs.get("n_jobs", -1))
        self.random_state = kwargs.get("seed", None)

        self.acq_beta = float(kwargs.get("acq_beta", 1.0))
        self.et_model = None
        self.acquisition_mode = kwargs.get('acquisition_mode', 'var') 

    # ------------------------------------------------------------
    # MODEL FITTING
    # ------------------------------------------------------------
    def _fit_model(self):
        X = self.train_x
        y_raw = self.train_y

        # --- Auto-detect and normalize y shape ---
        if y_raw is None:
            raise ValueError("train_y is None inside _fit_model")

        # Case 1: y is 1D → already fine
        if y_raw.ndim == 1:
            y = y_raw

        # Case 2: y is 2D
        elif y_raw.ndim == 2:
            # (N, 1) → squeeze to (N,)
            if y_raw.shape[1] == 1:
                y = y_raw[:, 0]

            # (N, M) with M > 1 → multi-output
            else:
                y = y_raw

        else:
            raise ValueError(f"Unexpected train_y shape: {y_raw.shape}")

        self.et_model = ExtraTreesRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            n_jobs=self.n_jobs,
            bootstrap=False,
            random_state=self.random_state,
        )
        self.et_model.fit(X, y)
        
        self.train_leaf_ids = np.stack(
            [tree.apply(self.train_x) for tree in self.et_model.estimators_],
            axis=0
        )

    def _predict_mean_var(self, X_unit):
        """
        Computes:
        - mean prediction
        - variance across trees
        - tree-proximity discrepancy
        """

        T = len(self.et_model.estimators_)
        B = X_unit.shape[0]

        # ------------------------------------------------------------
        # 1. Per-tree predictions (same as before)
        # ------------------------------------------------------------
        all_tree_preds = np.stack(
            [tree.predict(X_unit) for tree in self.et_model.estimators_],
            axis=0
        )  # (T, B)

        mean_pred = all_tree_preds.mean(axis=0)
        var_pred = all_tree_preds.var(axis=0)
        return mean_pred, var_pred
    
    # ------------------------------------------------------------
    # PREDICTION + UNCERTAINTY + DISCREPANCY
    # ------------------------------------------------------------
    def _predict_mean_var_discrepancy(self, X_unit):
        """
        Computes:
        - mean prediction
        - variance across trees
        - tree-proximity discrepancy
        """

        T = len(self.et_model.estimators_)
        B = X_unit.shape[0]

        # ------------------------------------------------------------
        # 1. Per-tree predictions (same as before)
        # ------------------------------------------------------------
        all_tree_preds = np.stack(
            [tree.predict(X_unit) for tree in self.et_model.estimators_],
            axis=0
        )  # (T, B)

        mean_pred = all_tree_preds.mean(axis=0)
        var_pred = all_tree_preds.var(axis=0)

        # ------------------------------------------------------------
        # 2. Precompute training leaf IDs once per model fit
        # ------------------------------------------------------------
        if not hasattr(self, "train_leaf_ids"):
            self.train_leaf_ids = np.stack(
                [tree.apply(self.train_x) for tree in self.et_model.estimators_],
                axis=0
            )  # (T, N_train)

        # ------------------------------------------------------------
        # 3. Compute leaf IDs for candidate points
        # ------------------------------------------------------------
        cand_leaf_ids = np.stack(
            [tree.apply(X_unit) for tree in self.et_model.estimators_],
            axis=0
        )  # (T, B)

        # ------------------------------------------------------------
        # 4. Compute proximity: fraction of trees where x and x_i share leaf
        # ------------------------------------------------------------
        # matches[t, b, i] = 1 if leaf_t(x_b) == leaf_t(x_i)
        matches = (cand_leaf_ids[:, :, None] == self.train_leaf_ids[:, None, :])
        # shape (T, B, N_train)

        # Sum over trees → (B, N_train)
        proximity = matches.sum(axis=0) / T

        # Max proximity per candidate
        max_prox = proximity.max(axis=1)

        # Discrepancy = 1 - max proximity
        discrepancy = 1.0 - max_prox

        return mean_pred, var_pred, discrepancy

    # ------------------------------------------------------------
    # ACQUISITION SCORE
    # ------------------------------------------------------------
    def _compute_acquisition_unchunked(self, X_unit):
        if self.acquisition_mode == 'var_dis':
            _, var_pred, discrepancy = self._predict_mean_var_discrepancy(X_unit)
            return discrepancy + self.acq_beta * np.sqrt(var_pred)
        if self.acquisition_mode == 'var':
            _, var_pred, discrepancy = self._predict_mean_var_discrepancy(X_unit)
            return var_pred
        if self.acquisition_mode == 'rand':
            return self.rng.random(len(X_unit))

    # ------------------------------------------------------------
    # OPTIONAL: CV TUNING FOR N_ESTIMATORS
    # ------------------------------------------------------------
    def tune_n_estimators(self, min_trees=50, max_trees=800, step_factor=2.0):
        best_rmse = None
        best_n = None

        X = self.train_x
        y = self.train_y.squeeze(-1)

        n = int(min_trees)
        while n <= max_trees:
            model = ExtraTreesRegressor(
                n_estimators=n,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                n_jobs=self.n_jobs,
                bootstrap=False,
                random_state=self.random_state,
            )

            kf = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
            rmses = []
            for tr, va in kf.split(X):
                model.fit(X[tr], y[tr])
                pred = model.predict(X[va])
                rmses.append(np.sqrt(mean_squared_error(y[va], pred)))

            rmse = float(np.mean(rmses))
            if best_rmse is None or rmse < best_rmse:
                best_rmse = rmse
                best_n = n

            n = int(n * step_factor)

        self.n_estimators = best_n
        return {"best_n_estimators": best_n, "best_rmse": best_rmse}

    # ------------------------------------------------------------
    # FOLD FITTING (for K-fold CV)
    # ------------------------------------------------------------
    def _fit_model_fold(self, X_tr, Y_tr):
        """
        Fit a fresh ExtraTrees model on arbitrary training data.
        Used only for K-fold CV.
        """

        # Flatten Y_tr if multi-output is present
        if Y_tr.ndim == 2 and Y_tr.shape[1] == 1:
            Y_tr = Y_tr.squeeze(-1)

        self.et_model_fold = ExtraTreesRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            n_jobs=self.n_jobs,
            bootstrap=False,
            random_state=self.random_state,
        )

        self.et_model_fold.fit(X_tr, Y_tr)

    # ------------------------------------------------------------
    # FOLD PREDICTION (for K-fold CV)
    # ------------------------------------------------------------
    def _predict_fold(self, X_val):
        """
        Predict using the fold-specific model.
        Returns ONLY predictions (no variance, no discrepancy).
        """

        # ExtraTreesRegressor returns shape (N,) or (N, M)
        pred = self.et_model_fold.predict(X_val)

        # Ensure 2D output for consistency with parent sampler
        if pred.ndim == 1:
            pred = pred.reshape(-1, 1)

        return pred

