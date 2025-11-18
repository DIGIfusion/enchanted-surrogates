import os
import math
import pickle
import warnings
import time
from scipy.special import erf
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import GPy
import time
from enchanted_surrogates.samplers.base_sampler import Sampler
from enchanted_surrogates.utils.precise_imports import import_sampler
from enchanted_surrogates.utils.timeout import run_with_timeout, FunctionTimeoutError, FunctionExecutionError
from enchanted_surrogates.utils.print_stats_table import print_stats_table

# This one has some numerical fixes

# ---------------------------
# Helper functions: RBF 1D integrals
# ---------------------------
def gaussian_1d_integral_between(xi, lengthscale, a, b):
    s = lengthscale
    coeff = math.sqrt(math.pi / 2.0) * s
    return coeff * (erf((b - xi) / (math.sqrt(2) * s)) - erf((a - xi) / (math.sqrt(2) * s)))

def gaussian_1d_double_integral(xi, xj, lengthscale, a, b):
    s = lengthscale
    pref = math.exp(- (xi - xj) ** 2 / (4.0 * s ** 2))
    s_eff = s / math.sqrt(2.0)
    coeff = math.sqrt(math.pi / 2.0) * s_eff
    mu = 0.5 * (xi + xj)
    return pref * coeff * (erf((b - mu) / (math.sqrt(2) * s_eff)) - erf((a - mu) / (math.sqrt(2) * s_eff)))

def rbf_kernel_product_integral_1d_vector(Xi, lengthscale, a, b):
    return np.array([gaussian_1d_integral_between(xi, lengthscale, a, b) for xi in Xi])

def rbf_kernel_product_double_integral_1d_matrix(Xi, Xj, lengthscale, a, b):
    n_i = Xi.shape[0]
    n_j = Xj.shape[0]
    M = np.empty((n_i, n_j), dtype=float)
    for i in range(n_i):
        for j in range(n_j):
            M[i, j] = gaussian_1d_double_integral(Xi[i], Xj[j], lengthscale, a, b)
    return M

# ---------------------------
# Sampler
# ---------------------------
class GpyAnalyticSobolSampler(Sampler):
    def __init__(self, *args, **kwargs):
        # configuration
        self.parameters = kwargs.get('parameters')
        self.bounds = kwargs.get('bounds')
        if not self.parameters and 'sub_sampler_config' in kwargs:
            self.parameters = kwargs['sub_sampler_config'].get('parameters')
        if not self.bounds and 'sub_sampler_config' in kwargs:
            self.bounds = kwargs['sub_sampler_config'].get('bounds')
        if not self.parameters or not self.bounds:
            raise ValueError("parameters and bounds must be provided")
        if len(self.parameters) != len(self.bounds):
            raise ValueError('The number of bounds and parameters must match.')

        self.acquisition_mode = kwargs.get("acquisition_mode", "variance")
        self.alpha = kwargs.get("alpha", 0.5)
        self.blend_string = kwargs.get("blend_string", None)
        self.chunk_size = kwargs.get("chunk_size", 5000)
        
        self.sampling_strategy = kwargs.get('sampling_strategy', 'random')
        self.n_ensembles = kwargs.get('n_ensembles', 1)
        self.batch_size = kwargs.get('batch_size', None) or 2
        self.initial_batch_size = kwargs.get('initial_batch_size', self.batch_size)
        self.initial_pool_samples_strategy = kwargs.get('initial_pool_samples_strategy', 'random')
        self.seed = kwargs.get('seed', 42)
        self.base_run_dir = kwargs.get('base_run_dir')
        self.num_repeats = kwargs.get('num_repeats', 1)
        self.include_index = kwargs.get('include_index', False)
        self.batch_number = 0
        self.submitted = 0
        self.custom_submitted = 0
        self.budget = kwargs.get('budget', None) or self.batch_size

        # numerical stabilization options
        self.use_log_products = kwargs.get('use_log_products', True)  # helps for large D
        self.small_positive_floor = kwargs.get('small_positive_floor', 1e-15)

        # optional sub-sampler (for initial or alternative sampling)
        self.sub_sampler_config = kwargs.get('sub_sampler_config', None)
        if self.sub_sampler_config:
            if 'parameters' not in self.sub_sampler_config:
                self.sub_sampler_config['parameters'] = self.parameters
            if 'bounds' not in self.sub_sampler_config:
                self.sub_sampler_config['bounds'] = self.bounds
            self.sub_sampler = import_sampler(self.sub_sampler_config['type'], self.sub_sampler_config)
        else:
            self.sub_sampler = None

        # pool sampler config: provides pool of unlabeled candidate points
        self.pool_sampler_config = kwargs.get('pool_sampler_config', None)
        self.pool_sampler = import_sampler(self.pool_sampler_config['type'], self.pool_sampler_config) if self.pool_sampler_config else None
        self.initial_pool_size = kwargs.get('initial_pool_size', 5000)
        self.pool = None
        self._init_pool()

        # training storage
        self.train = {}  # map tuple(x) -> y

        # GPy model (global) and cached hyperparams / solver
        self.gp_model = None
        self.kernel_variance = None
        self.lengthscales = None
        self.noise_variance = None
        self._K_cholesky = None
        self._solve_K = None

        # timeouts and misc
        self.write_batch_info_timeout = kwargs.get('write_batch_info_timeout', 5*60)
        self.do_write_batch_info = kwargs.get('do_write_batch_info', True)
        self.num_samples_at_last_write = 0
        self.write_batch_info_every_x_samples = kwargs.get('write_batch_info_every_x_samples',1)
        
        self.optimize_global = kwargs.get('optimize_global', True)  # optimize global hyperparams by default

    def split_integer(self, total, n):
        q, r = divmod(total, n)
        arr = np.full(n, q, dtype=int)
        arr[:r] += 1
        return arr

    # ---------------------------
    # Pool management
    # ---------------------------
    def _init_pool(self):
        rng = np.random.RandomState(self.seed)
        if self.pool_sampler_config:
            pool_sampler = import_sampler(self.pool_sampler_config['type'], self.pool_sampler_config)
            collected = []
            while len(collected) < self.initial_pool_size:
                pts = pool_sampler.get_next_samples()
                if not pts:
                    break
                for p in pts:
                    collected.append([p[param] for param in self.parameters])
                    if len(collected) >= self.initial_pool_size:
                        break
            if len(collected) == 0:
                raise ValueError('THE POOL SAMPLER DID NOT RETURN ANY SAMPLES')
            else:
                self.pool = np.array(collected, dtype=float)
        else:
            self.pool = rng.uniform(low=[b[0] for b in self.bounds],
                                    high=[b[1] for b in self.bounds],
                                    size=(self.initial_pool_size, len(self.bounds)))

    def _ensure_pool_size(self, min_size):
        if self.pool is None:
            self._init_pool()
            return
        if len(self.pool) >= min_size:
            return
        if self.pool_sampler:
            collected = []
            while len(collected) + len(self.pool) < min_size:
                pts = self.pool_sampler.get_next_samples()
                if not pts:
                    break
                for p in pts:
                    collected.append([p[param] for param in self.parameters])
            if collected:
                self.pool = np.vstack([self.pool, np.array(collected, dtype=float)])
            else:
                raise ValueError(f'POOL SAMPLER FAILED TO PROVIDE MORE SAMPLES AFTER THE POOL DROPPED BELOW A MINIMUM SIZE: {min_size}')

    # ---------------------------
    # Initial samples
    # ---------------------------
    def get_initial_samples(self, *args, **kwargs):
        if self.sub_sampler is not None:
            samples = self.sub_sampler.get_next_samples() * self.num_repeats
            if samples:
                self.batch_number += 1
                self.submitted += len(samples)
                self.custom_submitted += len(samples)
                self._remove_from_pool(samples)
                if self.include_index:
                    samples = [
                        {**samp, 'index': ind} for samp, ind in zip(samples, range(len(samples)))]
                return samples
            else: return None

        if self.pool is None or len(self.pool) == 0:
            self._init_pool()
        
        if self.initial_pool_samples_strategy == 'random':
            rng = np.random.RandomState(self.seed)
            n = min(self.initial_batch_size, len(self.pool))
            idxs = rng.choice(len(self.pool), size=n, replace=False)
            chosen = self.pool[idxs]
            self.pool = np.delete(self.pool, idxs, axis=0)
            samples = [{key: float(val) for key, val in zip(self.parameters, row)} for row in chosen] * self.num_repeats
        elif self.initial_pool_samples_strategy == 'first':
            n = min(self.initial_batch_size, len(self.pool))
            chosen = self.pool[:n]
            self.pool = self.pool[n:]
            samples = [{key: float(val) for key, val in zip(self.parameters, row)} for row in chosen] * self.num_repeats

        if self.include_index:
            samples = [
                {**samp, 'index': ind} for samp, ind in zip(samples, range(len(samples)))]

        self.batch_number += 1
        self.submitted += len(samples)
        self.custom_submitted += len(samples)
        return samples

    def _remove_from_pool(self, samples):
        if self.pool is None or len(self.pool) == 0:
            return
        vecs = np.array([[s[param] for param in self.parameters] for s in samples], dtype=float)
        to_delete = []
        for v in vecs:
            matches = np.all(np.isclose(self.pool, v, atol=1e-12, rtol=0.0), axis=1)
            idxs = np.where(matches)[0]
            if idxs.size > 0:
                to_delete.append(idxs[0])
        if to_delete:
            self.pool = np.delete(self.pool, to_delete, axis=0)

    # ---------------------------
    # Main sampling loop
    # ---------------------------
    def get_next_samples(self, batch_dir=None, *args, **kwargs):
        if self.batch_number == 0:
            return self.get_initial_samples()

        if not self.base_run_dir:
            raise RuntimeError('base_run_dir must be set to retrieve training data.')

        prev_batch_dir = os.path.join(self.base_run_dir, f'batch_{self.batch_number-1}')
        new_data_df = pd.read_csv(os.path.join(prev_batch_dir, 'enchanted_dataset.csv'))
        output_col = [col for col in new_data_df.columns if 'output' in col]
        if len(output_col) != 1:
            raise RuntimeError('Exactly one output column required.')

        train_df = new_data_df[self.parameters + output_col]
        new_train = {
            tuple(row[col] for col in self.parameters): float(row[output_col[0]])
            for _, row in train_df.iterrows()
        }
        self.train = {**self.train, **new_train}

        X = np.array([list(k) for k in self.train.keys()], dtype=float)
        Y = np.array(list(self.train.values()), dtype=float).reshape(-1, 1)

        # ---------------------------
        # Global GP fit (optimize hyperparameters once)
        # ---------------------------
        input_dim = X.shape[1]
        kernel = GPy.kern.RBF(input_dim=input_dim, ARD=True)
        self.gp_model = GPy.models.GPRegression(X, Y, kernel)
        self.gp_model.Gaussian_noise.variance.constrain_positive()
        if self.optimize_global:
            try:
                self.gp_model.optimize(messages=False)
            except Exception:
                pass
    
        # Extract and cache hyperparameters
        try:
            self.kernel_variance = float(self.gp_model.kern.variance.values[0])
        except Exception:
            self.kernel_variance = float(self.gp_model.kern.variance)
        try:
            ls = self.gp_model.kern.lengthscale.values
        except Exception:
            ls = np.atleast_1d(self.gp_model.kern.lengthscale)
        self.lengthscales = np.array(ls, dtype=float).reshape(-1)
        try:
            self.noise_variance = float(self.gp_model.likelihood.variance.values[0])
        except Exception:
            self.noise_variance = float(self.gp_model.likelihood.variance)

        # cache K Cholesky and solve routine for UQ (with scale-aware jitter)
        Xn = X.shape[0]
        noise = max(self.noise_variance, 1e-12)
        jitter = max(1e-10, 1e-6 * max(self.kernel_variance, noise))
        K_full = self.gp_model.kern.K(X) + np.eye(Xn) * (noise + jitter)
        try:
            L = np.linalg.cholesky(K_full)
            self._K_cholesky = L
            def solve_K(vec):
                y_ = np.linalg.solve(L, vec)
                x_ = np.linalg.solve(L.T, y_)
                return x_
            self._solve_K = solve_K
        except np.linalg.LinAlgError:
            K_inv = np.linalg.pinv(K_full)
            self._K_cholesky = None
            self._solve_K = lambda vec: K_inv.dot(vec)

        if self.do_write_batch_info:
            start_wbi = time.time()
            previous_batch_dir = os.path.join(self.base_run_dir, f'batch_{self.batch_number-1}')
            if self.custom_submitted - self.num_samples_at_last_write >= self.write_batch_info_every_x_samples or self.batch_number in [0,1,2,3]:
                self.write_batch_info(previous_batch_dir)
                self.num_samples_at_last_write = self.custom_submitted
                try:
                    with open(os.path.join(previous_batch_dir, 'gpy_model.pkl'), 'wb') as f:
                        pickle.dump(self.gp_model, f)
                except Exception:
                    pass
            end_wbi = time.time()
            print('WRITE BATCH INFO TOOK:', (end_wbi - start_wbi)/60, 'min')

        # ---------------------------
        # Ensure pool
        # ---------------------------
        desired_pool_min = kwargs.get('desired_pool_min', max(1000, 5 * self.batch_size))
        self._ensure_pool_size(desired_pool_min)
        if self.pool is None or len(self.pool) == 0:
            rng = np.random.RandomState(self.seed + self.batch_number)
            self.pool = rng.uniform(low=[b[0] for b in self.bounds],
                                    high=[b[1] for b in self.bounds],
                                    size=(self.initial_pool_size, len(self.bounds)))

        print('SELECTING NEW SAMPLES FROM POOL')
        # ---------------------------
        # Select new samples
        # ---------------------------
        samples = []
        if self.n_ensembles == 1:
            print(f'GETTING GLOBAL SCORE MODE:{self.acquisition_mode}')
            score_pool_global = self._compute_acquisition(self.pool, mode=self.acquisition_mode, blend_string=self.blend_string)
            score_pool_global = score_pool_global.flatten()
            print(f'BATCH SIZE:{self.batch_size} USING GLOBAL SCORE.')
            idx = list(np.argsort(-score_pool_global)[:self.batch_size])
            chosen_points = self.pool[idx]
            samples = [{key: float(v) for key, v in zip(self.parameters, row)} for row in chosen_points]
            self.pool = np.delete(self.pool, idx, axis=0)
        else:
            print(f'SPLITTING DATA INTO FOLDS')
            n_folds = min(self.n_ensembles, len(self.train))
            samples_per_fold = self.split_integer(self.batch_size, self.n_ensembles)
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=self.seed + self.batch_number)
            X_all = np.array([list(k) for k in self.train.keys()], dtype=float)
            Y_all = np.array(list(self.train.values()), dtype=float).reshape(-1, 1)

            chosen_indices = []
            for i, (train_idx, _) in enumerate(kf.split(X_all)):
                print(f'CALCULATING ACQUISITON FUNCTION FOR FOLD {i+1}')
                X_fold = X_all[train_idx]
                Y_fold = Y_all[train_idx]
                kernel_fold = GPy.kern.RBF(input_dim=input_dim, ARD=True)
                try:
                    kernel_fold.variance = self.kernel_variance
                    kernel_fold.lengthscale = self.lengthscales.copy()
                    kernel_fold.variance.fix(self.kernel_variance)
                    kernel_fold.lengthscale.fix(self.lengthscales.copy())
                except Exception:
                    try:
                        kernel_fold.variance = self.kernel_variance
                    except Exception:
                        pass

                model_fold = GPy.models.GPRegression(X_fold, Y_fold, kernel_fold)
                try:
                    model_fold.likelihood.variance = self.noise_variance
                    model_fold.Gaussian_noise.variance.fix(self.noise_variance)
                except Exception:
                    pass

                try:
                    model_fold.optimize_restarts(num_restarts=0, messages=False)
                except Exception:
                    pass

                scores = self._compute_acquisition(self.pool, mode=self.acquisition_mode, blend_string=self.blend_string, model=model_fold)
                idx_f = list(np.argsort(-scores)[:samples_per_fold[i]])
                chosen_indices.extend(idx_f)

            seen = set()
            unique_idxs = []
            for idx in chosen_indices:
                if idx not in seen:
                    seen.add(idx)
                    unique_idxs.append(idx)

            if len(unique_idxs) < self.batch_size:
                print('THE GPR ENSEMBLE SELECTED SOME OF THE SAME POINTS. USING GLOBAL SCORE TO GET MORE POINTS')
                print(f'GETTING GLOBAL SCORE MODE:{self.acquisition_mode}')
                score_pool_global = self._compute_acquisition(self.pool, mode=self.acquisition_mode, blend_string=self.blend_string)
                score_pool_global = score_pool_global.flatten()
                sorted_idx = list(np.argsort(-score_pool_global))
                for idx in sorted_idx:
                    if idx not in seen:
                        unique_idxs.append(idx)
                    if len(unique_idxs) >= self.batch_size:
                        break

            chosen_indices_final = unique_idxs[:self.batch_size]
            chosen_points = self.pool[chosen_indices_final]
            samples = [{key: float(v) for key, v in zip(self.parameters, row)} for row in chosen_points]
            self.pool = np.delete(self.pool, chosen_indices_final, axis=0)

        self.batch_number += 1
        if self.custom_submitted >= self.budget:
            return None
        if samples is not None:
            self.custom_submitted += len(samples)
        samples = samples * self.num_repeats
        if self.include_index:
            samples = [
                {**samp, 'index': ind} for samp, ind in zip(samples, range(len(samples)))]

        return samples

    # ---------------------------
    # Predictor
    # ---------------------------
    def surrogate_predict(self, samples):
        if self.gp_model is None:
            if len(self.train) == 0:
                raise RuntimeError('No training data to build surrogate for prediction.')
            X = np.array([list(k) for k in self.train.keys()], dtype=float)
            Y = np.array(list(self.train.values()), dtype=float).reshape(-1, 1)
            self.gp_model = GPy.models.GPRegression(X, Y, GPy.kern.RBF(input_dim=X.shape[1], ARD=True))
            try:
                self.gp_model.optimize(messages=False)
            except Exception:
                pass
        samples = np.asarray(samples, dtype=float)
        ypred, _ = self.gp_model.predict(samples)
        return ypred.flatten()

    # ---------------------------
    # Integrals and analytic UQ (predictor-only with normalization)
    # ---------------------------
    def _integral_k_over_domain(self, X_train):
        n = X_train.shape[0]
        D = X_train.shape[1]
        I = np.ones(n, dtype=float)
        for d in range(D):
            a, b = self.bounds[d]
            ls = self.lengthscales[d]
            Xi_d = X_train[:, d]
            I_d = rbf_kernel_product_integral_1d_vector(Xi_d, ls, a, b) / (b - a)  # normalize by width
            I *= I_d
            # diagnostics per dimension
            print(f"[INT-K] dim {d}: I_d(min,max)=({I_d.min():.3e},{I_d.max():.3e}), width={b-a:.3e}, ls={ls:.3e}")
        I *= self.kernel_variance
        print(f"[INT-K] kernel_variance={self.kernel_variance:.3e}, I(min,max)=({I.min():.3e},{I.max():.3e})")
        return I

    def _integral_kk_over_domain(self, X_train):
        n = X_train.shape[0]
        D = X_train.shape[1]
        C = np.ones((n, n), dtype=float)
        for d in range(D):
            a, b = self.bounds[d]
            ls = self.lengthscales[d]
            Xi_d = X_train[:, d]
            C_d = rbf_kernel_product_double_integral_1d_matrix(Xi_d, Xi_d, ls, a, b) / (b - a)  # normalize by width
            C *= C_d
            print(f"[INT-KK] dim {d}: C_d(min,max)=({C_d.min():.3e},{C_d.max():.3e}), width={b-a:.3e}, ls={ls:.3e}")
        C *= (self.kernel_variance ** 2)
        print(f"[INT-KK] kernel_variance^2={(self.kernel_variance**2):.3e}, C(min,max)=({C.min():.3e},{C.max():.3e})")
        return C

    def uq_analysis(self):
        if len(self.train) == 0:
            raise RuntimeError('No training data available for UQ analysis.')

        X = np.array([list(k) for k in self.train.keys()], dtype=float)
        y = np.array(list(self.train.values()), dtype=float).reshape(-1, 1)
        n, D = X.shape
        widths = np.array([b[1] - b[0] for b in self.bounds], dtype=float)
        vol = float(np.prod(widths))

        print(f"[UQ] n={n}, D={D}, bounds widths={widths}, vol={vol:.3e}")
        print(f"[UQ] lengthscales={self.lengthscales}, kernel_var={self.kernel_variance:.3e}, noise_var={self.noise_variance:.3e}")

        # integrals
        I = self._integral_k_over_domain(X)

        # Build stabilized K and its inverse via solves
        K_base = self.gp_model.kern.K(X)
        noise = max(self.noise_variance, 1e-12)
        jitter = max(1e-10, 1e-6 * max(self.kernel_variance, noise))
        K = K_base + np.eye(n) * (noise + jitter)

        if self._K_cholesky is None:
            try:
                L = np.linalg.cholesky(K)
                self._K_cholesky = L
                def solve_K(vec):
                    y_ = np.linalg.solve(L, vec)
                    x_ = np.linalg.solve(L.T, y_)
                    return x_
                self._solve_K = solve_K
            except np.linalg.LinAlgError:
                K_inv_fallback = np.linalg.pinv(K)
                self._solve_K = lambda vec: K_inv_fallback.dot(vec)

        # compute K_inv via solves with identity (more stable than pinv chaining)
        I_eye = np.eye(n)
        K_inv_cols = [self._solve_K(I_eye[:, i:i+1]).flatten() for i in range(n)]
        K_inv = np.column_stack(K_inv_cols)
        K_inv_y = self._solve_K(y)

        integral_m = I.reshape(1, -1).dot(K_inv_y)  # scalar
        mu = float(integral_m / vol)
        print(f"[UQ] mu={mu:.6e}")

        # Total predictor variance
        C = self._integral_kk_over_domain(X)
        A = K_inv.dot(C).dot(K_inv)
        var_pred_raw = float((y.T.dot(A).dot(y)) / vol - mu ** 2)
        var_pred = max(var_pred_raw, 0.0)
        print(f"[UQ] var_pred_raw={var_pred_raw:.6e}, var_pred(clamped)={var_pred:.6e}")

        # first-order Sobol (predictor-only)
        sobol_first = {}
        for j in range(D):
            # product over other dims with normalization; optionally in log-space
            if self.use_log_products:
                logs = np.zeros(n, dtype=float)
                for d in range(D):
                    if d == j:
                        continue
                    a, b = self.bounds[d]
                    ls = self.lengthscales[d]
                    Xi_d = X[:, d]
                    I_d = rbf_kernel_product_integral_1d_vector(Xi_d, ls, a, b) / (b - a)
                    logs += np.log(I_d + 1e-300)  # avoid log(0)
                    print(f"[UQ:j={j}] I_d(dim {d}) min/max=({I_d.min():.3e},{I_d.max():.3e})")
                prod_other = np.exp(logs)
            else:
                prod_other = np.ones(n, dtype=float)
                for d in range(D):
                    if d == j:
                        continue
                    a, b = self.bounds[d]
                    ls = self.lengthscales[d]
                    Xi_d = X[:, d]
                    I_d = rbf_kernel_product_integral_1d_vector(Xi_d, ls, a, b) / (b - a)
                    prod_other *= I_d
                    print(f"[UQ:j={j}] I_d(dim {d}) min/max=({I_d.min():.3e},{I_d.max():.3e})")

            print(f"[UQ:j={j}] prod_other min/max=({prod_other.min():.3e},{prod_other.max():.3e})")

            a_j, b_j = self.bounds[j]
            ls_j = self.lengthscales[j]
            Xi_j = X[:, j]
            Mj = rbf_kernel_product_double_integral_1d_matrix(Xi_j, Xi_j, ls_j, a_j, b_j) / (b_j - a_j)
            print(f"[UQ:j={j}] Mj min/max=({Mj.min():.3e},{Mj.max():.3e})")

            outer_prod = np.outer(prod_other, prod_other)
            B = (self.kernel_variance ** 2) * outer_prod * Mj

            num_raw = float((y.T.dot(K_inv.dot(B).dot(K_inv)).dot(y)) / vol - mu ** 2)
            num = num_raw if num_raw >= 0 else 0.0
            if 0 < num < self.small_positive_floor:
                num = self.small_positive_floor

            S_j = float(num / var_pred) if var_pred > 0 else 0.0
            sob_key = self.parameters[j] + '_sobolF'

            print(f"[UQ:j={j}] num_raw={num_raw:.6e}, num(clamped)={num:.6e}, S_j={S_j:.6e}")
            sobol_first[sob_key] = S_j

        batch_info = {
            'num_samples': [n],
            'mean': [mu],
            'std': [math.sqrt(var_pred)]
        }
        batch_info.update({k: [v] for k, v in sobol_first.items()})
        return batch_info

    # ---------------------------
    # Write batch info (with timeout)
    # ---------------------------
    def write_batch_info(self, batch_dir):
        print('WRITING BATCH INFO')
        try:
            self._write_batch_info_inner(batch_dir=batch_dir)
        except FunctionTimeoutError:
            warnings.warn(f"write_batch_info timed out after {self.write_batch_info_timeout} seconds; skipping batch info write for batch {self.batch_number-1}", UserWarning)
        except FunctionExecutionError as exc:
            warnings.warn(f"write_batch_info raised an exception: {exc}; skipping batch info write for batch {self.batch_number-1}", UserWarning)

    def _write_batch_info_inner(self, batch_dir):
        batch_info = self.uq_analysis()
        df = pd.DataFrame({k: v for k, v in batch_info.items()})
        print("[WRITE] batch_info head:\n", df.head().to_string(index=False))
        df.to_csv(os.path.join(batch_dir, 'batch_info.csv'), index=False)
        all_batch_info_path = os.path.join(os.path.dirname(batch_dir), 'batch_info.csv')
        if os.path.exists(all_batch_info_path):
            df.to_csv(all_batch_info_path, mode='a', header=False, index=False)
        else:
            df.to_csv(all_batch_info_path, mode='w', header=True, index=False)

    def _compute_acquisition(self, X_pool, mode="var", blend_string=None, model=None):
        start = time.time()
        if model is None:
            model = self.gp_model
        if mode == "var":
            mu, var = model.predict(X_pool)
            end = time.time()
            print('COMPUTING ACQUISITION TOOK:',(end-start)/60, 'min', f'MODE:{mode}')
            return var.flatten()

        elif mode == "gradVar":
            X_pool = np.atleast_2d(X_pool)
            dmu, _ = model.predictive_gradients(X_pool)
            grads = np.linalg.norm(dmu, axis=1).squeeze()
            end = time.time()
            print('COMPUTING ACQUISITION TOOK:', (end-start)/60, 'min', f'MODE:{mode}')
            return grads

        elif mode == "intVar":
            end = time.time()
            print('COMPUTING ACQUISITION TOOK:',(end-start)/60, 'min', f'MODE:{mode}')
            return self.integral_variance_reduction(X_pool)

        elif mode == "ensembleDisagreement":
            preds = []
            n_folds = min(5, max(2, len(self.train)))
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=self.seed + self.batch_number)
            X_all = np.array([list(k) for k in self.train.keys()], dtype=float)
            Y_all = np.array(list(self.train.values())).reshape(-1, 1)
            for train_idx, _ in kf.split(X_all):
                X_fold, Y_fold = X_all[train_idx], Y_all[train_idx]
                kernel_fold = GPy.kern.RBF(input_dim=X_fold.shape[1], ARD=True)
                kernel_fold.variance = self.kernel_variance
                kernel_fold.lengthscale = self.lengthscales.copy()
                model_fold = GPy.models.GPRegression(X_fold, Y_fold, kernel_fold)
                model_fold.likelihood.variance = self.noise_variance
                mu_f, _ = model_fold.predict(X_pool)
                preds.append(mu_f.flatten())
            preds = np.vstack(preds)
            end = time.time()
            print('COMPUTING ACQUISITION TOOK:',(end-start)/60, 'min', f'MODE:{mode}')
            return preds.var(axis=0)

        elif mode == "blend":
            if blend_string is None:
                raise ValueError("blend_string must be provided for blend mode")
            blend = self._parse_blend_string(blend_string)

            total = np.zeros(len(X_pool))
            for coeff, m in blend:
                if len(X_pool) > self.chunk_size:
                    scores = self._compute_acquisition_chunked(X_pool, mode=m, chunk_size=self.chunk_size)
                else:
                    scores = self._compute_acquisition(X_pool, mode=m)
                scores = (scores - scores.mean()) / (scores.std() + 1e-12)
                total += coeff * scores

            end = time.time()
            print('COMPUTING ACQUISITION TOOK:',(end-start)/60, 'min', f'MODE:{mode}')
            return total

        else:
            raise ValueError(f"Unknown acquisition mode: {mode}")

    def _compute_acquisition_chunked(self, X_pool, mode, chunk_size=5000):
        results = []
        for i in range(0, len(X_pool), chunk_size):
            block = X_pool[i:i+chunk_size]
            results.append(self._compute_acquisition(block, mode=mode))
        return np.concatenate(results)

    def _parse_blend_string(self, blend_string):
        parts = blend_string.split("_")
        blend = []
        for part in parts:
            coeff, mode = part.split("-", 1)
            blend.append((float(coeff), mode))
        return blend

    def integral_variance_reduction(self, X_pool):
        X_train = np.array([list(k) for k in self.train.keys()], dtype=float)
        n_train = X_train.shape[0]

        K = self.gp_model.kern.K(X_train) + np.eye(n_train) * max(self.noise_variance, 1e-12)
        K_inv = np.linalg.pinv(K)

        # normalized kernel mean features
        phi_train = self._integral_k_over_domain(X_train)    # shape (n_train,)
        phi_pool = self._integral_k_over_domain(X_pool)      # shape (n_pool,)

        K_cross = self.gp_model.kern.K(X_train, X_pool)      # shape (n_train, n_pool)
        K_self = np.diag(self.gp_model.kern.K(X_pool))       # shape (n_pool,)

        diff = phi_pool - K_cross.T.dot(K_inv).dot(phi_train)
        num = diff**2
        denom = K_self - np.sum(K_cross.T.dot(K_inv) * K_cross.T, axis=1)
        results = np.where(denom > 1e-12, num / denom, 0.0)
        return results

    # ---------------------------
    # Boilerplate
    # ---------------------------
    def brute_force_uq_analysis(self):
        raise NotImplementedError('Brute force UQ analysis not implemented in this sampler.')

    def register_future(self, future):
        return None

    def register_futures(self, futures):
        return None

if __name__ == '__main__':
    import sys
    import yaml
    from enchanted_surrogates.utils.get_batch_dirs import get_batch_dirs
    from enchanted_surrogates.utils.load_configuration import load_configuration
    from enchanted_surrogates.utils.precise_imports import import_sampler
    _, base_run_dir, write_every = sys.argv
    
    batch_dirs = get_batch_dirs(base_run_dir)
    
    listdir = os.listdir(base_run_dir)
    config_file_name = [name for name in listdir if '.yaml' in name]
    if len(config_file_name) > 1:
        raise FileNotFoundError('More than one .yaml file in base_run_dir, not sure which to use as config file')
    config_file_name = config_file_name[0]
    print('CONFIG FOUND:',os.path.join(base_run_dir, config_file_name))
    config = load_configuration(os.path.join(base_run_dir, config_file_name))
    
    print('debug sampler config', config.executor['sampler_config'])
    
    sampler_config = config.executor['sampler_config']
    sampler_config['base_run_dir'] = base_run_dir

    for i, batch_dir in enumerate(batch_dirs):
        if i==0 or i==1 or i%int(write_every)==0:
            gpy = GpyAnalyticSobolSampler(**sampler_config)
            gpy.write_batch_info(batch_dir)
