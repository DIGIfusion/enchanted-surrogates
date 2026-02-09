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
import shutil
import traceback

from enchanted_surrogates.samplers.base_sampler import Sampler
from enchanted_surrogates.utils.precise_imports import import_sampler
from enchanted_surrogates.utils.timeout import run_with_timeout, FunctionTimeoutError, FunctionExecutionError
from enchanted_surrogates.utils.print_stats_table import print_stats_table


# ---- analytic kernel integrals -------------------------------------------------

def gaussian_1d_integral_between(xi, lengthscale, a, b):
    s = lengthscale
    coeff = math.sqrt(math.pi / 2.0) * s
    # decompiler lost parentheses; restore standard form:
    # ∫ exp(-(x - xi)^2 / (2 s^2)) dx from a to b
    return coeff * (
        erf((b - xi) / (math.sqrt(2.0) * s)) -
        erf((a - xi) / (math.sqrt(2.0) * s))
    )


def gaussian_1d_double_integral(xi, xj, lengthscale, a, b):
    s = lengthscale
    # decompiler mangled this badly; this is the usual RBF product integral form:
    # prefactor exp(-(xi - xj)^2 / (4 s^2)), effective lengthscale s / sqrt(2)
    pref = math.exp(-((xi - xj) ** 2) / (4.0 * s ** 2))
    s_eff = s / math.sqrt(2.0)
    coeff = math.sqrt(math.pi / 2.0) * s_eff
    mu = 0.5 * (xi + xj)
    return pref * coeff * (
        erf((b - mu) / (math.sqrt(2.0) * s_eff)) -
        erf((a - mu) / (math.sqrt(2.0) * s_eff))
    )


def rbf_kernel_product_integral_1d_vector(Xi, lengthscale, a, b):
    return np.array(
        [gaussian_1d_integral_between(xi, lengthscale, a, b) for xi in Xi],
        dtype=float
    )


def rbf_kernel_product_double_integral_1d_matrix(Xi, Xj, lengthscale, a, b):
    n_i = Xi.shape[0]
    n_j = Xj.shape[0]
    M = np.empty((n_i, n_j), dtype=float)
    for i in range(n_i):
        for j in range(n_j):
            M[i, j] = gaussian_1d_double_integral(Xi[i], Xj[j], lengthscale, a, b)
    return M


# ---- main sampler --------------------------------------------------------------

class GpyAnalyticSobolSamplerOld(Sampler):
    def __init__(self, **kwargs):
        # required
        self.parameters = kwargs.get('parameters')
        self.bounds = kwargs.get('bounds')

        if self.parameters is None or self.bounds is None:
            raise ValueError('parameters and bounds must be provided')

        if len(self.parameters) != len(self.bounds):
            raise ValueError('The number of bounds and parameters must match.')

        self._lb = np.array([b[0] for b in self.bounds], dtype=float)
        self._ub = np.array([b[1] for b in self.bounds], dtype=float)
        self._range = self._ub - self._lb

        # config
        self.acquisition_mode = kwargs.get('acquisition_mode', 'variance')
        self.alpha = kwargs.get('alpha', 0.5)
        self.blend_string = kwargs.get('blend_string', None)
        self.chunk_size = kwargs.get('chunk_size', 5000)
        self.test_data_csv = kwargs.get('test_data_csv', None)
        self.test_data_name = kwargs.get('test_data_name', 'test')
        self.sampling_strategy = kwargs.get('sampling_strategy', 'random')
        self.n_ensembles = kwargs.get('n_ensembles', 1)
        self.batch_size = kwargs.get('batch_size', 2)
        self.initial_batch_size = kwargs.get('initial_batch_size', self.batch_size)
        self.initial_pool_samples_strategy = kwargs.get('initial_pool_samples_strategy', 'random')
        self.seed = kwargs.get('seed', 42)
        self.base_run_dir = kwargs.get('base_run_dir')
        self.num_repeats = kwargs.get('num_repeats', 1)
        self.include_index = kwargs.get('include_index', False)
        self.batch_number = 0
        self.submitted = 0
        self.custom_submitted = 0
        self.budget = kwargs.get('budget', self.batch_size)

        self.pool_csv_path = kwargs.get('pool_csv_path', None)


        # sub-sampler
        self.sub_sampler_config = kwargs.get('sub_sampler_config', None)
        if self.sub_sampler_config:
            self.sub_sampler_config.setdefault('parameters', self.parameters)
            self.sub_sampler_config.setdefault('bounds', self.bounds)
            self.sub_sampler = import_sampler(
                self.sub_sampler_config['type'],
                self.sub_sampler_config
            )
        else:
            self.sub_sampler = None

        # pool sampler
        self.pool_sampler_config = kwargs.get('pool_sampler_config', None)
        if self.pool_sampler_config:
            if self.pool_sampler_config.get('bounds') != [(0, 1)] * len(self.bounds):
                warnings.warn('Pool sampler bounds corrected to unit [0,1].')
                self.pool_sampler_config['bounds'] = [(0, 1)] * len(self.bounds)
            self.pool_sampler = import_sampler(
                self.pool_sampler_config['type'],
                self.pool_sampler_config
            )
        else:
            self.pool_sampler = None

        self.initial_pool_size = kwargs.get('initial_pool_size', 5000)
        self.pool = None
        self._init_pool()

        self.train = {}
        self.gp_model = None
        self.kernel_variance = None
        self.lengthscales = None
        self.noise_variance = None
        self._K_cholesky = None
        self._solve_K = None

        self.write_batch_info_timeout = kwargs.get('write_batch_info_timeout', 300)
        self.do_write_batch_info = kwargs.get('do_write_batch_info', True)
        self.num_samples_at_last_write = 0
        self.write_batch_info_every_x_samples = kwargs.get('write_batch_info_every_x_samples', 1)
        self.optimize_global = kwargs.get('optimize_global', True)

    # ---- helpers --------------------------------------------------------------

    def split_integer(self, total, n):
        q, r = divmod(total, n)
        arr = np.full(n, q, dtype=int)
        arr[:r] += 1
        return arr

    def to_unit(self, X):
        """Map real-bounds inputs to [0,1]."""
        X = np.asarray(X, dtype=float)
        return (X - self._lb) / self._range

    def from_unit(self, X_unit):
        """Map unit inputs back to real bounds."""
        X_unit = np.asarray(X_unit, dtype=float)
        return self._lb + X_unit * self._range

    # ---- pool management ------------------------------------------------------

    def _init_pool(self):
        rng = np.random.RandomState(self.seed)
        if self.pool_sampler_config:
            pool_sampler = import_sampler(
                self.pool_sampler_config['type'],
                self.pool_sampler_config
            )
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
                raise ValueError('The pool sampler did not return any samples.')

            self.pool = np.array(collected, dtype=float)
        
        if self.pool_csv_path is not None:
            df = pd.read_csv(self.pool_csv_path)
            self.pool = self.to_unit(df[self.parameters].to_numpy())
            output_col = [col for col in df.columns if 'output' in col]
            if len(output_col)>1:
                warnings.warn(f'MORE THAN ONE OUTPUT WAS FOUND IN THE POOL CSV: {output_col}. THE FIRST WILL BE TAKEN AS THE OUTPUT OF INTEREST. {output_col[0]}')
            output_col = output_col[0]
            self.pool_y = df[output_col].to_numpy() # unormalised

        else:
            self.pool = rng.uniform(
                low=0.0,
                high=1.0,
                size=(self.initial_pool_size, len(self.bounds))
            )

    def _ensure_pool_size(self, min_size):
        if self.pool is None:
            self._init_pool()
            return
        if len(self.pool) >= min_size:
            return
        if not self.pool_sampler_config:
            # nothing we can do
            return

        collected = []
        while len(self.pool) + len(collected) < min_size:
            pts = self.pool_sampler.get_next_samples()
            if not pts:
                break
            for p in pts:
                collected.append([p[param] for param in self.parameters])

        if collected:
            new = np.array(collected, dtype=float)
            self.pool = np.vstack([self.pool, new])

    # ---- sampling -------------------------------------------------------------

    def get_initial_samples(self):
        if self.sub_sampler is not None:
            samples = self.sub_sampler.get_next_samples()
            if not samples:
                return None
            samples = samples * self.num_repeats
            self.batch_number += 1
            self.submitted += len(samples)
            self.custom_submitted += len(samples)
            self._remove_from_pool(samples)
        else:
            if self.pool is None or len(self.pool) == 0:
                self._init_pool()

            if self.initial_pool_samples_strategy == 'random':
                rng = np.random.RandomState(self.seed)
                n = min(self.initial_batch_size, len(self.pool))
                idxs = rng.choice(len(self.pool), size=n, replace=False)
                chosen = self.pool[idxs]
                self.pool = np.delete(self.pool, idxs, axis=0)
                if self.pool_y is not None:
                    self.pool_y = np.delete(self.pool_y, idxs, axis=0)
            elif self.initial_pool_samples_strategy == 'first':
                n = min(self.initial_batch_size, len(self.pool))
                chosen = self.pool[:n]
                self.pool = self.pool[n:]
            else:
                raise ValueError(f"Unknown initial_pool_samples_strategy: {self.initial_pool_samples_strategy}")

            real_chosen = self.from_unit(chosen)
            samples = [
                {key: float(val) for key, val in zip(self.parameters, row)}
                for row in real_chosen
            ] * self.num_repeats

        if self.include_index:
            samples = [
                {**s, "index": i}
                for i, s in enumerate(samples)
            ]

        self.batch_number += 1
        self.submitted += len(samples)
        self.custom_submitted += len(samples)
        return samples

    def _remove_from_pool(self, samples):
        if self.pool is None or len(self.pool) == 0:
            return
        vecs = np.array(
            [[s[param] for param in self.parameters] for s in samples],
            dtype=float
        )
        vecs = self.to_unit(vecs)
        to_delete = []
        for v in vecs:
            matches = np.all(np.isclose(self.pool, v, atol=1e-12, rtol=0.0), axis=1)
            idxs = np.where(matches)[0]
            if idxs.size > 0:
                to_delete.append(idxs[0])
        if to_delete:
            self.pool = np.delete(self.pool, to_delete, axis=0)
            if self.pool_y is not None:
                self.pool_y = np.delete(self.pool_y, to_delete, axis=0)

    # ---- training data / GP ---------------------------------------------------

    def append_train_data(self, batch_dir=None):
        if batch_dir is None:
            new_data_df = pd.read_csv(os.path.join(self.base_run_dir, 'enchanted_dataset.csv'))
        else:
            new_data_df = pd.read_csv(os.path.join(batch_dir, 'enchanted_dataset.csv'))
        output_col = [col for col in new_data_df.columns if 'output' in col]
        if len(output_col) != 1:
            raise RuntimeError(f'Exactly one output column required in training data but found: {output_col}')
        train_df = new_data_df[self.parameters + output_col]
        new_train = {
            tuple(row[col] for col in self.parameters): float(row[output_col[0]])
            for _, row in train_df.iterrows()
        }
        # accumulate, don’t overwrite
        self.train.update(new_train)

    def _get_unitXY(self):
        if len(self.train) == 0:
            print('no training data, getting from base enchanted dataset')
            self.append_train_data()
        X_real = np.array([list(k) for k in self.train.keys()], dtype=float)
        Y = np.array(list(self.train.values()), dtype=float).reshape(-1, 1)
        X_unit = self.to_unit(X_real)
        return X_unit, Y

    def fit(self):
        X, Y = self._get_unitXY()
        input_dim = X.shape[1]
        kernel = GPy.kern.RBF(input_dim=input_dim, ARD=True)
        self.gp_model = GPy.models.GPRegression(X, Y, kernel)
        self.gp_model.Gaussian_noise.variance.constrain_positive()
        if self.optimize_global:
            try:
                self.gp_model.optimize(messages=False)
            except Exception as exc:
                print(
                    f"GLOBAL OPTIMIZE FAILED. ERROR: {exc}\nTRACEBACK:\n"
                    f"{traceback.format_exc()}"
                )

    # ---- main loop ------------------------------------------------------------

    def get_next_samples(self, batch_dir=None):
        if self.batch_number == 0:
            return self.get_initial_samples()

        if not self.base_run_dir:
            raise RuntimeError('base_run_dir must be set to retrieve training data.')

        prev_batch_dir = os.path.join(
            self.base_run_dir,
            f"batch_{self.batch_number - 1}"
        )
        self.append_train_data(prev_batch_dir)
        X, Y = self._get_unitXY()
        self.fit()
        self.cache_hypers()
        self.cache_K()

        if self.do_write_batch_info:
            start_wbi = time.time()
            previous_batch_dir = prev_batch_dir
            print(
                "debug is it writing every sample? "
                f"custom_submitted: {self.custom_submitted}, "
                f"num_samples_at_last_write: {self.num_samples_at_last_write}, "
                f"write_batch_info_every_x_samples: {self.write_batch_info_every_x_samples}, "
                f"batch_number {self.batch_number}"
            )
            if (
                self.custom_submitted - self.num_samples_at_last_write
                >= self.write_batch_info_every_x_samples
                #or self.batch_number in (0, 1, 2, 3)
            ):
                self.write_batch_info(previous_batch_dir)
                self.num_samples_at_last_write = self.custom_submitted
                try:
                    with open(os.path.join(previous_batch_dir, 'gpy_model.pkl'), 'wb') as f:
                        pickle.dump(self.gp_model, f)
                except Exception:
                    pass
            end_wbi = time.time()
            print('WRITE BATCH INFO TOOK:', (end_wbi - start_wbi) / 60.0, 'min')

        desired_pool_min = max(1000, 5 * self.batch_size)
        self._ensure_pool_size(desired_pool_min)
        print('SELECTING NEW SAMPLES FROM POOL')

        samples = []
        if self.n_ensembles == 1:
            print(f"GETTING GLOBAL SCORE MODE: {self.acquisition_mode}")
            score_pool_global = self._compute_acquisition(
                self.pool,
                mode=self.acquisition_mode,
                blend_string=self.blend_string,
                pool_y = self.pool_y
            ).flatten()
            print(f"BATCH SIZE: {self.batch_size} USING GLOBAL SCORE.")
            idx = list(np.argsort(-score_pool_global)[:self.batch_size])
            chosen_points = self.pool[idx]
            real_chosen_points = self.from_unit(chosen_points)
            samples = [
                {key: float(v) for key, v in zip(self.parameters, row)}
                for row in real_chosen_points
            ]
            self.pool = np.delete(self.pool, idx, axis=0)
            if self.pool_y is not None:
                self.pool_y = np.delete(self.pool_y, idx, axis=0)
        else:
            # ensemble logic kept close to decompiled version, but cleaned
            print('SPLITTING DATA INTO FOLDS')
            n_folds = min(self.n_ensembles, len(self.train))
            samples_per_fold = self.split_integer(self.batch_size, self.n_ensembles)
            kf = KFold(
                n_splits=n_folds,
                shuffle=True,
                random_state=self.seed + self.batch_number
            )
            X_all, Y_all = self._get_unitXY()
            input_dim = X_all.shape[1]
            chosen_indices = []
            for i, (train_idx, _) in enumerate(kf.split(X_all)):
                print(f"CALCULATING ACQUISITION FUNCTION FOR FOLD {i + 1}")
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

                scores = self._compute_acquisition(
                    self.pool,
                    mode=self.acquisition_mode,
                    blend_string=self.blend_string,
                    model=model_fold,
                    pool_y = self.pool_y
                )
                idx_f = list(np.argsort(-scores)[:samples_per_fold[i]])
                chosen_indices.extend(idx_f)

            seen = set()
            unique_idxs = []
            for idx in chosen_indices:
                if idx not in seen:
                    seen.add(idx)
                    unique_idxs.append(idx)

            if len(unique_idxs) < self.batch_size:
                print(
                    'THE GPR ENSEMBLE SELECTED SOME OF THE SAME POINTS. '
                    'USING GLOBAL SCORE TO GET MORE POINTS'
                )
                print(f"GETTING GLOBAL SCORE MODE: {self.acquisition_mode}")
                score_pool_global = self._compute_acquisition(
                    self.pool,
                    mode=self.acquisition_mode,
                    blend_string=self.blend_string,
                    pool_y = self.pool_y
                ).flatten()
                sorted_idx = list(np.argsort(-score_pool_global))
                for idx in sorted_idx:
                    if idx not in seen:
                        unique_idxs.append(idx)
                    if len(unique_idxs) >= self.batch_size:
                        break

            chosen_indices_final = unique_idxs[:self.batch_size]
            chosen_points = self.pool[chosen_indices_final]
            real_chosen_points = self.from_unit(chosen_points)
            samples = [
                {key: float(v) for key, v in zip(self.parameters, row)}
                for row in real_chosen_points
            ]
            self.pool = np.delete(self.pool, chosen_indices_final, axis=0)
            if self.pool_y is not None:
                self.pool_y = np.delete(self.pool_y, chosen_indices_final, axis=0)

        self.batch_number += 1
        if self.custom_submitted >= self.budget:
            return None
        if samples is not None:
            self.custom_submitted += len(samples)
        return samples

    # ---- GP caches ------------------------------------------------------------

    def cache_hypers(self):
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

    def cache_K(self):
        X, Y = self._get_unitXY()
        K_full = self.gp_model.kern.K(X) + np.eye(X.shape[0]) * max(self.noise_variance, 1e-8)
        jitter = 1e-8
        K_full_j = K_full + np.eye(K_full.shape[0]) * jitter
        try:
            L = np.linalg.cholesky(K_full_j)
            self._K_cholesky = L

            def solve_K(vec):
                y = np.linalg.solve(L, vec)
                x = np.linalg.solve(L.T, y)
                return x

            self._solve_K = solve_K
        except np.linalg.LinAlgError:
            K_inv = np.linalg.pinv(K_full_j)
            self._K_cholesky = None
            self._solve_K = lambda vec: K_inv.dot(vec)

    # ---- surrogate + integrals ------------------------------------------------

    def surrogate_predict(self, samples):
        """Accept samples in real bounds, scale to unit, predict with GP."""
        
        if self.gp_model is None:
            self.fit()

        samples_unit = self.to_unit(samples)
        ypred, _ = self.gp_model.predict(samples_unit)
        return ypred.flatten()

    def _integral_k_over_domain(self, X_unit):
        n, D = X_unit.shape
        I = np.ones(n, dtype=float)
        for d in range(D):
            a, b = 0.0, 1.0
            ls = self.lengthscales[d]
            Xi_d = X_unit[:, d]
            I_d = rbf_kernel_product_integral_1d_vector(Xi_d, ls, a, b)
            I *= I_d
        I *= self.kernel_variance
        return I

    def _integral_kk_over_domain(self, X_unit):
        n, D = X_unit.shape
        C = np.ones((n, n), dtype=float)
        for d in range(D):
            a, b = 0.0, 1.0
            ls = self.lengthscales[d]
            Xi_d = X_unit[:, d]
            C_d = rbf_kernel_product_double_integral_1d_matrix(Xi_d, Xi_d, ls, a, b)
            C *= C_d
        C *= self.kernel_variance ** 2
        return C

    def uq_analysis(self):
        print("\n\n ================================== \n")
        if len(self.train) == 0:
            raise RuntimeError('No training data available for UQ analysis.')

        X, y = self._get_unitXY()
        n, D = X.shape
        vol = 1.0

        I = self._integral_k_over_domain(X)
        K = self.gp_model.kern.K(X) + np.eye(n) * max(self.noise_variance, 1e-8)

        try:
            K_inv_y = self._solve_K(y)
        except Exception:
            print('cholesky method failed, falling back to pinv')
            K_inv = np.linalg.pinv(K)
            K_inv_y = K_inv.dot(y)

        integral_m = I.reshape(1, -1).dot(K_inv_y)
        mu = float(integral_m / vol)

        C = self._integral_kk_over_domain(X)
        try:
            if self._K_cholesky is not None:
                I_eye = np.eye(n)
                K_inv_cols = [
                    self._solve_K(I_eye[:, i:i + 1]).flatten()
                    for i in range(n)
                ]
                K_inv = np.column_stack(K_inv_cols)
            else:
                K_inv = np.linalg.pinv(K)

            A = K_inv.dot(C).dot(K_inv)
            var_pred = float(y.T.dot(A).dot(y) / vol - mu ** 2)
        except Exception:
            var_pred = 0.0

        var_pred = max(var_pred, 0.0)

        sobol_first = {}
        for j in range(D):
            prod_other = np.ones(n, dtype=float)
            for d in range(D):
                if d == j:
                    continue
                ls = self.lengthscales[d]
                Xi_d = X[:, d]
                I_d = rbf_kernel_product_integral_1d_vector(Xi_d, ls, 0.0, 1.0)
                prod_other *= I_d

            ls_j = self.lengthscales[j]
            Xi_j = X[:, j]
            Mj = rbf_kernel_product_double_integral_1d_matrix(Xi_j, Xi_j, ls_j, 0.0, 1.0)
            outer_prod = np.outer(prod_other, prod_other)
            B = self.kernel_variance ** 2 * outer_prod * Mj

            try:
                if K_inv is None:
                    K_inv = np.linalg.pinv(K)
                num = float(y.T.dot(K_inv.dot(B).dot(K_inv)).dot(y) / vol - mu ** 2)
            except Exception as exc:
                print('EXCEPTION WHEN GETTING num:', exc)
                print('TRACEBACK: \n', traceback.format_exc())
                num = 0.0

            num = max(num, 0.0)
            sobol_first[self.parameters[j] + '_sobolF'] = float(num / var_pred) if var_pred > 0 else 0.0

        batch_info = {
            'num_samples': [n],
            'mean': [mu],
            'std': [math.sqrt(var_pred)],
        }
        batch_info.update({k: [v] for k, v in sobol_first.items()})
        return batch_info

    # ---- batch info / regression ---------------------------------------------

    def write_batch_info(self, batch_dir):
        print('WRITING BATCH INFO')
        try:
            self._write_batch_info_inner(batch_dir=batch_dir)
        except FunctionTimeoutError:
            warnings.warn(
                f"write_batch_info timed out after {self.write_batch_info_timeout} seconds; "
                f"skipping batch info write for batch {self.batch_number - 1}",
                UserWarning
            )
        except FunctionExecutionError as exc:
            warnings.warn(
                f"write_batch_info raised an exception: {exc}; "
                f"skipping batch info write for batch {self.batch_number - 1}",
                UserWarning
            )

    def regression_test(self):
        if not self.test_data_csv:
            return None
        test_df = pd.read_csv(self.test_data_csv)
        out_col = [col for col in test_df.columns if 'output' in col]
        if len(out_col) != 1:
            raise ValueError(
                f"More than one output col detected when doing regression test. "
                f"Check {self.test_data_csv} and ensure only one column has 'output' in the name."
            )
        X_test = test_df[self.parameters].values
        y_test = test_df[out_col[0]].values
        y_pred = self.surrogate_predict(X_test)
        residuals = y_test - y_pred
        rmse = np.sqrt(np.nanmean(residuals ** 2))
        regression_results = {
            f"rmse_{len(y_test)}-{self.test_data_name}": [rmse]
        }
        return regression_results

    def _write_batch_info_inner(self, batch_dir, name=''):
        uq_results = self.uq_analysis()
        regression_results = self.regression_test()
        if regression_results:
            batch_info = {**uq_results, **regression_results}
        else:
            batch_info = uq_results
        df = pd.DataFrame(batch_info)
        df.to_csv(os.path.join(batch_dir, name + 'batch_info.csv'), index=False)

        all_batch_info_path = os.path.join(
            os.path.dirname(batch_dir),
            name + 'batch_info.csv'
        )
        if os.path.exists(all_batch_info_path):
            df.to_csv(all_batch_info_path, mode='a', header=False, index=False)
        else:
            df.to_csv(all_batch_info_path, mode='w', header=True, index=False)

    # ---- acquisition ----------------------------------------------------------

    def _compute_acquisition(self, X_pool, mode='var', blend_string=None, model=None, pool_y=None):
        start = time.time()
        if model is None:
            model = self.gp_model

        if mode == 'blend':
            if blend_string is None:
                raise ValueError('blend_string must be provided for blend mode')
            blend = self._parse_blend_string(blend_string)
            total = np.zeros(len(X_pool))
            for coeff, m in blend:
                if len(X_pool) > self.chunk_size:
                    scores = self._compute_acquisition_chunked(
                        X_pool, mode=m, chunk_size=self.chunk_size, model=model, pool_y=pool_y
                    )
                else:
                    scores = self._compute_acquisition_unchunked(
                        X_pool, mode=m, model=model, pool_y = pool_y
                    )
                scores = (scores - scores.mean()) / (scores.std() + 1e-12)
                total += coeff * scores
            end = time.time()
            print(
                'COMPUTING ACQUISITION TOOK:',
                (end - start) / 60.0,
                'min',
                f"MODE: {mode}"
            )
            return total

        if len(X_pool) > self.chunk_size:
            scores = self._compute_acquisition_chunked(
                X_pool, mode=mode, chunk_size=self.chunk_size, model=model, pool_y = pool_y
            )
        else:
            scores = self._compute_acquisition_unchunked(
                X_pool, mode=mode, model=model, pool_y = pool_y
            )
        end = time.time()
        print(
            'COMPUTING ACQUISITION TOOK:',
            (end - start) / 60.0,
            'min',
            f"MODE: {mode}"
        )
        return scores

    def _compute_acquisition_unchunked(self, X_pool, mode, chunk_size=5000, model=None, pool_y = None):
        if model is None:
            model = self.gp_model

        if mode == 'random':
            return np.random.uniform(0, 1, len(X_pool))
        elif mode == 'var':
            mu, var = model.predict(X_pool)
            return var.flatten()
        elif mode in ('gradVar', 'grad'):
            X_pool = np.atleast_2d(X_pool)
            dmu, _ = model.predictive_gradients(X_pool)
            grads = np.linalg.norm(dmu, axis=1).squeeze()
            return grads
        elif mode == 'intVar':
            return self.integral_variance_reduction(X_pool)
        elif mode == 'ensembleDisagreement':
            preds = []
            n_folds = min(5, max(2, len(self.train)))
            kf = KFold(
                n_splits=n_folds,
                shuffle=True,
                random_state=self.seed + self.batch_number
            )
            X_all, Y_all = self._get_unitXY()
            for train_idx, _ in kf.split(X_all):
                X_fold = X_all[train_idx]
                Y_fold = Y_all[train_idx]
                kernel_fold = GPy.kern.RBF(input_dim=X_fold.shape[1], ARD=True)
                kernel_fold.variance = self.kernel_variance
                kernel_fold.lengthscale = self.lengthscales.copy()
                model_fold = GPy.models.GPRegression(X_fold, Y_fold, kernel_fold)
                model_fold.likelihood.variance = self.noise_variance
                mu_f, _ = model_fold.predict(X_pool)
                preds.append(mu_f.flatten())
            preds = np.vstack(preds)
            return preds.var(axis=0)
        
        elif mode == "var_distpen":
            print('ACQUISITION MODE: var_distpen') 
            X_train, Y_train = self._get_unitXY()

            # GP posterior variance
            mu, var = model.predict_noiseless(X_pool)
            var = var.flatten()

            # Compute kernel similarity between pool points and training points
            # K_xt_x = k(X_pool, X_train)
            K_xt_x = model.kern.K(X_pool, X_train)  # shape (N_pool, N_train)

            # For each pool point, take the maximum similarity to any training point
            max_sim = np.max(K_xt_x, axis=1)  # shape (N_pool,)

            # Optionally normalize by kernel variance if needed
            # For many kernels, kern.variance is the output scale
            if hasattr(model.kern, "variance"):
                variance_scale = model.kern.variance[0] if np.ndim(model.kern.variance) > 0 else model.kern.variance
                # Avoid divide-by-zero
                if variance_scale > 0:
                    max_sim = max_sim / variance_scale

            # Distance penalty: low when similar to existing points, high when far
            # You can tune alpha; alpha=1 is a good default
            alpha = 1.0
            penalty = 1.0 - np.clip(max_sim, 0.0, 1.0) ** alpha

            score = var * penalty
            return score
        
        elif mode == "eigf": # doi: 10.1016/j.ress.2024.109945
            print('ACQUISITION MODE: eigf')
            X_train, Y_train = self._get_unitXY()
            
            from sklearn.neighbors import KDTree
            tree = KDTree(X_train)  # X_train: (n_samples, d)
            dist, idx = tree.query(X_pool, k=1)
            y_nearest = Y_train[idx[:, 0]]
            
            mu, var = model.predict_noiseless(X_pool)
            
            score = (mu - y_nearest)**2 + var
            return score

        elif mode == "vigf": # doi:10.1016/j.ress.2024.109945
            print('ACQUISITION MODE: vigf')
            X_train, Y_train = self._get_unitXY()
            
            from sklearn.neighbors import KDTree
            tree = KDTree(X_train)  # X_train: (n_samples, d)
            dist, idx = tree.query(X_pool, k=1)
            y_nearest = Y_train[idx[:, 0]]

            mu, var = model.predict_noiseless(X_pool)

            score = 4*var * ((mu-y_nearest)**2 + 2*var)
            return score
        
        elif mode == "gradunc":
            X_pool = np.atleast_2d(X_pool)
            dmu, dvar = model.predictive_gradients(X_pool)

            # dvar is (N, D)
            if dvar.ndim != 2:
                raise ValueError(f"Unexpected dvar shape: {dvar.shape}")

            # Largest per-dimension gradient variance
            grad_unc = np.max(dvar, axis=1)

            return grad_unc

                        
        elif mode in ("oracle_rmse", 'oracle_ipv', 'oracle'):
            print('ACQUISITION MODE: oracle')
            X_train, Y_train = self._get_unitXY()
            assert pool_y is not None
            if len(X_pool) > os.cpu_count()*100:
                warnings.warn(f'ORACLE ACQUISITION IS HEAVY AND REQUIRES MANY CORES. THE POOL IS MUCH LARGER THEN THE NUMBER OF CORES, POOL {len(X_pool)}, CORES {os.cpu_count}. PLEASE CONSIDER A SMALLER POOL.')
            # y_model, _ = self.gp_model.predict_noiseless(X_pool)
 
            # orig_rmse = np.sqrt(np.mean((y_model.flatten() - pool_y.flatten())**2)) 

            test_df = pd.read_csv(self.test_data_csv)
            out_col = [col for col in test_df.columns if 'output' in col]
            if len(out_col) != 1:
                raise ValueError(
                    f"More than one output col detected when doing regression test. "
                    f"Check {self.test_data_csv} and ensure only one column has 'output' in the name."
                )
            X_test_unit = self.to_unit(test_df[self.parameters].values)
            y_test = test_df[out_col[0]].values
            y_pred, var_orig = self.gp_model.predict_noiseless(X_test_unit)
            residuals = y_test - y_pred
            orig_rmse = np.sqrt(np.nanmean(residuals ** 2))

            def new_point_rmse_decrease(x_new, y_new):
                # Ensure shapes
                x_new = np.asarray(x_new).reshape(1, -1)
                y_new = np.asarray(y_new).reshape(1, 1)

                # Append new point
                X = np.vstack([X_train, x_new])
                y = np.vstack([Y_train, y_new])

                # Build model with fixed hyperparameters
                new_model = self.make_fixed_hyperparam_copy(X, y)
                #new_model.optimize_restarts(num_restarts=1, verbose=False)

                # Predict on pool
                y_model, _ = new_model.predict_noiseless(X_test_unit)

                # Compute RMSE
                rmse = np.sqrt(np.mean((y_model.flatten() - y_test.flatten())**2)) 

                decrease = orig_rmse - rmse
                return decrease
            
            ipv_orig = np.nanmean(var_orig)        
            def oracle_ipv_reduction(x_new, y_new):
                # Ensure shapes
                x_new = np.asarray(x_new).reshape(1, -1)
                y_new = np.asarray(y_new).reshape(1, 1)

                # Augment training data
                X_aug = np.vstack([X_train, x_new])
                Y_aug = np.vstack([Y_train, y_new])

                # Build model with fixed hyperparameters
                model_aug = self.make_fixed_hyperparam_copy(X_aug, Y_aug)

                # Predict variance on test set
                _, var_aug = model_aug.predict_noiseless(X_test_unit)

                ipv_aug = np.nanmean(var_aug)

                # Score = reduction in integrated posterior variance
                return ipv_orig - ipv_aug


            from joblib import Parallel, delayed, cpu_count

            print("Joblib sees", cpu_count(), "cores for oracle acquisition parallelism.")

            if mode in ('oracle_rmse', 'oracle'):
                score = np.array(Parallel(n_jobs=-1, prefer="threads", verbose=10)(
                    delayed(new_point_rmse_decrease)(xi, yi)
                    for xi, yi in zip(X_pool, pool_y)
                ))

            if mode == 'oracle_ipv':
                score = np.array(Parallel(n_jobs=-1, prefer="threads", verbose=10)(
                    delayed(oracle_ipv_reduction)(xi, yi)
                    for xi, yi in zip(X_pool, pool_y)
                ))

            return score

        else:
            raise ValueError(f"Unknown acquisition mode: {mode}")

    def make_fixed_hyperparam_copy(self, X_new, Y_new):
        # 1. Extract kernel hyperparameters from the trained model
        kern = self.gp_model.kern
        variance = float(kern.variance.values)
        lengthscales = kern.lengthscale.values.copy()

        # 2. Build a new kernel with the same structure
        new_kern = GPy.kern.RBF(input_dim=X_new.shape[1], ARD=True)
        jitter = 1e-8
        new_kern.variance = variance + jitter 
        new_kern.lengthscale = lengthscales

        # 3. Fix hyperparameters so they cannot be optimized
        new_kern.variance.fix()
        new_kern.lengthscale.fix()

        # 4. Build a new GP model using the fixed kernel
        new_model = GPy.models.GPRegression(X_new, Y_new, new_kern)

        # 5. Fix noise variance as well
        noise = float(self.gp_model.Gaussian_noise.variance.values)
        new_model.Gaussian_noise.variance = noise
        new_model.Gaussian_noise.variance.fix()

        return new_model


    def _compute_acquisition_chunked(self, X_pool, mode, chunk_size=5000, model=None, pool_y=None):
        results = []
        for i in range(0, len(X_pool), chunk_size):
            block = X_pool[i:i + chunk_size]
            results.append(
                self._compute_acquisition_unchunked(block, mode=mode, model=model, pool_y=pool_y)
            )
        return np.concatenate(results)

    def _parse_blend_string(self, blend_string):
        """
        Parse a blend string like '0.33-var_0.33-gradvar_0.34-intvar'
        Returns a list of (weight, mode) tuples.
        """
        parts = blend_string.split('_')
        blend = []
        for part in parts:
            coeff, mode = part.split('-', 1)
            blend.append((float(coeff), mode))
        return blend

    def integral_variance_reduction(self, X_pool):
        X_train, _ = self._get_unitXY()
        n_train = X_train.shape[0]
        K = self.gp_model.kern.K(X_train) + np.eye(n_train) * self.noise_variance
        K_inv = np.linalg.pinv(K)
        phi_train = self._integral_k_over_domain(X_train)
        phi_pool = self._integral_k_over_domain(X_pool)
        K_cross = self.gp_model.kern.K(X_train, X_pool)
        K_self = np.diag(self.gp_model.kern.K(X_pool))
        diff = phi_pool - K_cross.T.dot(K_inv).dot(phi_train)
        num = diff ** 2
        denom = K_self - np.sum(K_cross.T.dot(K_inv) * K_cross.T, axis=1)
        results = np.where(denom > 1e-12, num / denom, 0.0)
        return results

    # ---- misc -----------------------------------------------------------------

    def add_rmse_column_to_batch_info(self):
        from enchanted_surrogates.utils.get_batch_dirs import get_batch_dirs
        
        batch_dirs = get_batch_dirs(self.base_run_dir)
        for i, batch_dir in enumerate(batch_dirs):
            if not os.path.exists(os.path.join(batch_dir, 'enchanted_dataset.csv')):
                continue
            self.append_train_data(batch_dir)
            if os.path.exists(os.path.join(batch_dir, 'gpy_model.pkl')):
                print('WRITING BATCH INFO FOR:', batch_dir)
                with open(os.path.join(batch_dir, 'gpy_model.pkl'), 'rb') as file:
                    self.gp_model = pickle.load(file)
                self.cache_hypers()
                self.cache_K()
                print("\n\n ================================== \n")
                reg_results = self.regression_test()
                if reg_results is None:
                    continue
                reg_results['num_samples'] = [len(self.train)]
                df = pd.DataFrame(reg_results)
                reg_path = os.path.join(os.path.dirname(batch_dir), 'regression_info.csv')
                print('debug, reg_path', reg_path)
                if os.path.exists(reg_path):
                    df.to_csv(reg_path, mode='a', header=False, index=False)
                else:
                    df.to_csv(reg_path, mode='w', header=True, index=False)

        batch_info_csv = os.path.join(self.base_run_dir, 'batch_info.csv')
        shutil.copy2(batch_info_csv, os.path.join(self.base_run_dir, 'batch_info_orig.csv'))
        merge_secondary_into_primary(
            primary_csv=batch_info_csv,
            secondary_csv=os.path.join(self.base_run_dir, 'regression_info.csv'),
            out_csv=batch_info_csv,
            key='num_samples'
        )

    def brute_force_uq_analysis(self):
        raise NotImplementedError('Brute force UQ analysis not implemented in this sampler.')

    def register_future(self, future):
        return None

    def register_futures(self, futures):
        return None
    
    
def merge_secondary_into_primary(primary_csv: str,
                                 secondary_csv: str,
                                 out_csv: str = None,
                                 key: str = "num_samples",
                                 how: str = "left",
                                 require_exact_match: bool = True) -> pd.DataFrame:
    """
    Load two CSV files into pandas DataFrames, add columns from the secondary
    DataFrame to the primary DataFrame for rows with the same `key` values,
    but only add columns that do not already exist in the primary.

    Args:
        primary_csv: path to primary CSV (kept as the main table / order preserved).
        secondary_csv: path to secondary CSV (source of extra columns).
        out_csv: optional path to write the resulting DataFrame to CSV.
        key: column name present in both CSVs used to align rows (default "num_samples").
        how: merge strategy relative to primary. Default "left" keeps all primary rows.
        require_exact_match: if True, assert that the set of key values in the
            secondary is a superset of those in primary (or exactly matching if how="inner"),
            otherwise raises ValueError. If False, missing keys in secondary will remain NaN.

    Returns:
        result_df: pandas DataFrame (primary with added columns from secondary).
    """
    # Load
    df_p = pd.read_csv(primary_csv)
    df_s = pd.read_csv(secondary_csv)

    # Basic sanity
    if key not in df_p.columns:
        raise KeyError(f"Primary CSV does not contain key column '{key}'")
    if key not in df_s.columns:
        raise KeyError(f"Secondary CSV does not contain key column '{key}'")

    # Ensure key uniqueness in secondary if we intend to merge 1:1
    if df_s[key].duplicated().any():
        # If duplicates are expected, user should aggregate beforehand.
        raise ValueError(f"Secondary CSV contains duplicate '{key}' values; please aggregate or deduplicate.")

    # Check matching keys if required
    prim_keys = set(df_p[key].unique())
    sec_keys  = set(df_s[key].unique())

    if require_exact_match:
        missing_in_secondary = prim_keys - sec_keys
        if missing_in_secondary:
            raise ValueError(f"The following {key} values are in primary but missing in secondary: "
                             f"{sorted(list(missing_in_secondary))[:10]}{'...' if len(missing_in_secondary)>10 else ''}")

    # Select only columns from secondary that don't already exist in primary (except the key)
    new_cols = [c for c in df_s.columns if c != key and c not in df_p.columns]
    if not new_cols:
        # Nothing to add — return primary as-is (optionally write out)
        if out_csv:
            df_p.to_csv(out_csv, index=False)
        return df_p

    # Prepare reduced secondary df with key + new columns
    df_s_reduced = df_s[[key] + new_cols].copy()

    # Merge: keep primary order; by default left join retains primary rows
    result = pd.merge(df_p, df_s_reduced, on=key, how=how, validate="one_to_one")

    # Optionally write to CSV
    if out_csv:
        result.to_csv(out_csv, index=False)

    return result

if __name__ == '__main__':
    import sys
    import yaml
    from enchanted_surrogates.utils.load_configuration import load_configuration
    from enchanted_surrogates.utils.precise_imports import import_sampler
    _, base_run_dir = sys.argv
            
    listdir = os.listdir(base_run_dir)
    config_file_name = [name for name in listdir if '.yaml' in name]
    if len(config_file_name) > 1:
        raise FileNotFoundError('More than one .yaml file in base_run_dir, not sure which to use as config file')
    config_file_name = config_file_name[0]
    print('CONFIG FOUND:',os.path.join(base_run_dir, config_file_name))
    config = load_configuration(os.path.join(base_run_dir, config_file_name))
    
    
    sampler_config = config.executor['sampler_config']
    sampler_config['base_run_dir'] = base_run_dir
    
    gpy = GpyAnalyticSobolSamplerOld(**sampler_config)
    
    gpy.add_rmse_column_to_batch_info()
    