import os
import math
import pickle
import warnings
import time
from scipy.special import erf
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import traceback
import GPy
import time
from enchanted_surrogates.samplers.base_sampler import Sampler
from enchanted_surrogates.utils.precise_imports import import_sampler
from enchanted_surrogates.utils.timeout import run_with_timeout, FunctionTimeoutError, FunctionExecutionError
from enchanted_surrogates.utils.print_stats_table import print_stats_table

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

        # store scaling factors
        self._lb = np.array([b[0] for b in self.bounds], dtype=float)
        self._ub = np.array([b[1] for b in self.bounds], dtype=float)
        self._range = self._ub - self._lb

        self.acquisition_mode = kwargs.get("acquisition_mode", "variance")
        self.alpha = kwargs.get("alpha", 0.5)
        self.blend_string = kwargs.get("blend_string", None)
        self.chunk_size = kwargs.get("chunk_size", 5000)
        self.test_data_csv = kwargs.get("test_data_csv", None)
        self.test_data_name = kwargs.get("test_data_name", "test")
        
        self.sampling_strategy = kwargs.get('sampling_strategy', 'random')
        self.n_ensembles = kwargs.get('n_ensembles', 1) #useful if using a single acquisition function to help spread out the samples.
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
        # pool sampler
        self.pool_sampler_config = kwargs.get('pool_sampler_config', None)
        if self.pool_sampler_config:
            # enforce unit bounds
            if self.pool_sampler_config.get('bounds') != [(0.0,1.0)]*len(self.bounds):
                warnings.warn("Pool sampler bounds corrected to unit [0,1].")
                self.pool_sampler_config['bounds'] = [(0.0,1.0)]*len(self.bounds)
            self.pool_sampler = import_sampler(self.pool_sampler_config['type'], self.pool_sampler_config)
        else:
            self.pool_sampler = None
        
        self.initial_pool_size = kwargs.get('initial_pool_size', 5000)
        self.pool = None
        self._init_pool()
        
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
        # Start with all q’s
        arr = np.full(n, q, dtype=int)
        # Distribute the remainder across the first r entries
        arr[:r] += 1
        return arr

    # ---------------------------
    # Scaling helpers
    # ---------------------------
    def to_unit(self, X):
        """Map real-bounds inputs to [0,1]."""
        return (np.asarray(X) - self._lb) / self._range

    def from_unit(self, X_unit):
        """Map unit inputs back to real bounds."""
        return self._lb + np.asarray(X_unit) * self._range

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
            self.pool = rng.uniform(low=0,
                                    high=1,
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
        # If a sub-sampler is configured, prefer it for initial samples
        if self.sub_sampler is not None:
            samples = self.sub_sampler.get_next_samples()
            if samples:
                self.batch_number += 1
                self.submitted += len(samples)
                self.custom_submitted += len(samples)
                # remove samples from pool if present
                self._remove_from_pool(samples)
                samples = samples * self.num_repeats
                if self.include_index:
                    samples = [
                        {**samp, 'index': ind} for samp, ind in zip(samples, range(len(samples)))]
                return samples

        # Otherwise return random points from pool
        if self.pool is None or len(self.pool) == 0:
            self._init_pool()
        
        if self.initial_pool_samples_strategy == 'random':
            rng = np.random.RandomState(self.seed)
            n = min(self.initial_batch_size, len(self.pool))
            idxs = rng.choice(len(self.pool), size=n, replace=False)
            chosen = self.pool[idxs]
            real_chosen = self.from_unit(chosen)
            self.pool = np.delete(self.pool, idxs, axis=0)
            samples = [{key: float(val) for key, val in zip(self.parameters, row)} for row in real_chosen]
        elif self.initial_pool_samples_strategy == 'first':
            n = min(self.initial_batch_size, len(self.pool))
            chosen = self.pool[:n]
            real_chosen = self.from_unit(chosen)
            self.pool = self.pool[n:]
            samples = [{key: float(val) for key, val in zip(self.parameters, row)} for row in real_chosen]

        samples = samples * self.num_repeats
        if self.include_index:
            samples = [
                {**samp, 'index': ind} for samp, ind in zip(samples, range(len(samples)))]

        self.batch_number += 1
        self.submitted += len(samples)
        self.custom_submitted += len(samples)
        return samples

    def _remove_from_pool(self, samples):
        # takes in real space samples. Needs to be converted into unit samples
        if self.pool is None or len(self.pool) == 0:
            return
        # build array of sample vectors
        vecs = np.array([[s[param] for param in self.parameters] for s in samples], dtype=float)
        vecs = self.to_unit(vecs)
        to_delete = []
        for v in vecs:
            # find matching row (exact match)
            matches = np.all(np.isclose(self.pool, v, atol=1e-12, rtol=0.0), axis=1)
            idxs = np.where(matches)[0]
            if idxs.size > 0:
                to_delete.append(idxs[0])
        if to_delete:
            self.pool = np.delete(self.pool, to_delete, axis=0)

    # ---------------------------
    # Main sampling loop
    # ---------------------------
    
    def get_data(self):
        data_df = pd.read_csv(os.path.join(self.base_run_dir, 'enchanted_dataset.csv'))
        output_col = [col for col in data_df.columns if 'output' in col]
        if len(output_col) != 1:
            raise RuntimeError('Exactly one output column required.')

        X_real = data_df[self.parameters].to_numpy()
        Y = data_df[output_col].to_numpy().reshape(-1,1)
        return X_real, Y

    def _get_unitXY(self):
        X_real, Y = self.get_data()
        X_unit = self.to_unit(X_real)
        return X_unit, Y
        
    # def fit(self):
    #     X, Y = self._get_unitXY()
    #     # ---------------------------
    #     # Global GP fit (optimize hyperparameters once)
    #     # ---------------------------
    #     input_dim = X.shape[1]
    #     kernel = GPy.kern.RBF(input_dim=input_dim, ARD=True)
    #     self.gp_model = GPy.models.GPRegression(X, Y, kernel)
    #     self.gp_model.Gaussian_noise.variance.constrain_positive()
    #     if self.optimize_global:
    #         try:
    #             self.gp_model.optimize(messages=False)
    #         except Exception as exc:
    #             print(f'GLOBAL OPTIMIZE FAILED. ERROR: {exc} \n TRACEBACK:\n{traceback.format_exc()}')
    
    def _get_unitXY_with_noise(self):
        """
        Returns:
        X_unit     : scaled inputs (unique points)
        Y_unique   : averaged outputs at each unique point
        noise_vars : variance of repeats at each point (jitter if none)
        se_vars    : standard error of variance at each point
        mean_sems  : standard error of the mean at each point
        counts     : number of repeats at each point
        """
        
        X_real, Y = self.get_data()

        # group by unique points
        unique_points, inverse = np.unique(X_real, axis=0, return_inverse=True)
        noise_vars = np.zeros(len(unique_points))
        se_vars = np.zeros(len(unique_points))
        Y_unique = np.zeros(len(unique_points))
        mean_sems = np.zeros(len(unique_points))
        counts = np.zeros(len(unique_points), dtype=int)

        for i, up in enumerate(unique_points):
            idxs = np.where(inverse == i)[0]
            y_vals = Y[idxs].flatten()
            counts[i] = len(y_vals)
            Y_unique[i] = np.mean(y_vals)

            if counts[i] > 1:
                v = np.var(y_vals, ddof=1)
                noise_vars[i] = v
                # standard error of variance estimate
                se_vars[i] = np.sqrt(2 * (v**2) / (counts[i] - 1))
                # standard error of the mean
                mean_sems[i] = np.std(y_vals, ddof=1) / np.sqrt(counts[i])
            else:
                noise_vars[i] = 1e-8  # jitter if no repeats
                se_vars[i] = 1e-8     # jitter for SE as well
                mean_sems[i] = 1e-8   # jitter for SEM

        X_unit = self.to_unit(unique_points)
        return X_unit, Y_unique.reshape(-1,1), noise_vars, se_vars, mean_sems, counts


    def fit(self):
        """
        Fit the main GP surrogate using per-point noise via heteroscedastic regression.
        If repeats exist, also fit the noise GP.
        """
        X, Y, noise_vars, se_vars, _, _ = self._get_unitXY_with_noise()
        kernel = GPy.kern.RBF(input_dim=X.shape[1], ARD=True)

        # Heteroscedastic model: allows per-point variances
        self.gp_model = GPy.models.GPHeteroscedasticRegression(X, Y, kernel)

        # Inject known per-point noise variances and fix them        
        nv = noise_vars.reshape(-1, 1)
        self.gp_model.likelihood.variance[:] = nv
        self.gp_model.likelihood.variance.fix()

        # Optimize kernel hyperparameters only
        if self.optimize_global:
            try:
                self.gp_model.optimize(messages=False)
            except Exception as exc:
                print(f'GLOBAL OPTIMIZE FAILED. ERROR: {exc}')

        # Fit noise GP if any repeats exist (non-jitter)
        if self.num_repeats > 1:
            self.fit_noise(X, noise_vars, se_vars)

    def fit_noise(self, X, noise_vars, se_vars):
        """
        Fit a GP to model per-point noise (std), with heteroscedastic training noise
        equal to the standard error of the variance estimate at each point.
        """
        # Train targets = std (sqrt of variance)
        noise_targets = np.sqrt(np.maximum(noise_vars, 0.0)).reshape(-1, 1)
        kernel = GPy.kern.RBF(input_dim=X.shape[1], ARD=True)

        # Heteroscedastic model for noise GP
        self.noise_gp = GPy.models.GPHeteroscedasticRegression(X, noise_targets, kernel)

        # Per-point training noise = SE(var) (fixed)
        se = se_vars.reshape(-1, 1)
        self.noise_gp.likelihood.variance[:] = se**2
        self.noise_gp.likelihood.variance.fix()

        try:
            self.noise_gp.optimize(messages=False)
        except Exception as exc:
            print(f'GLOBAL NOISE OPTIMIZE FAILED. ERROR: {exc}')

    def predict_noise(self, X_test):
        """
        Predict per-point noise (std dev) using the fitted noise GP.
        Returns:
        - predicted std (mean of std process)
        - predictive error on the std (posterior std of the noise GP)
        """
        if not hasattr(self, 'noise_gp') or self.noise_gp is None:
            raise RuntimeError("Noise GP not fitted. Call fit() first with repeats.")

        X_unit_test = self.to_unit(X_test)
        pred_std_mean, pred_std_var = self.noise_gp.predict_noiseless(X_unit_test)

        pred_std = pred_std_mean.flatten()
        pred_std_err = np.sqrt(np.maximum(pred_std_var.flatten(), 0.0))
        return pred_std, pred_std_err
    
    def predict_single_run_error(self, X_test):
        """
        Approximate the error of a single run at X_test:
        single_run_error ≈ 2 * (predicted noise std + its predictive error)
        This corresponds to ~97% Gaussian coverage (≈ 2σ) with a conservative bump.
        """
        pred_std, pred_err = self.predict_noise(X_test)
        return 2.0 * (pred_std + pred_err)

    def get_next_samples(self, batch_dir=None, *args, **kwargs):
        if self.batch_number == 0:
            return self.get_initial_samples()

        if not self.base_run_dir:
            raise RuntimeError('base_run_dir must be set to retrieve training data.')

        self.fit()
        print('debug shape variances self.gp_model.likelihood.variance.shape:', self.gp_model.likelihood.variance.shape)
            
        # Extract and cache hyperparameters
        self.cache_hypers()
        # cache K Cholesky and solve routine for UQ
        self.cache_K()
        if self.do_write_batch_info:
            start_wbi = time.time()
            previous_batch_dir = os.path.join(self.base_run_dir, f'batch_{self.batch_number-1}')
            print(f'debug is it writing every sample? custom_submitted: {self.custom_submitted}, num_samples_at_last_write: {self.num_samples_at_last_write}, self.write_batch_info_every_x_samples: {self.write_batch_info_every_x_samples}, self.batch_number {self.batch_number}')
            if self.custom_submitted - self.num_samples_at_last_write >= self.write_batch_info_every_x_samples or self.batch_number in [0,1,2,3]:
                # Now the surrogate is trained we can write batch info
                self.write_batch_info(previous_batch_dir)
                self.num_samples_at_last_write = self.custom_submitted

                # optionally save global model
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

        print('SELECTING NEW SAMPLES FROM POOL')
        # ---------------------------
        # Select new samples
        # ---------------------------
        samples = []
        if self.n_ensembles == 1:
            print(f'GETTING GLOBAL SCORE MODE:{self.acquisition_mode}')
            # Predictive variance on pool using global model (not used for fold selection but useful fallback)
            score_pool_global = self._compute_acquisition(self.pool, mode=self.acquisition_mode, blend_string=self.blend_string)
            score_pool_global = score_pool_global.flatten()

            print(f'BATCH SIZE:{self.batch_size} USING GLOBAL SCORE.')
            idx = list(np.argsort(-score_pool_global)[:self.batch_size])
            chosen_points = self.pool[idx]
            real_chosen_points = self.from_unit(chosen_points)
            samples = [{key: float(v) for key, v in zip(self.parameters, row)} for row in real_chosen_points]
            self.pool = np.delete(self.pool, idx, axis=0)
        else:
            print(f'SPLITTING DATA INTO FOLDS')
            # Use folds but reuse global hyperparams (no re-optimizing)
            X_all, Y_all, noise_vars_all, se_vars_all, _, _ = self._get_unitXY_with_noise()

            n_folds = min(self.n_ensembles,len(Y_all))
            samples_per_fold = self.split_integer(self.batch_size, self.n_ensembles)
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=self.seed + self.batch_number)
            
            chosen_indices = []
            for i, train_idx, _ in enumerate(kf.split(X_all)):
                print(f'CALCULATING ACQUISITON FUNCTION FOR FOLD {i+1}')
                X_fold = X_all[train_idx]
                Y_fold = Y_all[train_idx]
                # Build kernel with cached hyperparams and do NOT optimize
                kernel_fold = GPy.kern.RBF(input_dim=X_fold.shape[1], ARD=True)
                
                
                # assign hyperparameters back to kernel object
                try:
                    kernel_fold.variance = self.kernel_variance
                    kernel_fold.lengthscale = self.lengthscales.copy()
                    # fix kernel parameters so they are not re-optimized accidentally
                    kernel_fold.variance.fix(self.kernel_variance)
                    kernel_fold.lengthscale.fix(self.lengthscales.copy())
                except Exception:
                    try:
                        kernel_fold.variance = self.kernel_variance
                    except Exception:
                        pass

                model_fold = GPy.models.GPHeteroscedasticRegression(X_fold, Y_fold, kernel_fold)
                nv = noise_vars_all[train_idx].reshape(-1, 1)

                model_fold.likelihood.variance[:] = nv
                model_fold.likelihood.variance.fix()

                # try:
                #     # Do not optimize kernel; only allow small local noise adjustment if desired
                #     model_fold.optimize_restarts(num_restarts=0, messages=False)
                # except Exception:
                #     # ignore fold optimization failures
                #     pass

                # _, var_f = model_fold.predict(self.pool)
                # var_f = var_f.flatten()
                # idx_f = int(np.argmax(var_f))
                # chosen_indices.append(idx_f)
                
                scores = self._compute_acquisition(self.pool, mode=self.acquisition_mode, blend_string=self.blend_string, model=model_fold)
                idx_f = list(np.argsort(-scores)[:samples_per_fold[i]])
                chosen_indices.extend(idx_f)
                # chosen_points = self.pool[idxs]
                # samples = [{key: float(v) for key, v in zip(self.parameters, row)} for row in chosen_points]
                # self.pool = np.delete(self.pool, idxs, axis=0)

            # make unique while preserving order
            seen = set()
            unique_idxs = []
            for idx in chosen_indices:
                if idx not in seen:
                    seen.add(idx)
                    unique_idxs.append(idx)

            # if we need more points (duplicates), take additional top global var ones
            if len(unique_idxs) < self.batch_size:
                print('THE GPR ENSEMBLE SELECTED SOME OF THE SAME POINTS. USING GLOBAL SCORE TO GET MORE POINTS')
                print(f'GETTING GLOBAL SCORE MODE:{self.acquisition_mode}')
                # Predictive variance on pool using global model (not used for fold selection but useful fallback)
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
            real_chosen_points = self.from_unit(chosen_points)
            samples = [{key: float(v) for key, v in zip(self.parameters, row)} for row in real_chosen_points]
            # remove chosen points from pool
            self.pool = np.delete(self.pool, chosen_indices_final, axis=0)

        samples = samples * self.num_repeats
        if self.include_index:
            samples = [{**samp, 'index': ind} for samp, ind in zip(samples, range(self.custom_submitted,len(samples)))]
        
        # increment counters and return
        self.batch_number += 1
        if self.custom_submitted >= self.budget:
            print('DOING LIGHT POST PROCESSING FROM SAPLER')
            self.post_process()
            return None

        if samples is not None:
            self.custom_submitted += len(samples)
        
        return samples

    def load_model(self, model_path=None, directory=None):
        if model_path == None:
            if os.path.exists(os.path.join(directory, 'gpy_model.csv')):
                model_path = os.path.join(directory, 'gpy_model.csv')
        
        with open(model_path, 'rb') as file:
            self.gp_model = pickle.load(file)
            self.cache_hypers()
            self.cache_K()
    
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
        self.noise_variances = self.gp_model.likelihood.variance[:]

        # try:
        #     self.noise_variance = float(self.gp_model.likelihood.variance.values[0])
        # except Exception:
        #     self.noise_variance = float(self.gp_model.likelihood.variance)

    def cache_K(self):
        X, Y = self._get_unitXY()
        diag_noise = np.diag(self.gp_model.likelihood.variance[:])
        K_full = self.gp_model.kern.K(X) + diag_noise
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

    
    # ---------------------------
    # Predictor
    # ---------------------------
    
    def surrogate_predict(self, samples):
        """
        Accept samples in real bounds, scale to unit, predict with GP.
        Return mean and posterior std of the mean function.
        """
        if self.gp_model is None:
            self.fit()

        samples_unit = self.to_unit(samples)
        ypred, post_var = self.gp_model.predict_noiseless(samples_unit)
        return ypred.flatten(), np.sqrt(np.maximum(post_var.flatten(), 0.0))

        
    # ---------------------------
    # Integrals and analytic UQ (predictor-only: Marrel et al.)
    # ---------------------------
    def _integral_k_over_domain(self, X_unit):
        n, D = X_unit.shape
        I = np.ones(n, dtype=float)
        for d in range(D):
            a, b = 0.0, 1.0
            ls = self.lengthscales[d]
            Xi_d = X_unit[:, d]
            I_d = rbf_kernel_product_integral_1d_vector(Xi_d, ls, a, b)
            # I_d *= self._range[d]   # <-- scale by real interval length
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
            # C_d *= self._range[d]   # <-- scale by real interval length
            C *= C_d
        C *= (self.kernel_variance ** 2)
        return C

    def uq_analysis(self):
        print("\n\n\n ================================== \n\n\n")

        X, y = self._get_unitXY()
        n, D = X.shape
        vol = 1 #np.prod([b[1] - b[0] for b in self.bounds])
        # real_vol = np.prod(self._range)   # real domain volume

        # integrals
        I = self._integral_k_over_domain(X)

        # Use cached solver to invert K without forming explicit inverse
        # K = self.gp_model.kern.K(X) + np.eye(n) * max(self.noise_variance, 1e-8)
        K = self.gp_model.kern.K(X) + np.diag(np.maximum(self.noise_variances, 1e-8))
        try:
            K_inv_y = self._solve_K(y)
        except Exception:
            print('cholsky method failed, falling back to pinv')
            K_inv = np.linalg.pinv(K)
            K_inv_y = K_inv.dot(y)

        integral_m = I.reshape(1, -1).dot(K_inv_y)  # scalar
        mu = float(integral_m / vol)

        # C matrix and total predictor variance
        C = self._integral_kk_over_domain(X)
        try:
            # compute A = K^{-1} C K^{-1} y efficiently: first K^{-1} C
            # Use K_inv as linear operator via solve if possible
            # compute K^{-1} C via solving K X = C for X
            # for stability and simplicity we compute K_inv explicitly if small n
            K_inv = None
            if self._K_cholesky is not None:
                # compute K_inv by solving K * E = I
                I_eye = np.eye(n)
                K_inv = np.column_stack([self._solve_K(I_eye[:, i:i+1]).flatten() for i in range(n)])
            else:
                K_inv = np.linalg.pinv(K)
            A = K_inv.dot(C).dot(K_inv)
            var_pred = float((y.T.dot(A).dot(y)) / vol - mu ** 2)
        except Exception:
            var_pred = 0.0

        var_pred = max(var_pred, 0.0)

        # first-order Sobol (predictor-only)
        sobol_first = {}
        for j in range(D):
            prod_other = np.ones(n, dtype=float)
            for d in range(D):
                if d == j:
                    continue
                ls = self.lengthscales[d]
                Xi_d = X[:, d]
                I_d = rbf_kernel_product_integral_1d_vector(Xi_d, ls, 0, 1)
                # I_d *= self._range[d]
                prod_other *= I_d

            ls_j = self.lengthscales[j]
            Xi_j = X[:, j]
            Mj = rbf_kernel_product_double_integral_1d_matrix(Xi_j, Xi_j, ls_j, 0, 1)
            # Mj *= self._range[j]
            outer_prod = np.outer(prod_other, prod_other)
            B = (self.kernel_variance ** 2) * outer_prod * Mj

            try:
                if K_inv is None:
                    K_inv = np.linalg.pinv(K)
                num = float((y.T.dot(K_inv.dot(B).dot(K_inv)).dot(y)) / vol - mu ** 2)
                # print('DEBUG num:',num)
            except Exception as exc:
                print('EXCEPTION WHEN GETTING num:', exc)
                print('TRACEBACK: \n', traceback.format_exc())
                num = 0.0
            num = max(num, 0.0)
            # print(f'debug var_pred: {var_pred} | num: {num}')
            sobol_first[self.parameters[j] + '_sobolF'] = float(num / var_pred) if var_pred > 0 else 0.0

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
            # run_with_timeout(self._write_batch_info_inner, self.write_batch_info_timeout, kwargs={'batch_dir': batch_dir})
            self._write_batch_info_inner(batch_dir=batch_dir)
        except FunctionTimeoutError:
            warnings.warn(f"write_batch_info timed out after {self.write_batch_info_timeout} seconds; skipping batch info write for batch {self.batch_number-1}", UserWarning)
        except FunctionExecutionError as exc:
            warnings.warn(f"write_batch_info raised an exception: {exc}; skipping batch info write for batch {self.batch_number-1}", UserWarning)

    def regression_test(self):
        if not self.test_data_csv:
            return None
        else:
            test_df = pd.read_csv(self.test_data_csv)
            out_col = [col for col in test_df.columns if 'output' in col]
            if len(out_col) > 2:
                raise ValueError(f'MORE THAN ONE OUTPUT COL DETETED WHEN DOING REGRESSION TEST. CHECK {self.test_data_csv} FILE AND ENSURE ONLY ONE COLUMN HAS output IN THE NAME.')
            X_test = test_df[self.parameters].values
            y_test = test_df[out_col[0]].values
            y_pred, _ = self.surrogate_predict(X_test)
            # print('debug, xtest', X_test)
            print('debug, ytest n nans', np.isnan(y_test).sum())
            print('debug, ypred n nans', np.isnan(y_pred).sum())
            residuals = y_test - y_pred
            print('debug n nans', np.isnan(residuals).sum())
            rmse = np.sqrt(np.nanmean((y_test - y_pred) ** 2))
            
            print('debug rmse', rmse)
            regression_results = {f'rmse_{len(y_test)}-{self.test_data_name}':[rmse]}
            return regression_results
        
    def _write_batch_info_inner(self, batch_dir, name=''):
        uq_results = self.uq_analysis()
        regression_results = self.regression_test()
        if regression_results:
            batch_info = {**uq_results, **regression_results}
        else:
            batch_info = uq_results
        df = pd.DataFrame({k: v for k, v in batch_info.items()})
        df.to_csv(os.path.join(batch_dir, name+'batch_info.csv'), index=False)
        all_batch_info_path = os.path.join(os.path.dirname(batch_dir), name+'batch_info.csv')
        if os.path.exists(all_batch_info_path):
            df.to_csv(all_batch_info_path, mode='a', header=False, index=False)
        else:
            df.to_csv(all_batch_info_path, mode='w', header=True, index=False)

    def _compute_acquisition(self, X_pool, mode="var", blend_string=None, model=None):
        start = time.time()        
        if mode == "blend":
            if blend_string is None:
                raise ValueError("blend_string must be provided for blend mode")
            blend = self._parse_blend_string(blend_string)

            total = np.zeros(len(X_pool))
            for coeff, m in blend:
                if len(X_pool) > self.chunk_size:
                    scores = self._compute_acquisition_chunked(X_pool, mode=m, chunk_size=self.chunk_size, model=model)
                    
                else:
                    scores = self._compute_acquisition_unchunked(X_pool, mode=m, model=model)
                scores = (scores - scores.mean()) / (scores.std() + 1e-12)
                total += coeff * scores

            end = time.time()
            print('COMPUTING ACQUISITION TOOK:',(end-start)/60, 'min', f'MODE:{mode}')
            return total

        else:
            if len(X_pool) > self.chunk_size:
                scores = self._compute_acquisition_chunked(X_pool, mode=mode, chunk_size=self.chunk_size)
                
            else:
                scores = self._compute_acquisition_unchunked(X_pool, mode=mode)
            end = time.time()
            print('COMPUTING ACQUISITION TOOK:',(end-start)/60, 'min', f'MODE:{mode}')
            return scores

    def _compute_acquisition_unchunked(self, X_pool, mode, chunk_size=5000, model=None):
        if model is None:
            model = self.gp_model
        if mode == "var":
            mu, var = model.predict_noiseless(X_pool)
            return var.flatten()
        
        if mode == "noise_and_var":
            # GP posterior variance
            print('debug shape variances self.gp_model.likelihood.variance.shape: in noise and var', self.gp_model.likelihood.variance.shape)
            # Y_metadata = {'output_index': np.arange(X_pool.shape[0]).reshape(-1,1)}
            mu, var_model = self.gp_model.predict_noiseless(X_pool)
            # _, var_model = self.gp_model.predict(X_pool)
            var_model = var_model.flatten()

            # predicted noise std + error from noise_gp
            noise_std, noise_err = self.predict_noise(self.from_unit(X_pool))
            var_noise = (noise_std + noise_err)**2

            return var_model + var_noise

        if mode == "noise":
            noise_std, noise_err = self.predict_noise(self.from_unit(X_pool))
            var_noise = (noise_std + noise_err)**2
            return var_noise

        elif mode == "grad":
            # Ensure X_pool is 2D
            X_pool = np.atleast_2d(X_pool)

            # Predictive gradients for the whole pool
            dmu, _ = model.predictive_gradients(X_pool)

            # dmu shape is (N, input_dim, output_dim)
            # Take the norm across the input_dim axis
            grads = np.linalg.norm(dmu, axis=1).squeeze()
            return grads

        elif mode == "intVar":
            return self.integral_variance_reduction(X_pool)

        else:
            raise ValueError(f"Unknown acquisition mode: {mode}")

    def _compute_acquisition_chunked(self, X_pool, mode, chunk_size=5000, model=None):
        results = []
        for i in range(0, len(X_pool), chunk_size):
            block = X_pool[i:i+chunk_size]
            results.append(self._compute_acquisition_unchunked(block, mode=mode, model=model))
        return np.concatenate(results)

    def _parse_blend_string(self, blend_string):
        """
        Parse a blend string like '0.33-var_0.33-gradvar_0.34-intvar'
        Returns a list of (weight, mode) tuples.
        """
        parts = blend_string.split("_")
        blend = []
        for part in parts:
            coeff, mode = part.split("-", 1)
            blend.append((float(coeff), mode))
        return blend


    def integral_variance_reduction(self, X_pool):
        X_train, _ = self._get_unitXY()
        n_train = X_train.shape[0]

        # Kernel matrix and inverse
        K = self.gp_model.kern.K(X_train) + np.diag(np.maximum(self.noise_variances, 1e-8))
        K_inv = np.linalg.pinv(K)

        # Kernel mean features for training points
        phi_train = self._integral_k_over_domain(X_train)  # shape (n_train,)

        # Kernel mean features for all pool points
        phi_pool = self._integral_k_over_domain(X_pool)    # shape (n_pool,)

        # Cross‑covariance between train and pool
        K_cross = self.gp_model.kern.K(X_train, X_pool)    # shape (n_train, n_pool)

        # Self‑covariance for pool points
        K_self = np.diag(self.gp_model.kern.K(X_pool))     # shape (n_pool,)

        # Compute numerator and denominator vectorized
        diff = phi_pool - K_cross.T.dot(K_inv).dot(phi_train)   # shape (n_pool,)
        num = diff**2
        denom = K_self - np.sum(K_cross.T.dot(K_inv) * K_cross.T, axis=1)

        # Safe division
        results = np.where(denom > 1e-12, num / denom, 0.0)
        return results

    def post_process(self):
        """
        Collapse repeated input points into unique entries and write
        enchanted_dataset_collapsed.csv with mean, std, mean_error, std_error, and count.

        Output file is written into self.base_run_dir (must be set).
        """
        if not self.base_run_dir:
            raise RuntimeError("base_run_dir must be set to write collapsed dataset.")

        # Get collapsed data
        X_unit, Y_unique, noise_vars, se_vars, se_Y, counts = self._get_unitXY_with_noise()
        X_real = self.from_unit(X_unit)

        # Compute statistics
        mean = Y_unique.flatten()
        se_mean = se_Y
        std = np.sqrt(np.maximum(noise_vars, 0.0))
        se_std = se_vars

        # Build dataframe
        df = pd.DataFrame(X_real, columns=self.parameters)
        df['count'] = counts
        df['mean'] = mean
        df['std'] = std
        df['se_mean'] = se_mean
        df['se_std'] = se_std

        # Write CSV
        out_path = os.path.join(self.base_run_dir, 'enchanted_dataset_collapsed.csv')
        df.to_csv(out_path, index=False)
        print(f"Collapsed dataset written to {out_path}")


    # ---------------------------
    # Boilerplate
    # ---------------------------
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
    from enchanted_surrogates.utils.get_batch_dirs import get_batch_dirs
    from enchanted_surrogates.utils.load_configuration import load_configuration
    from enchanted_surrogates.utils.precise_imports import import_sampler
    _, base_run_dir = sys.argv
    
    batch_dirs = get_batch_dirs(base_run_dir)
    
    listdir = os.listdir(base_run_dir)
    config_file_name = [name for name in listdir if '.yaml' in name]
    if len(config_file_name) > 1:
        raise FileNotFoundError('More than one .yaml file in base_run_dir, not sure which to use as config file')
    config_file_name = config_file_name[0]
    print('CONFIG FOUND:',os.path.join(base_run_dir, config_file_name))
    config = load_configuration(os.path.join(base_run_dir, config_file_name))
    
    
    sampler_config = config.executor['sampler_config']
    sampler_config['base_run_dir'] = base_run_dir
    
    gpy = GpyAnalyticSobolSampler(**sampler_config)    
    for i, batch_dir in enumerate(batch_dirs):
        gpy.append_train_data(batch_dir)
        if os.path.exists(os.path.join(batch_dir, 'gpy_model.pkl')):
            print('WRITING BATCH INFO FOR:',batch_dir)
            with open(os.path.join(batch_dir, 'gpy_model.pkl'), 'rb') as file:
                gpy.gp_model = pickle.load(file)
                # gpy.fit()
                gpy.cache_hypers()
                gpy.cache_K()
            print("\n\n\n ================================== \n\n\n")
            gpy._write_batch_info_inner(batch_dir, name='post2_')
            
            # reg_results = gpy.regression_test()
            # reg_results['num_samples'] = len(gpy.train)
            # df = pd.DataFrame({k: v for k, v in reg_results.items()})
            # reg_path = os.path.join(os.path.dirname(batch_dir), 'regression_info.csv')
            # if os.path.exists(reg_path):
            #     df.to_csv(reg_path, mode='a', header=False, index=False)
            # else:
            #     df.to_csv(reg_path, mode='w', header=True, index=False)
         
        # if i == 4:
        #     break
   
 
    # merge_secondary_into_primary(
    #     primary_csv=os.path.join(base_run_dir, 'batch_info.csv'),
    #     secondary_csv=os.path.join(base_run_dir, 'regression_info.csv'),
    #     out_csv=os.path.join(base_run_dir, 'merged_batch_info.csv'),
    #     key='num_samples')
 

    # merge_secondary_into_primary(
    #     primary_csv='/scratch/project_462000954/daniel/enchanted_test/AUG_33585_UQ_12D_anovaSpatDim32/batch_info.csv',
    #     secondary_csv='/scratch/project_462000954/daniel/enchanted_test/AUG_33585_UQ_12D_anovaSpatDim32/post3batch_info.csv',
    #     out_csv=os.path.join('/scratch/project_462000954/daniel/enchanted_test/AUG_33585_UQ_12D_anovaSpatDim32/', 'merged_batch_info.csv'),
    #     key='num_samples')
