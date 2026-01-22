import os
import math
import pickle
import warnings
import time
from scipy.special import erf
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import traceback
import GPy
import time
from enchanted_surrogates.samplers.base_sampler import Sampler
from enchanted_surrogates.utils.precise_imports import import_sampler
from enchanted_surrogates.utils.timeout import run_with_timeout, FunctionTimeoutError, FunctionExecutionError
from enchanted_surrogates.utils.print_stats_table import print_stats_table
from enchanted_surrogates.utils.get_batch_dirs import get_batch_dirs


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
        self.ignore_zeros = kwargs.get('ignore_zeros', False)
        self.acquisition_mode = kwargs.get("acquisition_mode", "variance")
        self.alpha = kwargs.get("alpha", 0.5)
        self.blend_string = kwargs.get("blend_string", None)
        self.chunk_size = kwargs.get("chunk_size", 5000)
        self.test_data_csv = kwargs.get("test_data_csv", None)
        self.test_data_name = kwargs.get("test_data_name", "test")
        self.normalize_y = kwargs.get("normalize_y", True)
        self.sampling_strategy = kwargs.get('sampling_strategy', 'random')
        self.n_ensembles = kwargs.get('n_ensembles', 1) #useful if using a single acquisition function to help spread out the samples.
        self.batch_size = kwargs.get('batch_size', None) or 2
        self.initial_batch_size = kwargs.get('initial_batch_size', self.batch_size)
        self.initial_pool_samples_strategy = kwargs.get('initial_pool_samples_strategy', 'random')
        self.seed = kwargs.get('seed', 42)
        self.rng = np.random.RandomState(self.seed)
            
        self.base_run_dir = kwargs.get('base_run_dir')
        self.num_repeats = kwargs.get('num_repeats', 1)
        self.include_index = kwargs.get('include_index', False)
        self.batch_number = 0
        self.submitted = 0
        self.custom_submitted = 0
        self.budget = kwargs.get('budget', None) or self.batch_size * self.num_repeats
        self.output_col = kwargs.get('output_col', None)
        self.global_noise = kwargs.get('global_noise', 0)
        self.optimise_global_noise_if_no_repeats = kwargs.get('optimise_global_noise_if_no_repeats', True)
        
        self.rand1 = True
        self.slices_res = kwargs.get('slices_res', 10)

        self.max_y = kwargs.get('max_y',None) # above this value data will be ignored. Potentially because it blongs to a different class
        
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
        self.pool_y = None
        self.pool_csv_path = kwargs.get('pool_csv_path', None)
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

        # Normalisation for main GP targets
        self._y_mean = None
        self._y_std  = None

        # Normalisation for noise GP targets
        self._noise_mean = None
        self._noise_std  = None


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
        # rng = np.random.RandomState(self.seed)
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
        if self.pool_csv_path is not None:
            df = pd.read_csv(self.pool_csv_path)
            self.pool = self.to_unit(df[self.parameters].to_numpy())
            output_col = [col for col in df.columns if 'output' in col]
            if len(output_col)>1:
                warnings.warn(f'MORE THAN ONE OUTPUT WAS FOUND IN THE POOL CSV: {output_col}. THE FIRST WILL BE TAKEN AS THE OUTPUT OF INTEREST. {output_col[0]}')
            output_col = output_col[0]
            self.pool_y = df[output_col].to_numpy() # unormalised

        else:
            self.pool = self.rng.uniform(low=0,
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
            n = min(self.initial_batch_size, len(self.pool))
            idxs = self.rng.choice(len(self.pool), size=n, replace=False)
            chosen = self.pool[idxs]
            real_chosen = self.from_unit(chosen)
            self.pool = np.delete(self.pool, idxs, axis=0)
            if self.pool_y is not None:
                self.pool_y = np.delete(self.pool_y, idxs, axis=0)
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
            if self.pool_y is not None:
                self.pool_y = np.delete(self.pool_y, to_delete, axis=0)

    # ---------------------------
    # Main sampling loop
    # ---------------------------
    
    def get_data(self, timeout=60*5):
        start = time.time()
        print('debug get_data batch_numer:', self.batch_number)
        if self.batch_number is not None:
            _batch_dirs = get_batch_dirs(self.base_run_dir)
            dfs = []
            print('debug len _bd', len(_batch_dirs))
            for _i in range(min(self.batch_number,len(_batch_dirs))):
                bd = _batch_dirs[_i]
                # print('debug get_data bd', bd)
                dd = os.path.join(bd, 'enchanted_dataset.csv')
                while not os.path.exists(dd):
                    time.sleep(0.1)
                    if time.time() - start > timeout:
                        warnings.warn(f'get data timeout, {dd} does not exist')
                        continue
                dfi = pd.read_csv(dd, on_bad_lines='warn')
                # print('debug get_data len(dfi)', len(dfi))
                dfs.append(dfi)
            if len(dfs) == 0:
                raise RuntimeError('NO DATA FOUND IN ANY BATCH DIRECTORIES')
            data_df = pd.concat(dfs, axis=0)
        else:
            data_df = pd.read_csv(os.path.join(self.base_run_dir, 'enchanted_dataset.csv'), on_bad_lines='warn')
        if self.output_col is None:
            output_col = [col for col in data_df.columns if 'output' in col]
            self.output_col = output_col[0]

            if len(output_col) != 1:
                raise RuntimeError('Exactly one output column required.')

        X_real = data_df[self.parameters].to_numpy()
        Y = data_df[self.output_col].to_numpy().reshape(-1,1)
        if self.max_y:
            mask = Y<=self.max_y
            Y = Y[mask.ravel()]
            mask = mask.reshape(-1,1)
            X_real = X_real[mask.ravel()]
        
        if self.ignore_zeros:
            mask = Y.flatten()!=0
            Y = Y[mask]
            X_real = X_real[mask]

        return X_real, Y

    def _get_unitXY(self, normalize_y = False):
        X_real, Y = self.get_data()
        X_unit = self.to_unit(X_real)
        if normalize_y:
            Y = self.normalize_y(Y)
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
    
    def _get_unitXY_with_noise(self, normalize_y = False):
        """
        Returns:
        X_unit     : scaled inputs (unique points)
        Y_unique   : averaged outputs at each unique point
        noise_vars : variance of repeats at each point (jitter if none)
        se_vars    : standard error of variance at each point
        mean_sems  : standard error of the mean at each point
        counts     : number of repeats at each point
        """
        start = time.time()
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
                noise_vars[i] = self.global_noise + 1e-8  # jitter if no repeats
                se_vars[i] = 1e-8     # jitter for SE as well
                mean_sems[i] = 1e-8   # jitter for SEM

        X_unit = self.to_unit(unique_points)
        end = time.time()
        print(f'GETTING AND COLLAPSING DATA TOOK: {(end-start)/60} min')

        Y = Y_unique.reshape(-1,1)

        if normalize_y:
            # --- Normalization for main GP target ---
            print('debug mean Y', np.mean(Y))
            print('debug std Y', np.std(Y))
            Y_norm = self.normalize_y(Y)
            Y = Y_norm

        # counts must be more than 3 for it to be statistically valid
        if any(counts>1):
            mask = counts>3
            X_unit = X_unit[mask]
            Y = Y[mask]
            noise_vars = noise_vars[mask]
            se_vars = se_vars[mask]
            mean_sems = mean_sems[mask]
            counts = counts[mask]

        return X_unit, Y, noise_vars, se_vars, mean_sems, counts

    def var_to_std(self, var, var_err):
        """
        Convert variance ± error into std ± error using first‑order error propagation.

        Parameters
        ----------
        var : float or array
            Estimated variance.
        var_err : float or array
            Standard error (or std) of the variance estimate.

        Returns
        -------
        std : float or array
            Standard deviation = sqrt(var)
        std_err : float or array
            Error on the standard deviation
        """
        var = np.asarray(var)
        var_err = np.asarray(var_err)

        std = np.sqrt(var)
        std_err = var_err / (2 * std)

        return std, std_err

    def make_fixed_hyperparam_copy(self, X_new, Y_new):
        # 1. Extract kernel hyperparameters from the trained model
        kern = self.gp_model.kern
        variance = float(kern.variance.values)
        lengthscales = kern.lengthscale.values.copy()

        # 2. Build a new kernel with the same structure
        new_kern = GPy.kern.RBF(input_dim=X_new.shape[1], ARD=True)
        new_kern.variance = variance
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

    def fit(self):
        """
        Fit the main GP surrogate with normalization.

        Logic:
        - Always normalize Y.
        - If repeats exist → heteroscedastic GP using per‑point noise.
        - If no repeats and optimise_global_noise_if_no_repeats=True → GPRegression with global noise.
        - Else → heteroscedastic GP using jitter/noise_vars.
        """

        X, Y_norm, noise_vars, se_vars, se_mean, counts = self._get_unitXY_with_noise(normalize_y = self.normalize_y)
        print('debug y norm', Y_norm.shape)
        input_dim = X.shape[1]
        kernel = GPy.kern.RBF(input_dim=input_dim, ARD=True)

        # --- Case 1: Repeats exist → heteroscedastic GP with per‑point noise ---
        if np.any(counts > 1):
            print('FITTING: NOISE FROM REPEAT SAMPLES')
            self.gp_model = GPy.models.GPHeteroscedasticRegression(X, Y_norm, kernel)

            # scale noise after normalization
            nv = se_mean.reshape(-1, 1)
            if self.normalize_y:
                nv = (nv/self._y_std)**2
            self.gp_model.likelihood.variance[:] = nv
            self.gp_model.likelihood.variance.fix()

        # --- Case 2: No repeats, use global noise ---
        elif self.optimise_global_noise_if_no_repeats:
            print('FITTING: OPTIMISE NOISE')
            self.gp_model = GPy.models.GPRegression(X, Y_norm, kernel)

            # allow global noise to be optimized
            self.gp_model.Gaussian_noise.variance.constrain_positive()

        # --- Case 3: No repeats, fallback heteroscedastic with jitter/noise_vars ---
        else:
            print('FITTING: NOISE FIXED TO JITTER')
            self.gp_model = GPy.models.GPHeteroscedasticRegression(X, Y_norm, kernel)

            nv = noise_vars.reshape(-1, 1)
            if self.normalize_y:
                nv = (nv/self._y_std)**2
            self.gp_model.likelihood.variance[:] = nv
            self.gp_model.likelihood.variance.fix()


        if self.optimize_global:
            try:
                self.gp_model.optimize(messages=False)
            except Exception as exc:
                print(f'GLOBAL OPTIMIZE FAILED. ERROR: {exc}')

        # Fit noise GP if repeats exist
        if np.any(counts > 1) and self.num_repeats > 1:
            self.fit_noise(X, noise_vars, se_vars)


        self.cache_hypers()
        self.cache_K()

    def fit_noise(self, X=None, noise_vars=None, se_vars=None):
        if X is None or noise_vars is None or se_vars is None:
            X, _, noise_vars, se_vars, _, _ = self._get_unitXY_with_noise()

        # Convert variance ± error to std ± error
        std, std_err = self.var_to_std(noise_vars, se_vars)

        # --- Independent normalization for noise GP ---
        self._noise_mean = float(np.mean(std))
        self._noise_std  = float(np.std(std))
        if self._noise_std == 0:
            self._noise_std = 1.0

        std_norm     = (std - self._noise_mean) / self._noise_std
        std_err_norm = std_err / self._noise_std

        kernel = GPy.kern.RBF(input_dim=X.shape[1], ARD=True)
        self.noise_gp = GPy.models.GPHeteroscedasticRegression(X, std_norm.reshape(-1, 1), kernel)

        # Per-point training noise (fixed), normalized
        self.noise_gp.likelihood.variance[:] = (std_err_norm.reshape(-1, 1))**2
        self.noise_gp.likelihood.variance.fix()

        try:
            self.noise_gp.optimize(messages=False)
        except Exception as exc:
            print(f'GLOBAL NOISE OPTIMIZE FAILED. ERROR: {exc}')

    def var_to_std(self, var, var_err):
        """
        Convert variance ± error into std ± error using first‑order error propagation.

        Parameters
        ----------
        var : float or array
            Estimated variance.
        var_err : float or array
            Standard error (or std) of the variance estimate.

        Returns
        -------
        std : float or array
            Standard deviation = sqrt(var)
        std_err : float or array
            Error on the standard deviation
        """
        var = np.asarray(var)
        var_err = np.asarray(var_err)

        std = np.sqrt(var)
        std_err = var_err / (2 * std)

        return std, std_err

    def predict_noise(self, X_test):
        if not hasattr(self, 'noise_gp') or self.noise_gp is None:
            raise RuntimeError("Noise GP not fitted. Call fit() first with repeats.")

        X_unit_test = self.to_unit(X_test)
        pred_std_mean_norm, pred_std_var_norm = self.noise_gp.predict_noiseless(X_unit_test)

        # Rescale using noise normalization
        pred_std_mean = pred_std_mean_norm.flatten() * self._noise_std + self._noise_mean
        pred_std_err  = np.sqrt(np.maximum(pred_std_var_norm.flatten(), 0.0)) * self._noise_std
        return pred_std_mean, pred_std_err

    def predict_single_run_error(self, X_test):
        """
        Approximate the error of a single run at X_test:
        single_run_error ≈ 2 * (predicted noise std + its predictive error)
        This corresponds to ~97% Gaussian coverage (≈ 2σ) with a conservative bump.
        """
        pred_std, pred_err = self.predict_noise(X_test)
        return 2.0 * (pred_std + pred_err)

    def get_next_samples(self, batch_dir=None, *args, **kwargs):
        # ---------------------------
        # Ensure pool
        # ---------------------------
        desired_pool_min = kwargs.get('desired_pool_min', max(1000, 5 * self.batch_size))
        self._ensure_pool_size(desired_pool_min)
        if len(self.pool) < self.batch_size:
            warnings.warn('POOL SIZE IS SMALLER THAN BATCH SIZE SO SAMPLER ENDING EARLY BY RETURNING NONE')
            return None

        # ---------------------------
        # Initialisation
        # ---------------------------
        
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

        


        print('SELECTING NEW SAMPLES FROM POOL')
        # ---------------------------
        # Select new samples
        # ---------------------------
        samples = []
        if self.n_ensembles == 1:
            print(f'GETTING GLOBAL SCORE MODE:{self.acquisition_mode}')
            # Predictive variance on pool using global model (not used for fold selection but useful fallback)
            score_pool_global = self._compute_acquisition(self.pool, mode=self.acquisition_mode, blend_string=self.blend_string, y_pool = self.pool_y)
            score_pool_global = score_pool_global.flatten()
            
            print(f'BATCH SIZE:{self.batch_size} USING GLOBAL SCORE.')
            idx = list(np.argsort(-score_pool_global)[:self.batch_size])
            chosen_points = self.pool[idx]

            real_chosen_points = self.from_unit(chosen_points)
            samples = [{key: float(v) for key, v in zip(self.parameters, row)} for row in real_chosen_points]
            self.pool = np.delete(self.pool, idx, axis=0)
            if self.pool_y is not None:
                self.pool_y = np.delete(self.pool_y, idx, axis=0)
            
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
                
                scores = self._compute_acquisition(self.pool, mode=self.acquisition_mode, blend_string=self.blend_string, model=model_fold, y_pool = self.pool_y)
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
                score_pool_global = self._compute_acquisition(self.pool, mode=self.acquisition_mode, blend_string=self.blend_string, y_pool = self.pool_y)
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
            if self.pool_y is not None:
                self.pool_y = np.delete(self.pool_y, chosen_indices_final, axis=0)
            

        samples = samples * self.num_repeats
        if self.include_index:
            samples = [{**samp, 'index': ind} for samp, ind in zip(samples, range(self.custom_submitted,self.custom_submitted+len(samples)))]
        
        # increment counters and return
        if self.custom_submitted >= self.budget:
            current_batch_dir = os.path.join(self.base_run_dir, f'batch_{self.batch_number}')
            # Remove if empty
            try:
                os.rmdir(current_batch_dir)
                print(f"Removed empty directory: {current_batch_dir}")
            except OSError as e:
                print(f"Error: {e}")

            print('DOING LIGHT POST PROCESSING FROM SAPLER')
            print('plotting slices')
            self.plot_slices()
            self.post_process()
            return None

        if samples is not None:
            self.custom_submitted += len(samples)
        self.batch_number += 1
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
    
    def surrogate_predict(self, X_test):
        X_unit_test = self.to_unit(X_test)
        mean, var = self.gp_model.predict_noiseless(X_unit_test)
        if self.normalize_y:
            mean = mean.flatten() * self._y_std + self._y_mean
            var  = var.flatten() * (self._y_std**2)
        return mean, np.sqrt(var)

    # def surrogate_predict(self, samples):
    #     """
    #     Accept samples in real bounds, scale to unit, predict with GP.
    #     Return mean and posterior std of the mean function.
    #     """
    #     if self.gp_model is None:
    #         self.fit()

    #     samples_unit = self.to_unit(samples)
    #     ypred, post_var = self.gp_model.predict_noiseless(samples_unit)
    #     return ypred.flatten(), np.sqrt(np.maximum(post_var.flatten(), 0.0))

        
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

        X, y = self._get_unitXY(normalize_y=self.normalize_y)
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
            except Exception as exc:
                print('EXCEPTION WHEN GETTING num:', exc)
                print('TRACEBACK: \n', traceback.format_exc())
                num = 0.0
            num = max(num, 0.0)
            sobol_first[self.parameters[j] + '_sobolF'] = float(num / var_pred) if var_pred > 0 else 0.0

        # since the model was trained on normalised y values, the mean and std will be for the normalised values and need to be converted back to real values

        if self.normalize_y:
            mu = mu * self._y_std + self._y_mean
            var_pred = var_pred * (self._y_std ** 2)

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
            if self.max_y:
                mask = y_pred <= self.max_y
                X_test = X_test[mask]
                y_test = y_test[mask]
                y_pred = y_pred[mask]
            print('debug, ytest n nans', np.isnan(y_test).sum())
            print('debug, ypred n nans', np.isnan(y_pred).sum())
            residuals = y_test - y_pred
            fig = plt.figure()
            plt.hexbin(y_test, residuals, gridsize=50, cmap='plasma', bins=None, mincnt=1)
            residuals_save_dir = os.path.join(self.base_run_dir, 'residuals_plots')
            if not os.path.exists(residuals_save_dir):
                os.mkdir(residuals_save_dir)
            fig.savefig(os.path.join(residuals_save_dir, f"N-{self.gp_model.X.shape[0]}_residuals.png"))
            plt.close(fig)

            fig = plt.figure()
            plt.hexbin(y_test, y_pred, gridsize=50, cmap='plasma', bins=None, mincnt=1)
            residuals_save_dir = os.path.join(self.base_run_dir, 'prediction_plots')
            if not os.path.exists(residuals_save_dir):
                os.mkdir(residuals_save_dir)
            fig.savefig(os.path.join(residuals_save_dir, f"N-{self.gp_model.X.shape[0]}_prediction.png"))
            plt.close(fig)

            print('debug n nans', np.isnan(residuals).sum())
            rmse = np.sqrt(np.nanmean((y_test - y_pred) ** 2))
            if self.num_repeats > 1:
                noise_pred, _ = self.predict_noise(X_test)
                msse = np.nanmean((residuals/noise_pred) ** 2)
                nnrmse = np.sqrt(np.nanmean(residuals ** 2 / noise_pred ** 2))
                regression_results = {f'rmse_{len(y_test)}-{self.test_data_name}':[rmse], f'msse_{len(y_test)}-{self.test_data_name}':[msse],f'nnrmse_{len(y_test)}-{self.test_data_name}':[nnrmse]}
            else:
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

    def _compute_acquisition(self, X_pool, mode="var", blend_string=None, model=None, X_train=None, Y_train=None, y_pool=None):
        print('debug compute acquisition')
        start = time.time()        
        if mode == "blend":
            print('debug mode blend')
            if blend_string is None:
                raise ValueError("blend_string must be provided for blend mode")
            
            blend = self._parse_blend_string(blend_string)
            print('debug blend, coeff, mode:', blend)
            
            total = np.zeros(len(X_pool))
            for coeff, m in blend:
                if len(X_pool) > self.chunk_size:
                    scores = self._compute_acquisition_chunked(X_pool, mode=m, chunk_size=self.chunk_size, model=model, X_train=X_train, Y_train=Y_train, y_pool=y_pool)
                    
                else:
                    scores = self._compute_acquisition_unchunked(X_pool, mode=m, model=model, X_train=X_train, Y_train=Y_train, y_pool=y_pool)
                scores = (scores - scores.mean()) / (scores.std() + 1e-12)
                total += coeff * scores

            end = time.time()
            print('COMPUTING ACQUISITION TOOK:',(end-start)/60, 'min', f'MODE:{mode}')
            all_scores = total

        else:
            print('debug mode:', mode)
            if len(X_pool) > self.chunk_size:
                all_scores = self._compute_acquisition_chunked(X_pool, mode=mode, chunk_size=self.chunk_size, model=model, X_train=X_train, Y_train=Y_train, y_pool=y_pool)
                
            else:
                print('debug, computing acquisition unchunked, mode', mode)
                all_scores = self._compute_acquisition_unchunked(X_pool, mode=mode, model=model, X_train=X_train, Y_train=Y_train, y_pool=y_pool)
            end = time.time()
            print('COMPUTING ACQUISITION TOOK:',(end-start)/60, 'min', f'MODE:{mode}')
        
        all_scores_norm = (all_scores - all_scores.mean()) / (all_scores.std() + 1e-12)

        return all_scores_norm

    def _compute_acquisition_unchunked(self, X_pool, mode, chunk_size=5000, model=None, X_train=None, Y_train=None, y_pool=None):
        if model is None:
            model = self.gp_model
        
        if mode == "random":
            print('ACQUISITION MODE: random')
            score = self.rng.uniform(0,1,len(X_pool))
        
        elif mode == "var" or mode == "variance":
            print('ACQUISITION MODE: var')
            mu, var = model.predict_noiseless(X_pool)
            score = var.flatten()
        
        elif mode == "rand1var1":
            print('ACQUISITION MODE: rand1var1')
            self.rand1 = not self.rand1
            if self.rand1:
                score = np.random.uniform(0,1,len(X_pool))
            else:
                mu, var = model.predict_noiseless(X_pool)
                score = var.flatten()
        
        elif mode == "noise_and_var":
            print('ACQUISITION MODE: noise_and_var')
            # GP posterior variance
            # Y_metadata = {'output_index': np.arange(X_pool.shape[0]).reshape(-1,1)}
            mu, var_model = self.gp_model.predict_noiseless(X_pool)
            # _, var_model = self.gp_model.predict(X_pool)
            var_model = var_model.flatten()

            # predicted noise std + error from noise_gp
            noise_std, noise_err = self.predict_noise(self.from_unit(X_pool))
            var_noise = (noise_std + noise_err)**2

            score = var_model + var_noise
        
        elif mode == "var_distpen":
            print('ACQUISITION MODE: var_distpen') 
            if X_train is None:
                X_train, Y_train, noise_vars, se_vars, se_mean, counts = self._get_unitXY_with_noise(normalize_y = self.normalize_y)

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

        elif mode == "dist":
            if X_train is None:
                X_train, Y_train, noise_vars, se_vars, se_mean, counts = self._get_unitXY_with_noise(normalize_y = self.normalize_y)

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
            penalty = 1.0 - np.clip(max_sim, 0.0, 1.0)
            score = penalty

        elif mode == "eigf": # doi: 10.1016/j.ress.2024.109945
            print('ACQUISITION MODE: eigf')
            if X_train is None and Y_train is None:
                X_train, Y_train, noise_vars, se_vars, se_mean, counts = self._get_unitXY_with_noise(normalize_y = self.normalize_y)
            if X_train is not None and Y_train is not None:
                pass
            else:
                raise RuntimeError(f'X_train and Y_train should be provided together only one was provided. X_train {type(X_train)}, Y_train {type(Y_train)}')

            from sklearn.neighbors import KDTree
            tree = KDTree(X_train)  # X_train: (n_samples, d)
            dist, idx = tree.query(X_pool, k=1)
            y_nearest = Y_train[idx[:, 0]]
            
            mu, var = model.predict_noiseless(X_pool)
            
            score = (mu - y_nearest)**2 + var

        elif mode == "vigf": # doi:10.1016/j.ress.2024.109945
            print('ACQUISITION MODE: vigf')
            if X_train is None and Y_train is None:
                X_train, Y_train, noise_vars, se_vars, se_mean, counts = self._get_unitXY_with_noise(normalize_y = self.normalize_y)
            if X_train is not None and Y_train is not None:
                pass
            else:
                raise RuntimeError(f'X_train and Y_train should be provided together only one was provided. X_train {type(X_train)}, Y_train {type(Y_train)}')

            from sklearn.neighbors import KDTree
            tree = KDTree(X_train)  # X_train: (n_samples, d)
            dist, idx = tree.query(X_pool, k=1)
            y_nearest = Y_train[idx[:, 0]]

            mu, var = model.predict_noiseless(X_pool)

            score = 4*var * ((mu-y_nearest)**2 + 2*var)

        elif mode == "noise":
            print('ACQUISITION MODE: noise')
            noise_std, noise_err = self.predict_noise(self.from_unit(X_pool))
            var_noise = (noise_std + noise_err)**2
            score = var_noise

        elif mode == "grad":
            print('ACQUISITION MODE: grad')
            # Ensure X_pool is 2D
            X_pool = np.atleast_2d(X_pool)

            # Predictive gradients for the whole pool
            dmu, _ = model.predictive_gradients(X_pool)

            # dmu shape is (N, input_dim, output_dim)
            # Take the norm across the input_dim axis
            grads = np.linalg.norm(dmu, axis=1).squeeze()
            score = grads

        elif mode == "intVar":
            print('ACQUISITION MODE: intVar')
            if X_train is None:
                X_train, Y_train, noise_vars, se_vars, se_mean, counts = self._get_unitXY_with_noise(normalize_y = self.normalize_y)

            score = self.integral_variance_reduction(X_pool, model=model, X_train=X_train)

        elif mode == "oracle":
            print('ACQUISITION MODE: oracle')
            if X_train is None and Y_train is None:
                X_train, Y_train, noise_vars, se_vars, se_mean, counts = self._get_unitXY_with_noise(normalize_y = self.normalize_y)
            if X_train is not None and Y_train is not None:
                pass
            else:
                raise RuntimeError(f'X_train and Y_train should be provided together only one was provided. X_train {type(X_train)}, Y_train {type(Y_train)}')

            if len(X_pool) > os.cpu_count()*100:
                warnings.warn('ORACLE ACQUISITION IS HEAVY AND REQUIRES MANY CORES. THE POOL IS MUCH LARGER THEN THE NUMBER OF CORES, POOL {len(X_pool)}, CORES {os.cpu_count}. PLEASE CONSIDER A SMALLER POOL.')
            assert y_pool is not None
            y_model, _ = self.gp_model.predict_noiseless(X_pool)
            
            if self.normalize_y:
                y_pool_norm = self.normalize_y(y_pool)
                orig_rmse = np.sqrt(np.mean((y_model.flatten() - y_pool_norm.flatten())**2))

            else:
                orig_rmse = np.sqrt(np.mean((y_model.flatten() - y_pool.flatten())**2)) 

            def new_point_rmse_decrease(x_new, y_new):
                # Ensure shapes
                x_new = np.asarray(x_new).reshape(1, -1)
                y_new = np.asarray(y_new).reshape(1, 1)

                # Append new point
                X = np.vstack([X_train, x_new])
                y = np.vstack([Y_train, y_new])

                # Build model with fixed hyperparameters
                new_model = self.make_fixed_hyperparam_copy(X, y)

                # Predict on pool
                y_model, _ = new_model.predict_noiseless(X_pool)

                # Compute RMSE
                if self.normalize_y:
                    rmse = np.sqrt(np.mean((y_model.flatten() - y_pool_norm.flatten())**2))

                else:
                    rmse = np.sqrt(np.mean((y_model.flatten() - y_pool.flatten())**2)) 

                decrease = orig_rmse - rmse
                return decrease


            from joblib import Parallel, delayed

            rmse_decrease = Parallel(n_jobs=-1)(
                delayed(new_point_rmse_decrease)(xi, yi)
                for xi, yi in zip(X_pool, y_pool_norm)
            )

            score = np.array(rmse_decrease)
            
            
        else:
            raise ValueError(f"Unknown acquisition mode: {mode}")
        
        return score

    def normalize_y(self, Y, reset = False):
        """
        Normalize Y using stored mean/std if available,
        otherwise compute and store them.

        Parameters
        ----------
        Y : array-like, shape (N,) or (N,1)

        Returns
        -------
        Y_norm : array, shape (N,1)
            Normalized Y = (Y - mean) / std
        """
        Y = np.asarray(Y).reshape(-1, 1)

        # Compute and store normalization stats if not already set
        if self._y_mean is None or self._y_std is None or reset:
            self._y_mean = float(np.mean(Y))
            self._y_std  = float(np.std(Y))
            if self._y_std == 0:
                self._y_std = 1.0

        # Apply normalization
        Y_norm = (Y - self._y_mean) / self._y_std
        return Y_norm


    def _compute_acquisition_chunked(self, X_pool, mode, chunk_size=5000, model=None, X_train=None, Y_train=None, y_pool=None):
        results = []
        for i in range(0, len(X_pool), chunk_size):
            block = X_pool[i:i+chunk_size]
            results.append(self._compute_acquisition_unchunked(block, mode=mode, model=model, X_train=X_train, Y_train=Y_train, y_pool=y_pool))
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


    def integral_variance_reduction(self, X_pool, model=None, X_train=None):
        if X_train is None:
            X_train, _ = self._get_unitXY()
        
        if model is None:
            model = self.gp_model
        
        noise_variances = model.likelihood.variance[:]
    
        n_train = X_train.shape[0]

        # Kernel matrix and inverse
        K = model.kern.K(X_train) + np.diag(np.maximum(noise_variances, 1e-8))
        K_inv = np.linalg.pinv(K)

        # Kernel mean features for training points
        phi_train = self._integral_k_over_domain(X_train)  # shape (n_train,)

        # Kernel mean features for all pool points
        phi_pool = self._integral_k_over_domain(X_pool)    # shape (n_pool,)

        # Cross‑covariance between train and pool
        K_cross = model.kern.K(X_train, X_pool)    # shape (n_train, n_pool)

        # Self‑covariance for pool points
        K_self = np.diag(model.kern.K(X_pool))     # shape (n_pool,)

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

        print('Post Proc: Getting collapsed data')
        # Get collapsed data
        X_unit, Y_unique, noise_vars, se_vars, se_Y, counts = self._get_unitXY_with_noise()
        X_real = self.from_unit(X_unit)
        print('Post Proc: Calculating Stats')
        # Compute statistics
        mean = Y_unique.flatten()
        se_mean = se_Y
        std, se_std = self.var_to_std(noise_vars, se_vars)

        print('Post Proc: writing to file')
        # Build dataframe
        df = pd.DataFrame(X_real, columns=self.parameters)
        df['count'] = counts
        df['mean'] = mean
        df['std'] = std
        df['se_mean'] = se_mean
        df['se_std'] = se_std

        # Write CSV
        out_path = os.path.join(self.base_run_dir, self.output_col+'_enchanted_dataset_collapsed.csv')
        df.to_csv(out_path, index=False)
        print(f"Collapsed dataset written to {out_path}")
        df_avg_noise = pd.DataFrame({'mean_noise': [np.mean(df['std'])]})
        df_avg_noise.to_csv(os.path.join(self.base_run_dir, self.output_col+'_average_noise.csv'))


    def set_output_col(self):
        if self.output_col is None:
            data_df = pd.read_csv(os.path.join(self.base_run_dir, 'enchanted_dataset.csv'), on_bad_lines='warn', nrows=0)
            output_col = [col for col in data_df.columns if 'output' in col]
            if len(output_col)>1:
                warnings.warn(f'WHEN SETTING OUTPUT COL THERE WERE MORE THAN ONE COLUMNS WITH output STRING: {output_col}, TAKING FIRST: {output_col[0]}')
            self.output_col = output_col[0]
        return self.output_col

    def plot_slices(self):
        from enchanted_surrogates.samplers.slices_sampler_2d import SlicesSampler2D
        save_dir = os.path.join(self.base_run_dir, 'slice_plots')
        os.makedirs(save_dir, exist_ok=True)
        dim = len(self.parameters)
        budget = (dim*(dim-1) / 2) * self.slices_res**2
        # res = int(budget / (dim*(dim-1) / 2))
        slice_samp = SlicesSampler2D(parameters=self.parameters, bounds=self.bounds, base_run_dir=save_dir, res=self.slices_res, budget=budget)
        samples = slice_samp.get_samples()
        df = pd.DataFrame(samples)
        X_slice = df[self.parameters].to_numpy()
        Y_slice, Y_error = self.surrogate_predict(X_slice)
        print('debug Y_slice', Y_slice, Y_error)
        Y_slice_noise, Y_slice_noise_error = self.predict_noise(X_slice)
        print('debug len y len df', len(Y_slice), len(df))
        df_plot = pd.DataFrame(samples)
        if self.output_col is None:
            self.set_output_col()
        
        df_plot[self.output_col+'_noise_output'] = Y_slice_noise
        df_plot[self.output_col.replace('output','')+'_noise_outerror'] = Y_slice_noise_error
        slice_samp.plot_full_grid(df=df_plot, name=f'{self.output_col}_gpy_noise_N{self.gp_model.X.shape[0]*self.num_repeats}_', dots_x=self.from_unit(self.noise_gp.X))

        df_plot = pd.DataFrame(samples)
        if not 'output' in self.output_col:
            out_label = self.output_col+'_output'
        else:
            out_label = self.output_col
        df_plot[out_label] = Y_slice
        df_plot[out_label.replace('output', '')+'_outerror'] = Y_error
        slice_samp.plot_full_grid(df=df_plot, name=f'{self.output_col}_gpy_N{self.gp_model.X.shape[0]*self.num_repeats}_', dots_x=self.from_unit(self.gp_model.X))

    def plot_threshold_histograms_grid(self, threshold, res=20, bins=20):
        from enchanted_surrogates.samplers.grid_sampler import GridSampler
        gs = GridSampler(parameters=self.parameters, bounds=self.bounds, num_samples=res)
        samples = gs.get_next_samples()
        df = pd.DataFrame(samples)
        X = df[self.parameters].to_numpy()
        print('debug len df len X', len(df), len(X))
        Y, _ = self.surrogate_predict(X)
        print('debug len df len Y', len(df), len(Y))
        if self.output_col is None:
            self.set_output_col()
        print('debug len df len Y', len(df), len(Y))
        df[self.output_col] = Y
        self.plot_threshold_histograms(df, threshold, bins=bins)
        
    def plot_threshold_histograms(self, df, threshold, bins=20):
        """
        Plot mirrored histograms for each parameter:
        - Above threshold: counts shown upwards
        - Below threshold: counts shown inverted downwards

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing columns for self.parameters and self.output_col.
        threshold : float
            Threshold value for the output column.
        bins : int
            Number of bins for the histogram.
        """
        # if self.output_col is None:
        #     raise RuntimeError("Output column not set. Call get_data() first.")

        if self.output_col is None:
            self.set_output_col()

        mask_above = df[self.output_col] > threshold
        mask_below = ~mask_above

        n_params = len(self.parameters)
        fig, axes = plt.subplots(n_params, 1, figsize=(7, 3 * n_params), constrained_layout=True)

        if n_params == 1:
            axes = [axes]

        for ax, param in zip(axes, self.parameters):
            values_above = df.loc[mask_above, param]
            values_below = df.loc[mask_below, param]

            # Histogram for above threshold
            ax.hist(values_above, bins=bins, color='steelblue', edgecolor='black', alpha=0.7, label=f'>{threshold}')

            # Histogram for below threshold (inverted)
            counts, bin_edges = np.histogram(values_below, bins=bins)
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            ax.bar(bin_centers, -counts, width=(bin_edges[1]-bin_edges[0]),
                color='salmon', edgecolor='black', alpha=0.7, label=f'≤{threshold}')

            ax.axhline(0, color='black', linewidth=1)
            ax.set_title(f"Histogram of {param} relative to threshold {threshold}")
            ax.set_xlabel(param)
            ax.set_ylabel("Count (above vs. below)")
            ax.legend()
        num_train = self.gp_model.X.shape[0] * self.num_repeats
        save_dir = os.path.join(self.base_run_dir, 'threshold_histograms')
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, f'N{num_train}_thresh{threshold}_hist.png'))
        plt.close(fig)

    def oracle_test(self, train_df=None, train_csv=None,
                    test_df=None, test_csv=None,
                    max_steps=None, random_seed=0):
        """
        Oracle vs Random active learning test on a fully labeled dataset.
        Hyperparameters are optimized ONCE on the full training set, then frozen.

        Parameters
        ----------
        train_df, test_df : pandas DataFramex§
            Number of acquisition steps to simulate.
        random_seed : int
            Seed for random baseline.

        Returns
        -------
        oracle_rmse : list
        random_rmse : list
        fig : matplotlib Figure
        """

        import matplotlib.pyplot as plt
        from sklearn.metrics import mean_squared_error
        import numpy as np
        import pandas as pd
        import GPy
        
        print('performing oracle test')
        
        # ----------------------------
        # Load data
        # ----------------------------
        
        print('loading data')
        if train_df is None and train_csv is not None:
            train_df = pd.read_csv(train_csv)
        if test_df is None and test_csv is not None:
            test_df = pd.read_csv(test_csv)

        if train_df is None and test_df is None:
            raise ValueError("Provide either train_df/test_df or train_csv/test_csv")

        print('debug test_df is None', test_df is None)
        # If only one dataset provided → split 50/50
        if test_df is None:
            df = train_df.copy()
            df = df.sample(frac=1.0, random_state=random_seed)
            mid = len(df) // 2
            train_df = df.iloc[:mid].reset_index(drop=True)
            test_df = df.iloc[mid:].reset_index(drop=True)

        # ----------------------------
        # Identify input/output columns
        # ----------------------------
        out_cols = [c for c in train_df.columns if "output" in c.lower()]
        if len(out_cols) != 1:
            raise ValueError("Exactly one output column required containing 'output'")
        ycol = out_cols[0]

        X_train_full = train_df[self.parameters].to_numpy()
        y_train_full = train_df[ycol].to_numpy().reshape(-1, 1)

        X_test = test_df[self.parameters].to_numpy()
        y_test = test_df[ycol].to_numpy().reshape(-1, 1)

        if max_steps is None:
            max_steps = len(X_train_full) - 1
            
        print('max steps:', max_steps)
        # ----------------------------
        # Step 1: Optimize hyperparameters ONCE on full training set
        # ----------------------------
        print('optimizing hypers')
        X_unit_full = self.to_unit(X_train_full)
        kernel = GPy.kern.RBF(input_dim=X_unit_full.shape[1], ARD=True)
        model_full = GPy.models.GPRegression(X_unit_full, y_train_full, kernel)

        try:
            model_full.optimize(messages=False)
        except Exception:
            pass

        # Freeze optimized hyperparameters
        kernel_fixed = GPy.kern.RBF(input_dim=X_unit_full.shape[1], ARD=True)
        kernel_fixed.variance = float(model_full.kern.variance.values[0])
        kernel_fixed.lengthscale = model_full.kern.lengthscale.values.copy()
        self.lengthscales = kernel_fixed.lengthscale
        self.kernel_variance = kernel_fixed.variance

        # ----------------------------
        # Helper: fit GP with fixed hypers and compute RMSE
        # ----------------------------
        def fit_and_rmse(Xtr, Ytr, do_optimize=False):
            # Scale training data
            X_unit = self.to_unit(Xtr)

            # Build model with fixed kernel copy
            model = GPy.models.GPRegression(X_unit, Ytr, kernel_fixed.copy())

            # Optional optimization
            if do_optimize:
                try:
                    model.optimize(messages=False)
                except Exception as e:
                    print("Warning: GP optimization failed:", e)

            # Predict on test set
            X_test_unit = self.to_unit(X_test)
            mu, _ = model.predict(X_test_unit)
            
            print('debug len(X_test)', len(X_test))

            # Compute RMSE
            rmse = np.sqrt(mean_squared_error(y_test, mu))
            return rmse, model

        # ----------------------------
        # Initial seed: 1 random point
        # ----------------------------
        rng = np.random.RandomState(random_seed)
        idx0 = rng.choice(len(X_train_full), size=1, replace=False)

        X_oracle = X_train_full[idx0].copy()
        y_oracle = y_train_full[idx0].copy()

        X_random = X_train_full[idx0].copy()
        y_random = y_train_full[idx0].copy()

        active = ['rand1var1', 'var_distpen', 'eigf', 'vigf'] #, 'intVar', 'blend']
        blend_string = [
            None if i != 'blend' else '0.33-var_0.33-grad_0.34-intVar'
            for i in active
        ]

        # !!!!!!!!!!!!!!!!!!!! Needs normalisation
        X_active = [X_train_full[idx0].copy() for i in active]
        y_active = [y_train_full[idx0].copy() for i in active]
        Xpool_active = [X_train_full.copy() for i in active]
        ypool_active = [y_train_full.copy() for i in active]
        
        pool_idx_oracle = np.setdiff1d(np.arange(len(X_train_full)), idx0)
        pool_idx_random = np.setdiff1d(np.arange(len(X_train_full)), idx0)
        oracle_rmse = []
        random_rmse = []
        active_rmse = [[] for i in active]

        # ----------------------------
        # Main oracle loop
        # ----------------------------
        print('running main loop')
        for step in range(max_steps):
            start_time = time.time()
            print(f'Running step: {step}/{max_steps}')
            # ----------------------------
            # ORACLE SELECTION
            # ----------------------------

            from joblib import Parallel, delayed
            def evaluate_index(i, X_oracle, y_oracle, X_train_full, y_train_full):
                X_try = np.vstack([X_oracle, X_train_full[i:i+1]])
                y_try = np.vstack([y_oracle, y_train_full[i:i+1]])
                rmse_i, model = fit_and_rmse(X_try, y_try, do_optimize=True)
                return rmse_i, i

            # Run in parallel
            results = Parallel(n_jobs=-1)(
                delayed(evaluate_index)(i, X_oracle, y_oracle, X_train_full, y_train_full)
                for i in pool_idx_oracle
            )
            
            # Find best
            best_rmse, best_idx = min(results, key=lambda x: x[0])

            # Update oracle set
            X_oracle = np.vstack([X_oracle, X_train_full[best_idx:best_idx+1]])
            y_oracle = np.vstack([y_oracle, y_train_full[best_idx:best_idx+1]])

            pool_idx_oracle = pool_idx_oracle[pool_idx_oracle != best_idx]

            rmse_oracle, model = fit_and_rmse(X_oracle, y_oracle, do_optimize=True)
            oracle_rmse.append(rmse_oracle)

            # ----------------------------
            # RANDOM SELECTION
            # ----------------------------
            if len(pool_idx_random) == 0:
                break

            ridx = rng.choice(pool_idx_random, size=1)[0]
            X_random = np.vstack([X_random, X_train_full[ridx:ridx+1]])
            y_random = np.vstack([y_random, y_train_full[ridx:ridx+1]])

            pool_idx_random = pool_idx_random[pool_idx_random != ridx]

            rmse_random, model = fit_and_rmse(X_random, y_random, do_optimize=True)
            random_rmse.append(rmse_random)

            # ----------------------------
            # ACTIVE ACQUISTIONS
            # ----------------------------
            for i, act in enumerate(active):
                if step == 0:
                    rmse, model = fit_and_rmse(X_random, y_random, do_optimize=True)
                else:
                    rmse, model = fit_and_rmse(X_active[i], y_active[i], do_optimize=True)
                # print('debug model', model)
                scores = self._compute_acquisition(Xpool_active[i], mode=act, blend_string=blend_string[i], model=model, X_train=X_active[i], Y_train=y_active[i])
                print('debug, mode, scores 10', act, scores[:10])
                best_idx = np.argmax(scores)
                print('debug, mode, best idx', best_idx)
                
                # Update active set
                X_active[i] = np.vstack([X_active[i], Xpool_active[i][best_idx:best_idx+1]])
                y_active[i] = np.vstack([y_active[i], ypool_active[i][best_idx:best_idx+1]])
    
                # Option 1: using np.delete with axis=0
                Xpool_active[i] = np.delete(Xpool_active[i], best_idx, axis=0)
                ypool_active[i] = np.delete(ypool_active[i], best_idx, axis=0)

                rmse, model = fit_and_rmse(X_active[i], y_active[i], do_optimize=True) 
                active_rmse[i].append(rmse)

            # ----------------------------
            # Plot
            # ----------------------------
            print('plotting')
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(range(1, len(oracle_rmse)+1), oracle_rmse, label="Oracle", lw=2)
            ax.plot(range(1, len(random_rmse)+1), random_rmse, label="Random", lw=2)
            for i, act in enumerate(active):
                if act == 'blend':
                    ax.plot(range(1, len(active_rmse[i])+1), active_rmse[i], label=act, lw=2)
                else:
                    ax.plot(range(1, len(active_rmse[i])+1), active_rmse[i], label=act, lw=2, linestyle='--')
            ax.set_xlabel("Number of samples")
            ax.set_ylabel("Test RMSE")
            ax.set_title("Oracle vs Random Active Learning")
            ax.legend()
            ax.grid(True)
            fig.savefig(os.path.join(self.base_run_dir, 'oracle_vs_random.png'))
            plt.close(fig)
            print(f'Step {step} completed in {(time.time() - start_time)/60:.2f} min | Oracle RMSE: {rmse_oracle:.4f} | Random RMSE: {rmse_random:.4f}')
            print(f'estimated time left: {((time.time() - start_time)/60)*(max_steps - step):.2f} min')

        print('FINNISHED ORACLE TEST')
        
        return oracle_rmse, random_rmse, fig


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
    for i in range(len(batch_dirs)):
        print('debug _batch dirs', batch_dirs[i])
            
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
    gpy.batch_number=4
    outputs = ['ascot_runtime_min','lost_power_W_output','number_particles','current_drive_A','power_deposited_to_ions_W','power_deposited_to_electrons_W']
    for out in outputs:
        gpy.output_col = out
        gpy.fit()
        gpy.fit_noise()
        gpy.plot_slices()
        gpy.post_process()
    # last_write=0
    # write_every=500
    # for i, batch_dir in enumerate(batch_dirs):
    #     gpy.batch_number = i
    #     # if os.path.exists(os.path.join(batch_dir, 'gpy_model.pkl')):
    #     print('WRITING BATCH INFO FOR:',batch_dir)
    #     # with open(os.path.join(batch_dir, 'gpy_model.pkl'), 'rb') as file:
    #         # gpy.gp_model = pickle.load(file)
    #         # X, Y = gpy.get_data()
    #         # if len(X)<=520:
    #         #     continue
    #     X, Y = gpy.get_data()
    #     num_samples = len(X)
    #     if num_samples - last_write <= write_every:
    #         continue
    #     else:
    #         last_write = num_samples
    #     gpy.fit()
    #     gpy.cache_hypers()
    #     gpy.cache_K()
    #     gpy.plot_slices(res=30)
    #     # gpy._write_batch_info_inner(batch_dir, name='post2_')
        
    #     # reg_results = gpy.regression_test()
    #     # reg_results['num_samples'] = gpy.gp_model.X.shape[0]
    #     # print('debug reg results', reg_results)
    #     # df = pd.DataFrame({k: v for k, v in reg_results.items()})
    #     # reg_path = os.path.join(os.path.dirname(batch_dir), 'regression_info.csv')
    #     # if os.path.exists(reg_path):
    #     #     df.to_csv(reg_path, mode='a', header=False, index=False)
    #     # else:
    #     #     df.to_csv(reg_path, mode='w', header=True, index=False)
    #     # print("\n\n\n ================================== \n\n\n")
            
         
    #     # if i == 2:
    #     #     break

    # print('MAKING COLLAPSED DATASET')
    # for bn in np.linspace(1,207,10):
    #     gpy.batch_number = int(bn)
    #     gpy.fit()
    #     gpy.plot_slices()
    # gpy.plot_threshold_histograms_grid(threshold=0.01, res=30, bins=100)
    # gpy.post_process()
 
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
