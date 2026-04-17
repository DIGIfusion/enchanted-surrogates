import os
import math
import pickle
import warnings
import time
import shutil
import traceback

import math

from joblib import Parallel, delayed
from sklearn.neighbors import KDTree
from scipy.stats import norm

from enchanted_surrogates.samplers.base_sampler import Sampler

from gpytorch.utils.cholesky import psd_safe_cholesky

from gpytorch.constraints import Interval

from linear_operator.utils.errors import NotPSDError

import numpy as np
import pandas as pd
from scipy.special import erf
from sklearn.model_selection import KFold # Still useful for ensemble logic if desired
import torch
from torch import nanmean # Using torch.nanmean for robust mean
from torch.nn import ModuleList

import gpytorch
from gpytorch.models import ExactGP
from gpytorch.means import ConstantMean
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.kernels import (
    ScaleKernel, MaternKernel, RBFKernel,
    RQKernel
)
from gpytorch.distributions import MultivariateNormal

from enchanted_surrogates.utils.logger import get_logger
log = get_logger(__name__)


# --- Analytical 1D Integrals (adapted for PyTorch where possible, maintaining math for non-standard) ---
# These functions will operate on single scalar inputs for xi (and xj) as they are 1D.
# When used with batch inputs, we'll map/vectorize.

import time

def time_format(seconds: int) -> str:
    """
    Convert a duration in seconds into 'Xd HH:MM:SS' format.
    Days are included only when nonzero.
    """
    days = seconds // 86400
    t = time.gmtime(seconds)

    if days > 0:
        return f"{days}d {t.tm_hour:02d}:{t.tm_min:02d}:{t.tm_sec:02d}"
    else:
        return f"{t.tm_hour:02d}:{t.tm_min:02d}:{t.tm_sec:02d}"


def _get_alpha_from_matern_nu(nu, lengthscale):
    if nu == 0.5:
        return 1.0 / lengthscale
    elif nu == 1.5:
        return math.sqrt(3.0) / lengthscale
    elif nu == 2.5:
        return math.sqrt(5.0) / lengthscale
    else:
        raise ValueError(f"Matern {nu} not supported for analytical integral.")

def exp_1d_integral_between(xi, lengthscale, a, b):
    # xi, lengthscale, a, b are scalars
    l = float(lengthscale) # Ensure float for math lib
    xi_f = float(xi)
    a_f = float(a)
    b_f = float(b)
    # Using float() to ensure compatibility with math module
    return (
        2 * l
        - l * math.exp(-(xi_f - a_f) / l)
        - l * math.exp(-(b_f - xi_f) / l)
    )

def exp_1d_double_integral(xi, xj, lengthscale, a, b):
    # xi, xj, lengthscale, a, b are scalars
    l = float(lengthscale)
    xi_f = float(xi)
    xj_f = float(xj)
    a_f = float(a)
    b_f = float(b)
    r = abs(xi_f - xj_f)
    return (
        2 * l * math.exp(-r / l)
        - l * math.exp(-(r + 2 * (b_f - a_f)) / l)
    )

def matern32_1d_integral_between(xi, lengthscale, a, b):
    # xi, lengthscale, a, b are scalars
    l = float(lengthscale)
    xi_f = float(xi)
    a_f = float(a)
    b_f = float(b)
    alpha = math.sqrt(3) / l

    def F(t_val):
        t_val_f = float(t_val)
        return (
            (2 / alpha)
            - (1 + alpha * t_val_f) * math.exp(-alpha * t_val_f) / alpha
        )
    return F(b_f - xi_f) + F(xi_f - a_f)

def matern32_1d_double_integral(xi, xj, lengthscale, a, b):
    # xi, xj, lengthscale, a, b are scalars
    l = float(lengthscale)
    xi_f = float(xi)
    xj_f = float(xj)
    a_f = float(a)
    b_f = float(b)

    alpha = math.sqrt(3) / l
    r = abs(xi_f - xj_f)
    term1 = (2 / alpha + 4 / (alpha**3)) * math.exp(-alpha * r)
    term2 = (1 / alpha + 2 / (alpha**3)) * math.exp(-alpha * (r + 2 * (b_f - a_f)))
    return term1 - term2

def matern52_1d_integral_between(xi, lengthscale, a, b):
    # xi, lengthscale, a, b are scalars
    l = float(lengthscale)
    xi_f = float(xi)
    a_f = float(a)
    b_f = float(b)

    alpha = math.sqrt(5) / l

    def P(t_val):
        t_val_f = float(t_val)
        return 1 + alpha * t_val_f + (alpha**2) * (t_val_f**2) / 3

    def F(t_val):
        t_val_f = float(t_val)
        return (
            (2 / alpha)
            + (4 / (3 * alpha**3))
            - P(t_val_f) * math.exp(-alpha * t_val_f) / alpha
        )
    return F(b_f - xi_f) + F(xi_f - a_f)

def matern52_1d_double_integral(xi, xj, lengthscale, a, b):
    # xi, xj, lengthscale, a, b are scalars
    l = float(lengthscale)
    xi_f = float(xi)
    xj_f = float(xj)
    a_f = float(a)
    b_f = float(b)

    alpha = math.sqrt(5) / l
    r = abs(xi_f - xj_f)

    # Compact closed form:
    A = (2 / alpha + 4 / (3 * alpha**3) + 4 / (15 * alpha**5))
    B = (1 / alpha + 2 / (3 * alpha**3) + 2 / (15 * alpha**5))
    return A * math.exp(-alpha * r) - B * math.exp(-alpha * (r + 2 * (b_f - a_f)))

def gaussian_1d_integral_between(xi, lengthscale, a, b):
    # xi, lengthscale, a, b are scalars
    s = float(lengthscale)
    xi_f = float(xi)
    a_f = float(a)
    b_f = float(b)

    coeff = math.sqrt(math.pi / 2.0) * s
    return coeff * (
        erf((b_f - xi_f) / (math.sqrt(2.0) * s)) -
        erf((a_f - xi_f) / (math.sqrt(2.0) * s))
    )

def gaussian_1d_double_integral(xi, xj, lengthscale, a, b):
    # xi, xj, lengthscale, a, b are scalars
    s = float(lengthscale)
    xi_f = float(xi)
    xj_f = float(xj)
    a_f = float(a)
    b_f = float(b)

    pref = math.exp(-((xi_f - xj_f) ** 2) / (4.0 * s ** 2))
    s_eff = s / math.sqrt(2.0)
    coeff = math.sqrt(math.pi / 2.0) * s_eff
    mu_eff = 0.5 * (xi_f + xj_f)
    return pref * coeff * (
        erf((b_f - mu_eff) / (math.sqrt(2.0) * s_eff)) -
        erf((a_f - mu_eff) / (math.sqrt(2.0) * s_eff))
    )

# Note: RatQuad integrals are more complex and rely on `mpmath`.
# For a streamlined PyTorch version, we might skip RatQuad or use numerical integration
# if `mpmath` is not easily integrated with PyTorch tensors or is too slow.
# For now, I'll include the original Python `mpmath` functions.
def rq_1d_integral_between(xi, lengthscale, alpha, a, b):
    # Requires mpmath, so not easily PyTorch-native
    import mpmath as mp

    def z(x_val):
        return ( (x_val-xi)**2 ) / ( (x_val-xi)**2 + 2*alpha*(lengthscale**2) )

    def F_rq(x_val_f):
        return (
            math.sqrt(2*alpha)*lengthscale *
            float(mp.betainc(0.5, alpha-0.5, 0, z(x_val_f)))
        )
    return F_rq(b) - F_rq(a)

def rq_1d_double_integral(xi, xj, lengthscale, alpha, a, b):
    # Requires mpmath, so not easily PyTorch-native
    import mpmath as mp

    def k_rq(x_val, c_val):
        return (1 + (x_val-c_val)**2/(2*alpha*lengthscale**2))**(-alpha)

    f = lambda x_val: k_rq(x_val, xi)*k_rq(x_val, xj)
    return float(mp.quad(f, [a, b]))

# Map for kernels used in analytical integrals
KERNEL_INTEGRAL_MAP = {
    "RBF": {
        "single": gaussian_1d_integral_between,
        "double": gaussian_1d_double_integral,
        "gpytorch_class": RBFKernel,
        "nu": None
    },
    "Exponential": {
        "single": exp_1d_integral_between,
        "double": exp_1d_double_integral,
        "gpytorch_class": MaternKernel, # Matern with nu=0.5
        "nu": 0.5
    },
    "Matern32": {
        "single": matern32_1d_integral_between,
        "double": matern32_1d_double_integral,
        "gpytorch_class": MaternKernel,
        "nu": 1.5
    },
    "Matern52": {
        "single": matern52_1d_integral_between,
        "double": matern52_1d_double_integral,
        "gpytorch_class": MaternKernel,
        "nu": 2.5
    },
    "RatQuad": {
        "single": rq_1d_integral_between,
        "double": rq_1d_double_integral,
        "gpytorch_class": RQKernel,
        "nu": None # RQ doesn't have nu parameter in same way as Matern
    },
}

class GPyTorchGPR(ExactGP):
    """
    A clean, self-contained GPyTorch GP model:
    - Stores input bounds (x_min, x_max)
    - Stores output normalization (y_mean, y_std)
    - Provides real <-> unit conversion
    - Uses ARD kernels
    """

    def __init__(self, train_x, train_y, likelihood, kernel_type="Matern52",
                x_min=None, x_max=None):

        # ---------------------------------------------------------
        # 1. Compute and store X normalization BEFORE super()
        # ---------------------------------------------------------
        if x_min is None:
            x_min = train_x.min(dim=0).values
        if x_max is None:
            x_max = train_x.max(dim=0).values

        # Avoid zero-range dimensions
        x_range = x_max - x_min
        x_range[x_range < 1e-12] = 1.0

        # Normalize X to unit space
        train_x_unit = (train_x - x_min) / x_range

        # ---------------------------------------------------------
        # 2. Compute and store Y normalization BEFORE super()
        # ---------------------------------------------------------
        y_mean = train_y.mean()
        y_std = train_y.std()
        if y_std < 1e-12:
            y_std = torch.tensor(1.0, dtype=train_y.dtype, device=train_y.device)

        train_y_standardized = (train_y - y_mean) / y_std

        # ---------------------------------------------------------
        # 3. Call ExactGP with normalized inputs and standardized outputs
        # ---------------------------------------------------------
        super().__init__(train_x_unit, train_y_standardized.squeeze(-1), likelihood)

        input_dim = train_x.size(-1)

        # ---------------------------------------------------------
        # 4. Register buffers AFTER super()
        # ---------------------------------------------------------
        self.register_buffer("x_min", x_min.clone())
        self.register_buffer("x_max", x_max.clone())
        self.register_buffer("x_range", x_range.clone())

        self.register_buffer("y_mean", y_mean.clone())
        self.register_buffer("y_std", y_std.clone())

        # ---------------------------------------------------------
        # 5. Kernel + mean
        # ---------------------------------------------------------
        self.mean_module = ConstantMean()
        self.covar_module = self._create_kernel(kernel_type, input_dim)
    # ---------------------------------------------------------
    # Kernel factory
    # ---------------------------------------------------------
    def _create_kernel(self, kernel_type, input_dim):
        if kernel_type == "RBF":
            return ScaleKernel(RBFKernel(ard_num_dims=input_dim))
        elif kernel_type == "Exponential":
            return ScaleKernel(MaternKernel(nu=0.5, ard_num_dims=input_dim))
        elif kernel_type == "Matern32":
            return ScaleKernel(MaternKernel(nu=1.5, ard_num_dims=input_dim))
        elif kernel_type == "Matern52":
            return ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=input_dim))
        elif kernel_type == "RatQuad":
            warnings.warn("RQKernel alpha handling differs from GPy.")
            return ScaleKernel(RQKernel(ard_num_dims=input_dim))
        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}")

    # ---------------------------------------------------------
    # Forward pass
    # ---------------------------------------------------------
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    # -------------------------
    # X transforms
    # -------------------------
    def to_unit(self, X_real):
        return (X_real - self.x_min) / self.x_range

    def to_real(self, X_unit):
        return X_unit * self.x_range + self.x_min

    # -------------------------
    # Y transforms
    # -------------------------
    def standardize_y(self, y_real):
        return (y_real - self.y_mean) / self.y_std

    def unstandardize_y(self, y_std):
        return y_std * self.y_std + self.y_mean

    def unstandardize_var(self, var_std):
        return var_std * (self.y_std ** 2)




class GpyTorchActiveSampler(Sampler):
    """
    A sampler using GPyTorch for GPR.
    
    Experimental:
        uq_analysis
            GPR integral over the space
            GPR variance over the space
            GPR sobol indices 
    """
    def __init__(self, **kwargs):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.double # Use double precision for GP calculations
        self.write_batch_every = kwargs.get('write_batch_every', 10)
        self.plot_residuals_every = kwargs.get('plot_residuals_every', 20)
        self.parameters = kwargs.get('parameters')
        self.bounds = kwargs.get('bounds')
        self.pool_csv_path = kwargs.get('pool_csv_path', None)
        self.pool_chunk_size = kwargs.get('pool_chunk_size', 1000)
        self.fixed_pool_values = kwargs.get('fixed_pool_values', None)
        self.allowed_pool_values = kwargs.get('allowed_pool_values', None)
        
        self.output_col = kwargs.get('output_col', None)
        self.output_column_name = self.get_output_col(csv_path=self.pool_csv_path)
        
        self.base_run_dir = kwargs.get('base_run_dir', None)
        if self.base_run_dir is None:
            raise ValueError("base_run_dir must be specified in config")
        self.seed = kwargs.get("seed", None)
        self._set_seed(self.seed)
        self.do_uq_analysis = kwargs.get('do_uq_analysis', False) # it seems to be flawed, always getting matrix is not positive definite when making L, current 'fix' causes the code to stall take forever

        os.makedirs(self.base_run_dir, exist_ok=True) # Ensure run directory exists

        if self.bounds == 'set_with_pool_csv' and self.pool_csv_path:
            df = pd.read_csv(self.pool_csv_path)
            self.bounds = []
            for p in self.parameters:
                p_min = df[p].min()
                p_max = df[p].max()
                self.bounds.append([p_min, p_max])
            print('BOUNDS:', self.bounds)

        if self.parameters is None or self.bounds is None:
            raise ValueError('parameters and bounds must be provided')
        if len(self.parameters) != len(self.bounds):
            raise ValueError('The number of bounds and parameters must match.')

        self._lb = torch.tensor([b[0] for b in self.bounds], dtype=self.dtype, device=self.device)
        self._ub = torch.tensor([b[1] for b in self.bounds], dtype=self.dtype, device=self.device)
        self._range = self._ub - self._lb
        self.input_dim = len(self.parameters)

        self.total_pool_size = kwargs.get('total_pool_size', 10000)
        # allowed_pool_values: dict mapping parameter name -> list/array of allowed REAL-space values
        self.allowed_pool_values = kwargs.get('allowed_pool_values', None)
        if self.allowed_pool_values is not None:
            # Normalize keys and convert lists to numpy arrays for fast sampling
            if not isinstance(self.allowed_pool_values, dict):
                raise ValueError('allowed_pool_values must be a dict mapping parameter->list_of_values')
            for k in list(self.allowed_pool_values.keys()):
                if k not in self.parameters:
                    raise ValueError(f"allowed_pool_values contains unknown parameter '{k}'")
                self.allowed_pool_values[k] = np.asarray(self.allowed_pool_values[k])

        # GPR related attributes
        self.gp_model: GPyTorchGPR = None
        self.likelihood: GaussianLikelihood = None
        self.train_x = torch.empty(0, self.input_dim, dtype=self.dtype, device=self.device)
        self.train_y = torch.empty(0, 1, dtype=self.dtype, device=self.device)
        self.optimizer = None # Will be initialized during fit
        self.mll = None # Marginal Log Likelihood

        self.kernel_type = kwargs.get('kernel_type', 'RBF')
        self.fix_noise = kwargs.get('fix_noise', False)
        self.noise_variance = kwargs.get('noise_variance', 1e-8) # Initial noise variance

        # Sampler configuration
        self.acquisition_mode = kwargs.get('acquisition_mode', 'variance')
        # Batch selection strategy when batch_size > 1: 'best_score' picks top-K from
        # a single scoring of the pool; 'fantasy_labels' greedily selects one-by-one
        # and fantasizes the GP label (using the GP predictive mean) before
        # selecting the next point.
        self.acquisition_batch_mode = kwargs.get('acquisition_batch_mode', 'best_score')
        self.batch_size = kwargs.get('batch_size', 2)
        self.initial_batch_size = kwargs.get('initial_batch_size', self.batch_size)
        self.num_repeats = kwargs.get('num_repeats', 1)
        self.batch_number = 0
        self.submitted_samples = 0
        self.prev_submitted_samples = 0
        self.budget = kwargs.get('budget', float('inf')) # Total samples to acquire

        self.test_data_csv = kwargs.get('test_data_csv', None)
        self.clean_pool_csv = False
        if self.pool_csv_path is None:
            self._init_pool_stream_random()
            self.clean_pool_csv = True

        self._init_pool_stream_csv()
            
        self.removed_indices = set()
        self._next_row_index = 0  # increments as we stream

    def to_unit_torch(self, X_real_t):
        """Real → unit (torch)."""
        return (X_real_t - self._lb) / self._range

    def from_unit_torch(self, X_unit_t):
        """Unit → real (torch)."""
        return self._lb + X_unit_t * self._range

    def to_unit_numpy(self, X_real_np):
        """Real → unit (numpy)."""
        lb = self._lb.cpu().numpy()
        rng = self._range.cpu().numpy()
        return (X_real_np - lb) / rng

    def from_unit_numpy(self, X_unit_np):
        """Unit → real (numpy)."""
        lb = self._lb.cpu().numpy()
        rng = self._range.cpu().numpy()
        return lb + X_unit_np * rng

    def get_output_col(self, df=None, csv_path=None):
        if self.output_col:
            return self.output_col
        if csv_path:
            df = pd.read_csv(csv_path)
        if df is not None:
            output_cols = [col for col in df.columns if 'output' in col]
            if len(output_cols) != 1:
                raise RuntimeError(f'Exactly one output column required but found: {output_cols}')
            return output_cols[0]
        raise ValueError("Cannot determine output column without df or csv_path.")

    def _set_seed(self, seed):
        import random
        import numpy as np
        import torch

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Make CUDA deterministic (optional but recommended)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def _init_pool_stream_random(self):
        log.info("Generating random pool CSV...")
        start_time = time.time()
        self._rng = np.random.default_rng(self.seed)

        # If no CSV path was provided, create one
        if self.pool_csv_path is None:
            self.pool_csv_path = os.path.join(os.path.dirname(self.base_run_dir), 'tmp', "generated_pool.csv")
            os.makedirs(os.path.dirname(self.pool_csv_path), exist_ok=True)

        # Create empty CSV with header
        df_header = pd.DataFrame(columns=self.parameters)
        df_header.to_csv(self.pool_csv_path, index=False)

        rows_remaining = self.total_pool_size

        while rows_remaining > 0:
            chunk_size = min(self.pool_chunk_size, rows_remaining)

            # Build chunk in REAL space
            chunk_real = np.zeros((chunk_size, self.input_dim), dtype=float)

            for param_index, param_name in enumerate(self.parameters):

                # allowed_pool_values
                if self.allowed_pool_values and param_name in self.allowed_pool_values:
                    allowed_values = np.asarray(self.allowed_pool_values[param_name])
                    chosen = self._rng.choice(allowed_values, size=chunk_size)
                    chunk_real[:, param_index] = chosen

                # fixed_pool_values
                elif self.fixed_pool_values and param_name in self.fixed_pool_values:
                    fixed_values = np.asarray(self.fixed_pool_values[param_name])
                    chosen = self._rng.choice(fixed_values, size=chunk_size)
                    chunk_real[:, param_index] = chosen

                # uniform sampling
                else:
                    low, high = self.bounds[param_index]
                    chunk_real[:, param_index] = self._rng.uniform(
                        float(low), float(high), size=chunk_size
                    )

            # Append chunk to CSV
            df_chunk = pd.DataFrame(chunk_real, columns=self.parameters)
            df_chunk.to_csv(self.pool_csv_path, mode="a", header=False, index=False)

            rows_remaining -= chunk_size
        end_time = time.time()
        
        log.info(f'Generating the random pool csv took: {time_format(end_time-start_time)}')
        
    def _init_pool_stream_csv(self):
        self._csv_iter = pd.read_csv(
            self.pool_csv_path,
            chunksize=self.pool_chunk_size
        )
        
    def get_next_pool_chunk(self):
        try:
            df = next(self._csv_iter)
        except StopIteration:
            return None, None, None

        # Assign global row indices for this chunk
        start_index = self._next_row_index
        end_index = start_index + len(df)
        chunk_indices = list(range(start_index, end_index))
        self._next_row_index = end_index

        # Filter out removed rows
        mask = [idx not in self.removed_indices for idx in chunk_indices]
        df = df[mask]
        chunk_indices = [idx for idx, keep in zip(chunk_indices, mask) if keep]

        # If everything was removed, fetch next chunk
        if df.empty:
            return self.get_next_pool_chunk()

        # Convert to unit space
        real_values = df[self.parameters].to_numpy()
        unit_values = self.to_unit_numpy(real_values)

        # Optional y column
        out_col = self.get_output_col(df=df)
        if out_col in df.columns:
            y_values = df[out_col].to_numpy()
        else:
            y_values = None

        return unit_values, y_values, chunk_indices


    def _remove_from_pool(self, pool_indices):
        self.removed_indices.update(pool_indices)

    def append_train_data(self, dataset_path=None):
        if dataset_path is None:
            dataset_path = os.path.join(self.base_run_dir, 'enchanted_dataset.csv')
        if not os.path.exists(dataset_path):
            print(f"Warning: Dataset path '{dataset_path}' not found. Skipping data append.")
            return

        new_df = pd.read_csv(dataset_path)
        out_col = self.get_output_col(df=new_df)

        clean_df = new_df[self.parameters + [out_col]].dropna()
        if clean_df.empty:
            print(f"Warning: No valid data found in '{dataset_path}' after dropping NaNs.")
            return

        # Convert to REAL-SPACE tensors
        X_real = torch.tensor(clean_df[self.parameters].values,
                            dtype=self.dtype, device=self.device)
        Y_real = torch.tensor(clean_df[out_col].values,
                            dtype=self.dtype, device=self.device).reshape(-1, 1)

        # ---------------------------------------------------------
        # FIRST TIME: initialize training set in REAL SPACE
        # ---------------------------------------------------------
        if self.train_x.numel() == 0:
            self.train_x = X_real
            self.train_y = Y_real
            print(f"Initialized training set with {len(X_real)} points.")
            return

        # ---------------------------------------------------------
        # Deduplicate in REAL SPACE
        # ---------------------------------------------------------
        diff = X_real.unsqueeze(1) - self.train_x.unsqueeze(0)
        is_same = (diff.abs() < 1e-9).all(dim=2)
        is_duplicate = is_same.any(dim=1)

        mask_new = ~is_duplicate
        num_new = mask_new.sum().item()

        if num_new > 0:
            self.train_x = torch.cat([self.train_x, X_real[mask_new]], dim=0)
            self.train_y = torch.cat([self.train_y, Y_real[mask_new]], dim=0)
            print(f"Appended {num_new} new unique training points.")
        else:
            print("No new unique training points found.")

    def fit_gpr_model(self, num_restarts=5):
        if self.train_x.numel() == 0:
            raise RuntimeError("Cannot fit GPR: No training data available.")
        if self.train_x.size(0) < 2:
            print("Not enough training data for GPR fit (need at least 2 points). Skipping fit.")
            return

        best_mll_value = -float("inf")
        best_model = None
        best_likelihood = None

        for restart in range(num_restarts):
            print(f"\n=== Restart {restart+1}/{num_restarts} ===")

            likelihood = GaussianLikelihood(
                noise_constraint=gpytorch.constraints.GreaterThan(1e-8)
            ).to(self.dtype).to(self.device)

            if self.fix_noise:
                likelihood.noise = self.noise_variance
                likelihood.raw_noise.requires_grad_(False)

            # Model computes y_mean/y_std internally
            model = GPyTorchGPR(
                self.train_x,
                self.train_y.squeeze(-1),
                likelihood,
                kernel_type=self.kernel_type,
                x_min=self._lb,
                x_max=self._ub
            ).to(self.dtype).to(self.device)

            # Lengthscale bounds
            model.covar_module.base_kernel.lengthscale_constraint = Interval(0.0, 2.0)

            # Randomize ARD lengthscales
            with torch.no_grad():
                D = self.train_x.shape[1]
                rand_ls = 0.1 + 0.9 * torch.rand(D, dtype=self.dtype, device=self.device)
                model.covar_module.base_kernel.lengthscale.copy_(rand_ls)

            model.train()
            likelihood.train()

            optimizer = torch.optim.LBFGS(
                model.parameters(),
                lr=0.5,
                max_iter=50,
                line_search_fn="strong_wolfe"
            )

            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

            # Use the model's internally standardized training targets
            y_std = model.train_targets

            train_x_unit = model.to_unit(self.train_x)
            def closure():
                optimizer.zero_grad()
                output = model(train_x_unit)
                loss = -mll(output, y_std)
                loss.backward()
                return loss

            optimizer.step(closure)

            # Evaluate MLL
            model.eval()
            likelihood.eval()
            with torch.no_grad():
                output = model(train_x_unit)
                mll_value = mll(output, y_std).item()

            print(f"Restart {restart+1}: MLL = {mll_value:.6f}")

            if mll_value > best_mll_value:
                best_mll_value = mll_value
                best_model = model
                best_likelihood = likelihood

        # Store best model
        self.gp_model = best_model
        self.likelihood = best_likelihood

        print("\n=== Best model selected ===")
        print(f"  Best MLL: {best_mll_value:.6f}")
        print(f"  Noise: {self.likelihood.noise.item():.4e}")
        print(f"  Outputscale: {self.gp_model.covar_module.outputscale.item():.4f}")
        for i, p in enumerate(self.parameters):
            print(f"  Lengthscale ({p}): {self.gp_model.covar_module.base_kernel.lengthscale.squeeze()[i].item():.4f}")

    def surrogate_predict(self, X_real):
        X_real_t = torch.tensor(X_real, device=self.device, dtype=self.dtype)
        X_unit = self.gp_model.to_unit(X_real_t)

        self.gp_model.eval()
        self.likelihood.eval()

        with torch.no_grad():
            posterior = self.gp_model(X_unit)
            mean_unit = posterior.mean
            var_unit = posterior.variance

        mean_real = self.gp_model.unstandardize_y(mean_unit)
        var_real = self.gp_model.unstandardize_var(var_unit)

        return mean_real.cpu().numpy(), var_real.cpu().numpy()
        
    def _get_normalized_hypers(self):
        """Extracts normalized hyperparameters from the fitted model."""
        if self.gp_model is None:
            raise RuntimeError("GP model not fitted.")
        
        with torch.no_grad():
            kernel = self.gp_model.covar_module
            outputscale = kernel.outputscale.item()
            lengthscales_unit = kernel.base_kernel.lengthscale.squeeze().cpu().numpy()
            noise_variance = self.likelihood.noise.item()
            
            # The lengthscales in GPyTorch's ARD (Automatic Relevance Determination) kernels
            # are typically directly associated with the normalized input space [0,1].
            # So, they are already "normalized" in that sense.
            # If we need lengthscales in *real* space, we'd multiply by self._range.
            # However, the analytical integrals expect lengthscales relative to the integration range [0,1],
            # so the model's lengthscales are the correct ones to use.
            
        return outputscale, lengthscales_unit, noise_variance

    def _make_1d_kernel_info(self, dim_idx):
        """
        Returns kernel name, outputscale, lengthscale for a specific dimension,
        and optionally nu/alpha for Matern/RatQuad.
        """
        kernel_map_entry = KERNEL_INTEGRAL_MAP.get(self.kernel_type)
        if not kernel_map_entry:
            raise NotImplementedError(f"Kernel type {self.kernel_type} not supported for analytical Sobol integrals.")
        
        outputscale, lengthscales_unit, _ = self._get_normalized_hypers()
        
        # GPyTorch lengthscales are typically `ard_num_dims=input_dim`
        # so we select the lengthscale for the current dimension.
        ls_d = lengthscales_unit[dim_idx]

        info = [self.kernel_type, outputscale, ls_d]
        if kernel_map_entry.get("nu") is not None:
            info.append(kernel_map_entry["nu"])
        if self.kernel_type == "RatQuad":
            # For RatQuad, need the alpha parameter (power).
            # Default botorch RQKernel's `alpha` is 1.0 (or a learnable parameter).
            # Assuming it's fixed or learned in the model.
            rq_alpha = self.gp_model.covar_module.base_kernel.alpha.item()
            info.append(rq_alpha)

        return info
    
    def _get_1d_integral_func(self, kernel_type, integral_type):
        """Retrieves the correct 1D integral function from the map."""
        entry = KERNEL_INTEGRAL_MAP.get(kernel_type)
        if not entry:
            raise ValueError(f"Kernel type {kernel_type} not defined in KERNEL_INTEGRAL_MAP.")
        func = entry.get(integral_type)
        if func is None:
            raise NotImplementedError(f"Integral type '{integral_type}' not implemented for kernel '{kernel_type}'.")
        return func

    def _integral_k_over_domain(self, X_unit: torch.Tensor):
        """
        Computes ∫ k(x,xi) dx for each xi in X_unit.
        Returns a tensor.
        """
        n, D = X_unit.shape
        # Integrals are over the unit hypercube [0,1]^D
        a, b = 0.0, 1.0

        # Initialize I as outputscale (variance) if using ScaleKernel, else 1.0
        outputscale, _, _ = self._get_normalized_hypers()
        I = torch.full((n,), float(outputscale), dtype=self.dtype, device=self.device)

        for d in range(D):
            xi_d_scalars = X_unit[:, d].cpu().numpy() # Convert to numpy for scalar loop in math-based integrals
            
            info = self._make_1d_kernel_info(d)
            name = info[0]
            ls = info[2] # lengthscale for this dimension

            single_integral_func = self._get_1d_integral_func(name, "single")

            # Handle RatQuad specifically for its `alpha` parameter
            if name == "RatQuad":
                rq_alpha = info[3] # alpha is the 4th element in info
                I_d_list = [single_integral_func(xi_val, ls, rq_alpha, a, b) for xi_val in xi_d_scalars]
            else:
                I_d_list = [single_integral_func(xi_val, ls, a, b) for xi_val in xi_d_scalars]
            
            I_d = torch.tensor(I_d_list, dtype=self.dtype, device=self.device)
            I *= I_d
        
        return I

    def _integral_kk_over_domain(self, X_unit: torch.Tensor):
        """
        Computes ∫ k(x,xi)k(x,xj) dx for all pairs (xi, xj) in X_unit.
        Returns an (n, n) tensor.
        """
        n, D = X_unit.shape
        a, b = 0.0, 1.0

        outputscale, _, _ = self._get_normalized_hypers()
        # Initial C contains (outputscale)^2 due to product of two kernels
        C = torch.full((n, n), float(outputscale**2), dtype=self.dtype, device=self.device)

        for d in range(D):
            xi_d_scalars = X_unit[:, d].cpu().numpy()
            
            info = self._make_1d_kernel_info(d)
            name = info[0]
            ls = info[2]

            double_integral_func = self._get_1d_integral_func(name, "double")

            C_d = torch.zeros((n, n), dtype=self.dtype, device=self.device)
            for i in range(n):
                for j in range(n):
                    if name == "RatQuad":
                        rq_alpha = info[3]
                        C_d[i, j] = double_integral_func(xi_d_scalars[i], xi_d_scalars[j], ls, rq_alpha, a, b)
                    else:
                        C_d[i, j] = double_integral_func(xi_d_scalars[i], xi_d_scalars[j], ls, a, b)
            C *= C_d
        return C

    def uq_analysis(self):
        """
        Experimental
        Performs UQ analysis and calculates Sobol main effect (first order) indices
        based on the GPyTorch model (with internal X/Y normalization).
        """
        print("\n\n ================================== \n")
        if self.train_x.numel() == 0:
            raise RuntimeError('No training data available for UQ analysis.')

        # Real-space training data
        X_train_real = self.train_x
        Y_train_real = self.train_y  # shape (N, 1)

        n, D = X_train_real.shape
        vol = 1.0  # unit hypercube volume

        self.gp_model.eval()
        self.likelihood.eval()

        # Unit-space training inputs (consistent with model internals)
        X_train_unit = self.gp_model.to_unit(X_train_real)

        with torch.no_grad(), gpytorch.settings.fast_computations(covar_root_decomposition=False):

            # Latent GP covariance (no likelihood noise)
            latent = self.gp_model(X_train_unit)
            K_full = latent.covariance_matrix  # (N, N)

            # Cholesky (with PSD repair if needed)
            try:
                L = psd_safe_cholesky(K_full)
            except NotPSDError:
                eigvals, eigvecs = torch.linalg.eigh(K_full)
                eigvals_clamped = torch.clamp(eigvals, min=1e-8)
                K_full_psd = eigvecs @ torch.diag(eigvals_clamped) @ eigvecs.mT
                L = torch.linalg.cholesky(K_full_psd)
                K_full = K_full_psd

            # Normalized targets (already standardized inside model)
            Y_norm = self.gp_model.train_targets.reshape(-1, 1)  # (N, 1)

            # Solve K_full * x = Y_norm
            K_inv_Y = torch.cholesky_solve(Y_norm, L)  # (N, 1)

            # ∫ k(x, x_i) dx over unit hypercube, using unit-space X
            I_x = self._integral_k_over_domain(X_train_unit)  # (N,)

            # Integral of posterior mean in normalized space
            integral_m_unit = (I_x.unsqueeze(0) @ K_inv_Y).squeeze()  # scalar
            mu_normalized = integral_m_unit

            # De-normalized mean
            mu = self.gp_model.unstandardize_y(mu_normalized).item()

            # ∫ k(x, x_i) k(x, x_j) dx over unit hypercube
            C_int_kk_unit = self._integral_kk_over_domain(X_train_unit)  # (N, N)

            # Helper: K^{-1} * vec via Cholesky
            def K_inv_solve_vec(vec):
                return torch.cholesky_solve(vec, L)

            # Full K^{-1}
            K_inv = K_inv_solve_vec(torch.eye(n, dtype=self.dtype, device=self.device))

            # A = K^{-1} C_int_kk K^{-1}
            A = K_inv @ C_int_kk_unit @ K_inv

            # Total variance in normalized space: Var[f] = Y^T A Y - mu^2
            var_pred_normalized = (Y_norm.T @ A @ Y_norm).squeeze() - mu_normalized**2
            var_pred_normalized = torch.clamp(var_pred_normalized, min=0.0).item()

            # De-normalize variance
            var_pred = var_pred_normalized * (self.gp_model.y_std**2).item()
            std_pred = math.sqrt(var_pred) if var_pred > 0 else 0.0

            # --- Sobol First Order Indices ---
            sobol_first = {}
            outputscale, lengthscales_unit, _ = self._get_normalized_hypers()

            for j in range(D):
                # Product of integrals over all dimensions except j
                prod_other_int_list = []
                for d_other in range(D):
                    if d_other == j:
                        continue

                    other_dim_info = self._make_1d_kernel_info(d_other)
                    other_name = other_dim_info[0]
                    other_ls = other_dim_info[2]

                    single_integral_func_other = self._get_1d_integral_func(other_name, "single")
                    xi_d_other_scalars = X_train_unit[:, d_other].cpu().numpy()

                    if other_name == "RatQuad":
                        rq_alpha_other = other_dim_info[3]
                        I_d_other_list = [
                            single_integral_func_other(xi_val, other_ls, rq_alpha_other, 0.0, 1.0)
                            for xi_val in xi_d_other_scalars
                        ]
                    else:
                        I_d_other_list = [
                            single_integral_func_other(xi_val, other_ls, 0.0, 1.0)
                            for xi_val in xi_d_other_scalars
                        ]

                    prod_other_int_list.append(
                        torch.tensor(I_d_other_list, dtype=self.dtype, device=self.device)
                    )

                prod_other = torch.ones(n, dtype=self.dtype, device=self.device)
                for vec in prod_other_int_list:
                    prod_other *= vec  # elementwise

                # 1D double integral for dimension j
                j_dim_info = self._make_1d_kernel_info(j)
                j_name = j_dim_info[0]
                j_ls = j_dim_info[2]
                double_integral_func_j = self._get_1d_integral_func(j_name, "double")
                xi_j_scalars = X_train_unit[:, j].cpu().numpy()

                Mj = torch.zeros((n, n), dtype=self.dtype, device=self.device)
                for i_Mj in range(n):
                    for k_Mj in range(n):
                        if j_name == "RatQuad":
                            rq_alpha_j = j_dim_info[3]
                            Mj[i_Mj, k_Mj] = double_integral_func_j(
                                xi_j_scalars[i_Mj], xi_j_scalars[k_Mj],
                                j_ls, rq_alpha_j, 0.0, 1.0
                            )
                        else:
                            Mj[i_Mj, k_Mj] = double_integral_func_j(
                                xi_j_scalars[i_Mj], xi_j_scalars[k_Mj],
                                j_ls, 0.0, 1.0
                            )

                outer_prod = torch.outer(prod_other, prod_other)
                B = (outputscale**2) * outer_prod * Mj

                num_normalized = (Y_norm.T @ K_inv @ B @ K_inv @ Y_norm).squeeze() - mu_normalized**2
                num_normalized = torch.clamp(num_normalized, min=0.0).item()

                if var_pred_normalized > 0:
                    sobol_first[self.parameters[j] + "_sobolF"] = num_normalized / var_pred_normalized
                else:
                    sobol_first[self.parameters[j] + "_sobolF"] = 0.0

        batch_info = {
            "mean": [mu],
            "std": [std_pred],
        }
        batch_info.update({k: [v] for k, v in sobol_first.items()})
        return batch_info

    def _has_enough_pool_points(self, required=None):
        """
        Returns True if the pool has at least `required` remaining points.
        If `required` is None, defaults to self.batch_size.
        """

        if required is None:
            required = self.batch_size

        # Total rows in CSV (excluding header), stored at initialization
        total = self.total_pool_size

        # Number of removed rows
        removed = len(self.removed_indices)

        # Remaining rows
        remaining = total - removed

        return remaining >= required


    def get_next_samples(self):
        """
        Main loop for acquiring new samples.
        """
        
        # 1. Update training data
        if self.batch_number == 0:
            # Initial random samples from pool
            chosen_unit = []
            chosen_indices = []
            num_initial_pool_chunks = math.ceil(self.initial_batch_size / self.pool_chunk_size)
            for i in range(num_initial_pool_chunks):
                unit_X, _, pool_indices = self.get_next_pool_chunk()
                chosen_unit.append(unit_X)
                chosen_indices.append(pool_indices)
            
            chosen_indices = np.concatenate(chosen_indices, axis=0)
            chosen_unit = np.concatenate(chosen_unit, axis=0)
            chosen_unit = chosen_unit[:self.initial_batch_size] # Trim to initial batch size
            
            self._remove_from_pool(chosen_indices) # Remove chosen points from pool
        else:
            log.info(f"Acquisition batch {self.batch_number}: Updating training data and fitting GP...")
            self.append_train_data()
            # Need at least two training points to fit a GP
            print('debug', 'Training data size:', self.train_x.size(0), 'points.')
            if self.train_x.numel() == 0 or self.train_x.size(0) < 2:
                raise RuntimeError("Not enough training data to fit GP for acquisition.")
            else:
                # 2. Fit/Update GPR model
                # The `SingleTaskGP` from botorch wraps GPyTorch ExactGP and handles transforms
                # For `SingleTaskGP` we pass real-domain X and Y, and it sets up transforms
                # However, for Sobol integration, we need hyperparam values *in the unit space*.
                # So we will continue to use our custom GPyTorchGPR and handle normalization explicitly for the GP fit first.
                self.fit_gpr_model()
                
                if not self._has_enough_pool_points():
                    remaining = self.total_pool_size - len(self.removed_indices)
                    warnings.warn(
                        f"Pool has only {remaining} remaining points, "
                        f"but batch size is {self.batch_size}. Cannot acquire new samples."
                    )
                    self.light_post_processing()
                    return None
                else:
                    remaining = self.total_pool_size - len(self.removed_indices)
                    # 3. Compute acquisition function over pool
                    start_time = time.time()
                    chosen_unit_indices = self._compute_acquisition_candidates(self.acquisition_mode)
                    end_time = time.time()
                    log.info(f'Computing the acquisition function over the {remaining} pool took: {time_format(end_time-start_time)}')
                    chosen_unit = self._get_unit_points_by_global_indices(chosen_unit_indices)
                    self._remove_from_pool(chosen_unit_indices)
        
        # check if we need more samples?
        if self.submitted_samples >= self.budget:
            print("Budget reached. No more samples will be acquired.")
            self.write_batch_info() # Final write at end of sampling
            self.light_post_processing()
            return None

        samples = []
        if chosen_unit is not None and len(chosen_unit) > 0:
            real_chosen_points = self.from_unit_numpy(chosen_unit)
            samples = [
                {key: float(v) for key, v in zip(self.parameters, row)}
                for row in real_chosen_points
            ] * self.num_repeats # Repeat if num_repeats > 1 (original feature kept)

        self.batch_number += 1
        num_new_samples = len(samples)

        self.prev_submitted_samples = self.submitted_samples
        self.submitted_samples += num_new_samples
        
        # Detect crossing of any multiple of write_batch_every
        if (self.prev_submitted_samples // self.write_batch_every) != (self.submitted_samples // self.write_batch_every) and self.batch_number > 1:
            start = time.time()
            self.write_batch_info()
            end = time.time()
            print('write batch info took:',(start-end)/60, 'minutes')
        # Detect crossing of any multiple of plot_residuals_every
        if (self.prev_submitted_samples // self.plot_residuals_every) != (self.submitted_samples // self.plot_residuals_every) and self.batch_number > 1:
            self.residuals_plot()
            self.residuals_plot(train=True,name='train_')

        
        if self.submitted_samples > self.budget:
            # If we overshot the budget, trim samples
            samples = samples[:num_new_samples - (self.submitted_samples - self.budget)]
            self.submitted_samples = self.budget
        
        if not samples:
            self.light_post_processing()
            return None # No more samples

        return samples

    def _compute_acquisition_candidates(self, mode):
        """
        Fully streaming acquisition over the entire pool.
        Uses get_next_pool_chunk() to iterate through the CSV-backed pool.
        Returns global pool indices of selected candidates.
        """
        # Reset pool streaming before scoring
        self._reset_csv_iterator()

        batch_size = int(self.batch_size)

        # For fantasy-label mode: global mask
        masked_global = set()

        # -------------------------
        # Streaming scoring
        # -------------------------
        def stream_scores(model):
            """
            Yields:
                scores: np.array of shape (chunk_size,)
                pool_indices: list of global indices for this chunk
            """
            while True:
                result = self.get_next_pool_chunk()
                
                X_unit_chunk, y_chunk, pool_indices = result

                if X_unit_chunk is None:
                    return

                # Convert to numpy

                scores = self._compute_acquisition_unchunked(
                    X_unit_chunk, mode, model=model, pool_y=y_chunk
                )

                yield scores, pool_indices

        # -------------------------
        # Streaming top‑K heap
        # -------------------------
        import heapq
        def top_k_stream(model):
            heap = []  # (score, global_idx)

            for scores, pool_indices in stream_scores(model):
                for score, global_idx in zip(scores, pool_indices):
                    if global_idx in self.removed_indices:
                        continue
                    if len(heap) < batch_size:
                        heapq.heappush(heap, (score, global_idx))
                    else:
                        if score > heap[0][0]:
                            heapq.heapreplace(heap, (score, global_idx))

            return np.array([idx for (_, idx) in sorted(heap)])

        # -------------------------
        # Streaming argmax (fantasy mode)
        # -------------------------
        def argmax_stream(model):
            best_score = -np.inf
            best_global_idx = None

            for scores, pool_indices in stream_scores(model):
                for score, global_idx in zip(scores, pool_indices):
                    if global_idx in self.removed_indices:
                        continue
                    if global_idx in masked_global:
                        continue
                    if score > best_score:
                        best_score = score
                        best_global_idx = global_idx

            return best_global_idx

        # ============================================================
        # CASE 1: K=1 or best_score mode → simple top‑K
        # ============================================================
        if batch_size <= 1 or self.acquisition_batch_mode == "best_score":
            return top_k_stream(self.gp_model)

        # ============================================================
        # CASE 2: fantasy_labels (sequential greedy)
        # ============================================================
        if self.acquisition_batch_mode == "fantasy_labels":
            selected_global = []
            current_model = self.gp_model

            for _ in range(batch_size):
                best_global_idx = argmax_stream(current_model)
                if best_global_idx is None:
                    break

                selected_global.append(best_global_idx)
                masked_global.add(best_global_idx)

                # Build fantasy label
                x_sel_unit = self._get_unit_points_by_global_indices([best_global_idx])[0] # shape (D,)
                x_sel_unit_t = torch.tensor(
                    x_sel_unit, dtype=self.dtype, device=self.device
                )

                with torch.no_grad():
                    posterior = current_model(x_sel_unit_t)
                    mu_std = posterior.mean

                x_sel_real = current_model.to_real(x_sel_unit_t).detach()
                y_fantasy_real = current_model.unstandardize_y(mu_std).reshape(-1, 1).detach()

                # Augment training data
                X_aug = torch.cat([
                    self.train_x.to(self.dtype).to(self.device),
                    x_sel_real.to(self.dtype).to(self.device)
                ], dim=0)

                Y_aug = torch.cat([
                    self.train_y.to(self.dtype).to(self.device),
                    y_fantasy_real.to(self.dtype).to(self.device)
                ], dim=0)

                # Build new fantasized model
                likelihood_tmp = GaussianLikelihood().to(self.dtype).to(self.device)
                model_tmp = GPyTorchGPR(
                    X_aug,
                    Y_aug.squeeze(-1),
                    likelihood_tmp,
                    kernel_type=self.kernel_type,
                    x_min=self._lb,
                    x_max=self._ub,
                ).to(self.dtype).to(self.device)

                # Copy hyperparameters
                try:
                    model_tmp.load_state_dict(self.gp_model.state_dict(), strict=False)
                except Exception:
                    pass
                try:
                    likelihood_tmp.load_state_dict(self.likelihood.state_dict())
                except Exception:
                    pass

                model_tmp.eval()
                likelihood_tmp.eval()
                current_model = model_tmp

            return np.array(selected_global)

        # ============================================================
        # Fallback: top‑K
        # ============================================================
        return top_k_stream(self.gp_model)

    def _reset_csv_iterator(self):
        self._csv_iter = pd.read_csv(self.pool_csv_path, chunksize=self.pool_chunk_size)
        self._next_row_index = 0

    def _get_unit_points_by_global_indices(self, indices):
        """
        Given a list/array of global pool indices, return the corresponding
        unit-space points as a NumPy array of shape (N, D).

        Fully streaming: reads the CSV in chunks and extracts only the needed rows.
        """

        target = set(indices)
        collected = []
        found = 0

        # Reset CSV iterator
        self._reset_csv_iterator()

        while True:
            result = self.get_next_pool_chunk()
            
            X_unit_chunk, _, pool_indices = result

            if X_unit_chunk is None:
                break

            # pool_indices is a list of global indices for this chunk
            for local_i, global_i in enumerate(pool_indices):
                if global_i in target:
                    collected.append(X_unit_chunk[local_i])
                    found += 1
                    if found == len(target):
                        # Found all requested points
                        return np.stack(collected, axis=0)

        # If we get here, something is wrong
        raise RuntimeError(
            f"Requested indices {indices} not fully found in pool."
        )

    def _compute_acquisition_unchunked(self, X_pool_unit, mode, model=None, pool_y=None):
        """
        Compute acquisition scores for X_pool_unit (unit space).
        All acquisitions (except oracle RMSE) are computed in standardized GP space.
        NO CHUNKING HERE — chunking is handled in _compute_acquisition_candidates.
        """

        if model is None:
            model = self.gp_model

        # Ensure tensor in unit space
        if isinstance(X_pool_unit, np.ndarray):
            X_pool_unit = torch.tensor(X_pool_unit, dtype=self.dtype, device=self.device)
        else:
            X_pool_unit = X_pool_unit.to(self.dtype).to(self.device)

        N = X_pool_unit.shape[0]

        # Normalized training data from the model
        X_train_unit = model.train_inputs[0]                    # (N_train, D)
        Y_train_std = model.train_targets.reshape(-1, 1)        # (N_train, 1)

        # Posterior in standardized space
        model.eval()
        with torch.no_grad():
            posterior = model(X_pool_unit)
            mu_std = posterior.mean          # (N_pool,)
            var_std = posterior.variance     # (N_pool,)

        mu_std_np = mu_std.cpu().numpy().flatten()
        var_std_np = var_std.cpu().numpy().flatten()

        # ============================================================
        # SIMPLE MODES
        # ============================================================

        if mode == "random":
            return np.random.rand(N)

        if mode == "var":
            # standardized variance
            return var_std_np

        if mode == "intVar":
            # your analytic routine already expects unit-space X
            return self.integral_variance_reduction(X_pool_unit.cpu().numpy())

        # ============================================================
        # DISTANCE-PENALIZED VARIANCE (standardized var)
        # ============================================================

        if mode == "var_distpen":
            K = model.covar_module.base_kernel(X_pool_unit, X_train_unit).evaluate().cpu().numpy()
            max_sim = np.clip(np.max(K, axis=1), 0.0, 1.0)
            return var_std_np * (1.0 - max_sim)

        if mode == "distpen":
            K = model.covar_module.base_kernel(X_pool_unit, X_train_unit).evaluate().cpu().numpy()
            max_sim = np.clip(np.max(K, axis=1), 0.0, 1.0)
            return 1.0 - max_sim

        # ============================================================
        # EIGF / VIGF (fully in standardized space)
        # ============================================================

        if mode in ("eigf", "vigf"):
            X_train_np = X_train_unit.cpu().numpy()
            Y_train_std_np = Y_train_std.cpu().numpy().flatten()

            tree = KDTree(X_train_np)
            _, idx = tree.query(X_pool_unit.cpu().numpy(), k=1)
            y_nearest_std = Y_train_std_np[idx[:, 0]]

            if mode == "eigf":
                return (mu_std_np - y_nearest_std)**2 + var_std_np

            if mode == "vigf":
                return 4 * var_std_np * ((mu_std_np - y_nearest_std)**2 + 2 * var_std_np)

        # ============================================================
        # MAX PREDICTED MEAN (standardized)
        # ============================================================

        if mode == "maxPred":
            return mu_std_np

        # ============================================================
        # EXPECTED IMPROVEMENT (standardized)
        # ============================================================

        if mode == "eim":
            # best observed in standardized space
            f_best_std = np.max(Y_train_std.cpu().numpy())
            sigma = np.sqrt(np.maximum(var_std_np, 1e-12))
            Z = (mu_std_np - f_best_std) / sigma
            ei_std = (mu_std_np - f_best_std) * norm.cdf(Z) + sigma * norm.pdf(Z)
            return np.maximum(ei_std, 0.0)

        # ============================================================
        # ORACLE MODE (RMSE REDUCTION IN REAL SPACE)
        # ============================================================

        if mode in ("oracle", "oracle_rmse"):
            if pool_y is None:
                raise RuntimeError("Oracle mode requires pool_y (true outputs for pool points).")
            if not hasattr(self, "X_test_real") or not hasattr(self, "Y_test_real"):
                raise RuntimeError("Oracle mode requires self.X_test_real and self.Y_test_real.")

            # Baseline RMSE in REAL space
            mu_base, _ = self.surrogate_predict(self.X_test_real)
            rmse_base = np.sqrt(np.mean((mu_base.flatten() - self.Y_test_real.flatten())**2))

            pool_y_np = np.asarray(pool_y).flatten()

            def _oracle_single(i):
                x_new_real = model.to_real(X_pool_unit[i].unsqueeze(0))
                y_new_real = pool_y_np[i]

                X_aug = torch.cat([self.train_x, x_new_real], dim=0)
                Y_aug = torch.cat([
                    self.train_y,
                    torch.tensor([[y_new_real]], dtype=self.dtype, device=self.device)
                ], dim=0)

                likelihood_tmp = GaussianLikelihood().to(self.dtype).to(self.device)
                model_tmp = GPyTorchGPR(
                    X_aug,
                    Y_aug.squeeze(-1),
                    likelihood_tmp,
                    kernel_type=self.kernel_type,
                    x_min=self._lb,
                    x_max=self._ub
                ).to(self.dtype).to(self.device)

                model_tmp.train()
                likelihood_tmp.train()

                optimizer = torch.optim.LBFGS(
                    model_tmp.parameters(),
                    lr=0.3,
                    max_iter=20,
                    line_search_fn="strong_wolfe"
                )
                mll_tmp = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood_tmp, model_tmp)
                y_std_tmp = model_tmp.train_targets

                def closure_tmp():
                    optimizer.zero_grad()
                    out = model_tmp(model_tmp.train_inputs[0])
                    loss = -mll_tmp(out, y_std_tmp)
                    loss.backward()
                    return loss

                optimizer.step(closure_tmp)

                mu_after, _ = self._surrogate_predict_with_model(model_tmp, likelihood_tmp, self.X_test_real)
                rmse_after = np.sqrt(np.mean((mu_after.flatten() - self.Y_test_real.flatten())**2))

                return rmse_base - rmse_after

            scores = Parallel(n_jobs=-1, backend="loky")(
                delayed(_oracle_single)(i) for i in range(N)
            )
            return np.array(scores)

        raise ValueError(f"Unknown acquisition mode: {mode}")


    def _calculate_rmse(self, model, X_test, Y_test):
        """Calculates RMSE for a given model and test set."""
        with torch.no_grad():
            preds = model.posterior(X_test).mean
            rmse = torch.sqrt(torch.nanmean((preds - Y_test) ** 2)).item()
        return rmse

    def _calculate_ipv(self, model, X_test):
        """Calculates Integrated Posterior Variance (IPV) for a given model and test set."""
        with torch.no_grad():
            _, variance = model.posterior(X_test).mean_var
            ipv = torch.nanmean(variance).item()
        return ipv

    def regression_test(self):
        if not self.test_data_csv or self.gp_model is None:
            return None

        test_df = pd.read_csv(self.test_data_csv)
        out_col = self.get_output_col(df=test_df)

        X_test_real = test_df[self.parameters].values
        Y_test_real = test_df[out_col].values

        y_pred, _ = self.surrogate_predict(X_test_real)  # denormalized predictions

        residuals = Y_test_real - y_pred
        rmse = np.sqrt(np.nanmean(residuals ** 2))

        # MAPE
        eps = 1e-12
        mape = np.nanmean(np.abs(residuals / (Y_test_real + eps)))

        # R-squared
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((Y_test_real - np.mean(Y_test_real)) ** 2)
        r2 = 1 - ss_res / (ss_tot + eps)

        regression_results = {
            f"rmse_{len(Y_test_real)}-test": [rmse],
            f"mape_{len(Y_test_real)}-test": [mape],
            f"r2_{len(Y_test_real)}-test": [r2],
        }
        return regression_results


    def save_model(self, save_dir, name=''):
        """Saves current GPyTorch model state dictionary."""
        os.makedirs(save_dir, exist_ok=True)
        if self.gp_model:
            model_path = os.path.join(save_dir, name + 'gpytorch_model.pth')
            # Save state dict to keep it light
            torch.save(self.gp_model.state_dict(), model_path)
            # You might also want to save the likelihood state_dict:
            torch.save(self.likelihood.state_dict(), os.path.join(save_dir, name + 'gpytorch_likelihood.pth'))
            print(f"GPyTorch model saved to {model_path}")
        else:
            print("No GPyTorch model to save.")

    def load_gp_model(self, model_path, likelihood_path):
        """Loads GPyTorch model and likelihood state dictionaries."""

        # Create empty likelihood
        likelihood = GaussianLikelihood().to(self.dtype).to(self.device)

        # Create an empty model with correct structure
        model = GPyTorchGPR(
            torch.empty(0, self.input_dim, dtype=self.dtype, device=self.device),
            torch.empty(0, dtype=self.dtype, device=self.device),
            likelihood,
            kernel_type=self.kernel_type
        ).to(self.dtype).to(self.device)

        # Load model weights
        if os.path.exists(model_path):
            state = torch.load(model_path, map_location=self.device)
            model.load_state_dict(state)

            # Load likelihood weights
            if os.path.exists(likelihood_path):
                lik_state = torch.load(likelihood_path, map_location=self.device)
                likelihood.load_state_dict(lik_state)

            # Attach to self
            self.gp_model = model
            self.likelihood = likelihood

            print(f"GPyTorch model loaded from {model_path}")

        else:
            print(f"Model file not found: {model_path}")

    # Simplified write_batch_info for core statistics
    def write_batch_info(self):
        print('WRITING BATCH INFO')
        batch_info = {'num_samples': [len(self.train_x)]}
        
        if self.do_uq_analysis:
            uq_results = self.uq_analysis()
            batch_info.update(uq_results)
            
        regression_results = self.regression_test()
        if regression_results:
            batch_info.update(regression_results)
            
        df = pd.DataFrame(batch_info)

        # Append to a master batch_info.csv for trend analysis
        all_batch_info_path = os.path.join(self.base_run_dir, 'batch_info.csv')
        if os.path.exists(all_batch_info_path):
            df.to_csv(all_batch_info_path, mode='a', header=False, index=False)
        else:
            df.to_csv(all_batch_info_path, mode='w', header=True, index=False)
        
        self.save_model(self.base_run_dir) # Save model after writing batch info

    def residuals_plot(self, out_dir=None, name='', y_range=None, train=False):
        import matplotlib.pyplot as plt
        import os
        import pandas as pd
        import numpy as np

        import logging
        logging.getLogger('matplotlib').setLevel(logging.WARNING)


        if out_dir is None:
            out_dir = os.path.join(self.base_run_dir, 'residuals_plots')
        os.makedirs(out_dir, exist_ok=True)

        # ---------------------------------------------------------
        # Select dataset: TRAIN or TEST
        # ---------------------------------------------------------
        if train:
            X = self.train_x.detach().cpu().numpy()
            y = self.train_y.detach().cpu().numpy().flatten()
            label = f"train-{len(self.train_x)}"
        else:
            if not self.test_data_csv:
                print("No test_data_csv provided — skipping residuals plot.")
                return None

            test_df = pd.read_csv(self.test_data_csv)
            out_col = self.get_output_col(df=test_df)

            X = test_df[self.parameters].values
            y = test_df[out_col].values
            label = f"test-{len(y)}_train-{len(self.train_x)}"

        # ---------------------------------------------------------
        # GP prediction (denormalized)
        # ---------------------------------------------------------
        y_pred_mean, _ = self.surrogate_predict(X)
        y_pred = y_pred_mean

        # Residuals
        residuals = y_pred - y

        # Optional y-range filtering
        if y_range is not None:
            y_min, y_max = y_range
            mask = (y >= y_min) & (y <= y_max)
            y = y[mask]
            y_pred = y_pred[mask]
            residuals = residuals[mask]

        # ---------------------------------------------------------
        # Metrics
        # ---------------------------------------------------------
        mse = np.mean((y_pred - y)**2)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs(y_pred - y) * 100 / np.abs(y))

        ss_res = np.sum((y_pred - y)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

        # ---------------------------------------------------------
        # Hyperparameters
        # ---------------------------------------------------------
        kernel = self.gp_model.covar_module
        lengthscales = kernel.base_kernel.lengthscale.detach().cpu().numpy().flatten()
        outputscale = kernel.outputscale.item()
        noise = self.likelihood.noise.item()

        ls_str = ", ".join([f"{p}: {lengthscales[i]:.4f}" for i, p in enumerate(self.parameters)])

        # ---------------------------------------------------------
        # Plotting
        # ---------------------------------------------------------
        fig, axes = plt.subplots(1, 2, figsize=(7, 3))

        # 1. Residuals hexbin
        ax = axes[0]
        hb = ax.hexbin(y, residuals, gridsize=40, cmap="viridis", mincnt=1)
        cb = fig.colorbar(hb, ax=ax)
        cb.set_label("Count")

        ax.set_xlabel("y")
        ax.set_ylabel("Residuals (y_pred - y)")
        ax.set_title(f"Residuals Hexbin ({label})")

        # 2. True vs Predicted
        ax2 = axes[1]
        hb2 = ax2.hexbin(y, y_pred, gridsize=40, cmap="viridis", mincnt=1)
        cb2 = fig.colorbar(hb2, ax=ax2)
        cb2.set_label("Count")

        ax2.set_xlabel("y")
        ax2.set_ylabel("y_pred")
        ax2.set_title(f"True vs Predicted ({label})")

        # Diagonal reference
        min_val = min(y.min(), y_pred.min())
        max_val = max(y.max(), y_pred.max())
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1)

        # Metrics + hyperparameters
        fig.text(
            0.5, 0.99,
            f"MAPE={mape:.4f} | RMSE={rmse:.4f} | R²={r2:.4f} | noise={noise:.2e} | outputscale={outputscale:.4f}\n"
            f"Lengthscales: {ls_str}",
            ha='center', va='top', fontsize=9
        )

        fig.tight_layout(rect=[0, 0, 1, 0.92])

        # Save
        save_path = os.path.join(out_dir, f"{name}residuals_{label}.png")
        print("Saving residuals plot to:", save_path)

        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def light_post_processing(self):
        if self.clean_pool_csv:
            os.remove(self.pool_csv_path)

        
    def register_future(self, future):
        return None

    def register_futures(self, futures):
        return None



# Example Usage (replace main function of original with this for testing)
if __name__ == '__main__':
    # Define parameters and bounds for an example
    # Using a simple 2D function like Ackley for demonstration.
    # We would typically load this from a config.
    parameters_example = ['x1', 'x2']
    bounds_example = [[-5.0, 5.0], [-5.0, 5.0]] # Real bounds

    # Create dummy pool CSV for initial test if needed
    if not os.path.exists('dummy_pool.csv'):
        num_pool_samples = 1000
        dummy_pool_data = {
            'x1': np.random.uniform(-5.0, 5.0, num_pool_samples),
            'x2': np.random.uniform(-5.0, 5.0, num_pool_samples),
            'output': np.random.rand(num_pool_samples) # Dummy output values
        }
        pd.DataFrame(dummy_pool_data).to_csv('dummy_pool.csv', index=False)
        print("Created dummy_pool.csv")

    # Create dummy test data CSV
    if not os.path.exists('dummy_test_data.csv'):
        num_test_samples = 200
        test_x1 = np.random.uniform(-5.0, 5.0, num_test_samples)
        test_x2 = np.random.uniform(-5.0, 5.0, num_test_samples)
        # Use Ackley function for a more realistic output for testing
        ackley_func = Ackley(dim=2, negate=True) # negate=True for maximization
        test_outputs = ackley_func(torch.tensor(np.array([test_x1, test_x2]).T, dtype=torch.double)).numpy()
        dummy_test_data = {
            'x1': test_x1,
            'x2': test_x2,
            'output': test_outputs
        }
        pd.DataFrame(dummy_test_data).to_csv('dummy_test_data.csv', index=False)
        print("Created dummy_test_data.csv")

    # Sampler configuration
    sampler_config_example = {
        'parameters': parameters_example,
        'bounds': bounds_example,
        'pool_csv_path': 'dummy_pool.csv', # Using the dummy pool
        'base_run_dir': 'torch_sobol_runs',
        'kernel_type': 'Matern52',
        'acquisition_mode': 'ei', # Try Expected Improvement
        'batch_size': 5,
        'initial_batch_size': 10,
        'budget': 50, # Max total samples to acquire
        'output_col': 'output', # Name of the output column
        'fix_noise': False,
        'noise_variance': 1e-6,
        'test_data_csv': 'dummy_test_data.csv'
    }

    sampler = GPyTorchAnalyticSobolSampler(**sampler_config_example)

    # Initial samples
    print("\n--- Getting initial samples ---")
    initial_samples = sampler.get_next_samples()
    print("Initial samples:", initial_samples)

    # Simulate running experiments and getting results for initial batch
    # Create a dummy dataset for the first batch
    batch_0_data = pd.DataFrame(initial_samples)
    ackley_func = Ackley(dim=2, negate=True)
    batch_0_data['output'] = ackley_func(torch.tensor(batch_0_data[sampler.parameters].values, dtype=torch.double)).numpy()
    
    batch_0_dir = os.path.join(sampler.base_run_dir, 'batch_0')
    os.makedirs(batch_0_dir, exist_ok=True)
    batch_0_csv_path = os.path.join(batch_0_dir, 'enchanted_dataset.csv')
    batch_0_data.to_csv(batch_0_csv_path, index=False)
    print(f"Saved initial samples with results to {batch_0_csv_path}")

    # Now, try to get subsequent batches
    while sampler.submitted_samples < sampler.budget:
        print(f"\n--- Batch {sampler.batch_number} ---")
        current_batch_dir = os.path.join(sampler.base_run_dir, f"batch_{sampler.batch_number}")
        
        # Get next samples, passing the path to the previous batch's results
        next_samples = sampler.get_next_samples(prev_dataset_path=os.path.join(sampler.base_run_dir, f"batch_{sampler.batch_number-1}", 'enchanted_dataset.csv'))
        
        if next_samples is None or not next_samples:
            print("No more samples to acquire or budget reached.")
            break

        print("Acquired samples:", next_samples)

        # Simulate running experiments for the new batch
        batch_n_data = pd.DataFrame(next_samples)
        batch_n_data['output'] = ackley_func(torch.tensor(batch_n_data[sampler.parameters].values, dtype=torch.double)).numpy()
        
        os.makedirs(current_batch_dir, exist_ok=True)
        batch_n_csv_path = os.path.join(current_batch_dir, 'enchanted_dataset.csv')
        batch_n_data.to_csv(batch_n_csv_path, index=False)
        print(f"Saved batch {sampler.batch_number-1} samples with results to {batch_n_csv_path}")

        sampler.write_batch_info(current_batch_dir)
        print(f"Total submitted samples: {sampler.submitted_samples}/{sampler.budget}")

    print("\n--- Final UQ Analysis and Regression Test ---")
    final_batch_dir = os.path.join(sampler.base_run_dir, f"batch_{sampler.batch_number-1}")
    if os.path.exists(os.path.join(final_batch_dir, 'enchanted_dataset.csv')):
        sampler.append_train_data(os.path.join(final_batch_dir, 'enchanted_dataset.csv'))
    sampler.fit_gpr_model()
    final_uq_results = sampler.uq_analysis()
    final_regression_results = sampler.regression_test()
    print("\nFinal UQ Results:")
    print(pd.DataFrame(final_uq_results))
    print("\nFinal Regression Results:")
    print(pd.DataFrame(final_regression_results))

    print("\n--- Cleanup dummy files (optional) ---")
    # Uncomment to clean up dummy files
    # os.remove('dummy_pool.csv')
    # os.remove('dummy_test_data.csv')
    # shutil.rmtree('torch_sobol_runs')
