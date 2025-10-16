import os
from enchanted_surrogates.samplers.base_sampler import Sampler
from enchanted_surrogates.utils.precise_imports import import_sampler
import numpy as np
import pandas as pd
import chaospy as cp
import pickle
import warnings
from enchanted_surrogates.utils.timeout import run_with_timeout, FunctionTimeoutError, FunctionExecutionError

class PolynomialChaosExpansionRegressionSampler(Sampler):
    def __init__(self, *args, **kwargs):
        self.parameters = kwargs.get('parameters')
        self.bounds = kwargs.get('bounds')
        if not self.parameters and 'sub_sampler_kwargs' in kwargs:
            self.parameters = kwargs['sub_sampler_kwargs'].get('parameters')
        
        if not self.bounds and 'sub_sampler_kwargs' in kwargs:
            self.bounds = kwargs['sub_sampler_kwargs'].get('bounds')        
        
        if len(self.parameters) != len(self.bounds):
            raise ValueError('The number of bounds and parameters must match.')

        self.sparse = kwargs.get('sparse', False)
        self.current_index = 0
        self.batch_number = 0
        self.train = {}
        self.all_batch_info = []
        self.base_run_dir = kwargs.get('base_run_dir')
        self.sampling_strategy = kwargs.get('sampling_strategy', 'random')  # random, lhs, halton, sobol, grid
        if self.sampling_strategy=='grid':
            warnings.warn('GRID SAMPLING STRATEGIE IS NOT SUITED TO BATCH SAMPLING. PLEASE USE ANOTHER STRATEGIE IF MORE THAN 1 BATCH IS REQUIRED')
        self.batch_size = kwargs.get('batch_size',None)
        if not self.batch_size:
            warnings.warn('BATCH SIZE NOT SET, USING DEFAULT VALUE OF 2')
            self.batch_size = 2
        self.sub_sampler_kwargs = kwargs.get('sub_sampler_kwargs', None)
        if self.sub_sampler_kwargs:
            if not 'parameters' in self.sub_sampler_kwargs:
                self.sub_sampler_kwargs['parameters'] = self.parameters
        
            if not 'bounds' in self.sub_sampler_kwargs:
                self.sub_sampler_kwargs['bounds'] = self.bounds 
        
            if self.batch_size:
                warnings.warn('BOTH BATCH SIZE AND sub_sampler_kwargs IS SET. THE BATCH SIZE WILL BE DECIDED BY THE SUB SAMPLER.')
            self.sub_sampler = import_sampler(self.sub_sampler_kwargs['type'], self.sub_sampler_kwargs)
        self.seed = kwargs.get('seed',42)
        self.dist = cp.J(*[cp.Uniform(b[0], b[1]) for b in self.bounds])
        self.poly_order = kwargs.get('poly_order')
        if not self.poly_order:
            warnings.warn('poly_order IS NOT SET IN SAMPLER. TAKING DEFAULT VALUE OF 3')
            self.poly_order = 3
        self.polynomials = cp.generate_expansion(self.poly_order, self.dist)
        self.norms = cp.E(self.polynomials**2, self.dist)  # cache this once
        
        self.fitted_poly = None
        self.nodes = None
        self.coeffs = None
        self.submitted = 0
        
        self.budget = kwargs.get('budget')
        if not self.budget and 'sub_sampler_kwargs' in kwargs:
            self.budget = kwargs['sub_sampler_kwargs'].get('budget')
            if not self.budget:
                self.budget = self.sub_sampler.budget             
    
        if not self.budget:
            warnings.warn(f'BUDGET NOT SET, SETTING TO THE BATCH SIZE ({self.batch_size}) TO GET A SINGLE BATCH')
            self.budget = self.batch_size
            
        
        
    def get_initial_samples(self, *args, **kwargs):
        if self.sub_sampler_kwargs:
            samples = self.sub_sampler.get_next_samples()
        else:
            np.random.seed(self.seed)
            self.nodes = self.dist.sample(size=self.batch_size, rule=self.sampling_strategy)
            samples = [{key: value for key, value in zip(self.parameters, params)} for params in self.nodes.T]
        self.batch_number += 1
        self.submitted += len(samples)
        return samples

    def get_next_samples(self, batch_dir=None, *args, **kwargs):
        if self.batch_number == 0:
            return self.get_initial_samples()

        if not self.base_run_dir:
            raise RuntimeError('base_run_dir must be set to retrieve training data.')

        new_data_df = pd.read_csv(os.path.join(self.base_run_dir, f'batch_{self.batch_number-1}', 'enchanted_dataset.csv'))
        output_col = [col for col in new_data_df.columns if 'output' in col]
        if len(output_col) != 1:
            raise RuntimeError('Exactly one output column required.')

        train_df = new_data_df[self.parameters + output_col]
        print('debug self.parameters', self.parameters)
        print('debug output_col', output_col)
        print('debug train_df[output col[0]]', train_df[output_col[0]])
        new_train = {
            tuple(row[col] for col in self.parameters): float(row[output_col[0]])
            for _, row in train_df.iterrows()
        }
        self.train.update(new_train)

        previous_samples = np.array([list(k) for k in self.train.keys()], dtype=np.float64).T
        evaluations = np.array(list(self.train.values()), dtype=float).flatten()
        
        self.fitted_poly = cp.fit_regression(self.polynomials, previous_samples, evaluations)
        
        previous_batch_dir = os.path.join(self.base_run_dir, f'batch_{self.batch_number-1}')
        self.save_poly_grid(previous_batch_dir)
        
        self.write_batch_info_timeout = kwargs.get('write_batch_info_timeout', 120)
        try:
            run_with_timeout(self.write_batch_info, timeout=self.write_batch_info_timeout, batch_dir=previous_batch_dir)
        except FunctionTimeoutError:
            warnings.warn(f"write_batch_info timed out after 300 seconds; skipping batch info write for batch {self.batch_number-1}", UserWarning)
        except FunctionExecutionError as exc:
            warnings.warn(f"write_batch_info raised an exception: {exc}; skipping batch info write for batch {self.batch_number-1}", UserWarning)

        self.seed += 1
        if self.sub_sampler_kwargs:
            samples = self.sub_sampler.get_next_samples()
        else:
            np.random.seed(self.seed)
            self.nodes = self.dist.sample(size=self.batch_size, rule=self.sampling_strategy)
            samples = [{key: value for key, value in zip(self.parameters, params)} for params in self.nodes.T
                       if tuple(params) not in self.train]

        self.batch_number += 1
        self.submitted += len(samples)
        return samples

    def save_poly_grid(self, batch_dir):
        os.makedirs(batch_dir, exist_ok=True)
        with open(os.path.join(batch_dir, 'fitted_poly.pkl'), 'wb') as file:
            pickle.dump(self.fitted_poly, file)


    def gram_matrix_fast(self, basis, dist, poly_order=None, quad_order=None, sparse=True, cache=True):
        from threading import Lock

        """
        Fast Gram matrix via quadrature and cached polynomial evaluations.

        Caching keys: (id(basis), id(dist), quad_order, sparse)
        Returns G shape (M, M) dtype=float
        """
        M = len(basis)
        if M == 0:
            return np.empty((0, 0), dtype=float)

        # cache structure on self (thread-safe simple cache)
        if cache and not hasattr(self, "_gram_cache"):
            self._gram_cache = {}
            self._gram_cache_lock = Lock()

        key = (id(basis), id(dist), quad_order, sparse)

        if cache:
            with getattr(self, "_gram_cache_lock", Lock()):
                cached = self._gram_cache.get(key, None) if hasattr(self, "_gram_cache") else None
            if cached is not None:
                return cached

        # choose quad order conservatively and minimally
        if quad_order is None:
            if poly_order is None:
                quad_order = max(6, 8)
            else:
                quad_order = max(6, 2 * int(poly_order))

        try:
            # generate quadrature once
            nodes, weights = cp.generate_quadrature(quad_order, dist, rule="gaussian", sparse=sparse)
            weights = np.asarray(weights).reshape(-1)  # (N,)
            # evaluate all basis polynomials at nodes producing V shape (M, N)
            # vectorize evaluation: call poly(*nodes) for multivariate nodes
            if nodes.ndim == 1 or nodes.shape[0] == 1:
                # 1D inputs: nodes is (N,) or (1,N)
                N_nodes = nodes.shape[-1]
                V = np.empty((M, N_nodes), dtype=float)
                for i, poly in enumerate(basis):
                    V[i, :] = np.asarray(poly(nodes)).reshape(-1)
            else:
                N_nodes = nodes.shape[1]
                V = np.empty((M, N_nodes), dtype=float)
                for i, poly in enumerate(basis):
                    V[i, :] = np.asarray(poly(*nodes)).reshape(-1)

            # Weighted inner product: use sqrt(weights) to improve numerical stability
            w_sqrt = np.sqrt(weights).reshape(1, -1)  # shape (1, N)
            Vw = V * w_sqrt  # broadcasts: (M, N)
            # BLAS-backed matrix multiply: (M, N) @ (N, M) -> (M, M)
            G = np.dot(Vw, Vw.T)

            # Ensure symmetric and float
            G = 0.5 * (G + G.T)
            G = G.astype(float)

            if cache:
                with getattr(self, "_gram_cache_lock", Lock()):
                    self._gram_cache[key] = G

            return G

        except Exception as exc:
            warnings.warn(f"Quadrature path failed ({exc}); falling back to safe pairwise evaluation", UserWarning)
            # fallback pairwise (upper-triangle)
            G = np.empty((M, M), dtype=float)
            for i in range(M):
                for j in range(i, M):
                    val = float(cp.E(basis[i] * basis[j], dist))
                    G[i, j] = val
                    G[j, i] = val
            if cache:
                with getattr(self, "_gram_cache_lock", Lock()):
                    self._gram_cache[key] = G
            return G

    def gram_matrix_upper_comprehension(self, basis, dist):
        """
        Build Gram matrix G_ij = E[basis[i] * basis[j]] without explicit Python loops.
        Uses a list comprehension and vectorized scatter with np.triu_indices.
        """
        M = len(basis)
        if M == 0:
            return np.empty((0, 0), dtype=float)

        # 1) upper-triangle index pairs
        iu = np.triu_indices(M)               # tuple of arrays (i_indices, j_indices)

        # 2) build list of polynomial products for upper triangle using list comprehension
        prods = [basis[i] * basis[j] for i, j in zip(iu[0], iu[1])]

        # 3) evaluate expectations for all products in one call
        vals = np.asarray(cp.E(prods, dist), dtype=float)   # shape (n_pairs,)

        # 4) scatter into symmetric Gram matrix using vectorized indexing
        G = np.empty((M, M), dtype=float)
        G[iu] = vals
        # mirror upper triangle to lower triangle without loops
        G = G + G.T - np.diag(np.diag(G))

        return G
        

    def gram_matrix(self, basis, dist, poly_order=None, quad_order=None, sparse=True, cache=True):
        """
        Fast Gram matrix builder G_ij = E[basis[i] * basis[j]] using quadrature.

        Parameters
        - basis: iterable of chaospy.Poly (length M)
        - dist: chaospy distribution
        - poly_order: integer polynomial order of the basis (optional; used to pick quad_order)
        - quad_order: explicit quadrature order (overrides poly_order)
        - sparse: whether to request sparse quadrature when available
        - cache: store/return cached Gram matrix on self to avoid recomputation

        Returns
        - G: numpy array shape (M, M) dtype=float
        """
        # quick checks
        M = len(basis)
        if M == 0:
            return np.empty((0, 0), dtype=float)

        if cache and hasattr(self, "_gram_cache"):
            key = (id(basis), id(dist), poly_order, quad_order, sparse)
            cached = self._gram_cache.get(key, None)
            if cached is not None:
                return cached

        # choose quadrature order conservatively
        if quad_order is None:
            if poly_order is None:
                quad_order = max(6, 8)
            else:
                quad_order = max(6, 2 * int(poly_order))

        # get nodes and weights
        try:
            nodes, weights = cp.generate_quadrature(quad_order, dist, rule="gaussian", sparse=sparse)
            # nodes shape: (d, N) ; weights shape: (N,)
            # evaluate each basis polynomial at all nodes -> V shape (M, N)
            # Note: cp.poly can be called with vectorized args; use tuple unpacking
            if nodes.ndim == 1:
                # one-dimensional input: nodes is 1D of length N, need to pass as single array
                vals = np.vstack([np.asarray(poly(nodes)) for poly in basis])
            else:
                # multi-dimensional: nodes is shape (d, N); call poly(*nodes) returns length-N array
                V = []
                # evaluate in Python loop but evaluation at nodes is cheap relative to many cp.E calls
                for poly in basis:
                    V.append(np.asarray(poly(*nodes)))
                vals = np.vstack(V)  # shape (M, N)

            # weighted inner product: G = V @ diag(weights) @ V.T
            # compute W^0.5 * V first to improve numerical stability
            w_sqrt = np.sqrt(np.asarray(weights).reshape(1, -1))
            Vw = vals * w_sqrt  # shape (M, N)
            G = Vw @ Vw.T       # shape (M, M)
            G = G.astype(float)

            if cache:
                if not hasattr(self, "_gram_cache"):
                    self._gram_cache = {}
                self._gram_cache[key] = G

            return G

        except Exception as e:
            warnings.warn(f"Quadrature-based Gram construction failed ({e}); falling back to pairwise cp.E loop", UserWarning)
            # fallback: robust pairwise evaluation (upper triangle)
            G = np.empty((M, M), dtype=float)
            for i in range(M):
                for j in range(i, M):
                    val = float(cp.E(basis[i] * basis[j], dist))
                    G[i, j] = val
                    G[j, i] = val
            if cache:
                if not hasattr(self, "_gram_cache"):
                    self._gram_cache = {}
                self._gram_cache[key] = G
            return G
    

    def write_batch_info(self, batch_dir):
        print('WRITING BATCH INFO')
        num_samples = len(self.train)

        print('COMPUTING EXPECTATION')
        expectation = cp.E(self.fitted_poly, self.dist)

        print('COMPUTING VARIANCE')
        # coeffs = np.asarray(self.fitted_poly.coefficients)
        # variance = np.sum(coeffs[1:]**2 * self.norms[1:])
        variance = cp.Var(self.fitted_poly, self.dist)

        print('COMPUTING FIRST ORDER SOBOL INDICES')
        sobol_first = cp.Sens_m(self.fitted_poly, self.dist)
        # exponents = self.fitted_poly.exponents  # shape: (num_terms, num_vars)
        # sobol_first = []
        # for i, param in enumerate(self.parameters):
        #     # First-order Sobol index for variable i
        #     mask_first = (exponents[:, i] > 0) & (np.sum(exponents, axis=1) == exponents[:, i])
        #     sobol_first.append(np.sum(coeffs[mask_first]**2 * self.norms[mask_first]) / variance)

        print('COMPUTING TOTAL ORDER SOBOL INDICES')
        sobol_total = cp.Sens_t(self.fitted_poly, self.dist)
        # sobol_total = []
        # for i, param in enumerate(self.parameters):
        #     mask_total = exponents[:, i] > 0
        #     sobol_total.append(np.sum(coeffs[mask_total]**2 * self.norms[mask_total]) / variance)

        # assert coeffs.shape[0] == self.norms.shape[0] == exponents.shape[0]

        sobol_first_dict = {param + '_sobolF': sf for param, sf in zip(self.parameters, sobol_first)}
        sobol_total_dict = {param + '_sobolT': st for param, st in zip(self.parameters, sobol_total)}

        batch_info = {
            'num_samples': [num_samples],
            'poly_order': self.poly_order,
            'mean': [expectation],
            'std': [np.sqrt(variance)]
        }
        batch_info.update(sobol_first_dict)
        batch_info.update(sobol_total_dict)

        df = pd.DataFrame(batch_info)
        df.to_csv(os.path.join(batch_dir, 'batch_info.csv'), index=False)

        all_batch_info_path = os.path.join(os.path.dirname(batch_dir), 'batch_info.csv')
        if os.path.exists(all_batch_info_path):
            df.to_csv(all_batch_info_path, mode='a', header=False, index=False)
        else:
            df.to_csv(all_batch_info_path, mode='w', header=True, index=False)


    # def write_batch_info(self, batch_dir, tol_orth=1e-8, tol_align=1e-6, quad_order_factor=2):
    #     """
    #     Write batch info with rigorous checks and safe fallbacks.

    #     Expects on `self`:
    #     - self.polynomials : chaospy basis used for PCE
    #     - self.fitted_poly  : chaospy.Poly with coefficients and exponents
    #     - self.norms        : optional precomputed norms (if present must align)
    #     - self.dist         : chaospy distribution
    #     - self.parameters   : list of parameter names (length = number of input dims)
    #     - self.poly_order   : polynomial order (optional)
    #     - self.train        : training dataset (for sample count)
    #     - self.all_batch_info : list to append resulting df
    #     """
        
    #     print('WRITING BATCH INFO')

    #     # Basic extracts and shape checks
    #     coeffs = np.asarray(self.fitted_poly.coefficients)
    #     exponents = np.asarray(self.fitted_poly.exponents)
    #     basis = getattr(self, "polynomials")
    #     dist = getattr(self, "dist")
    #     params = list(self.parameters)
    #     num_samples = len(self.train)

    #     if coeffs.ndim != 1:
    #         raise ValueError("coeffs must be a 1D array")
    #     if exponents.ndim != 2:
    #         raise ValueError("exponents must be shape (num_terms, num_vars)")
    #     if exponents.shape[0] != coeffs.shape[0]:
    #         raise ValueError("Mismatch: number of exponents rows != number of coefficients")

    #     M = coeffs.shape[0]
    #     d = exponents.shape[1]
    #     if d != len(params):
    #         raise ValueError("Dimension mismatch: exponents have %d columns but parameters length is %d" % (d, len(params)))

    #     orthogonal = True
    #     # takes too long
    #     # print('CHECKING ORTHOGONALITY')
    #     # try:
    #     #     G = self.gram_matrix_upper_comprehension(self.polynomials, self.dist)
    #     # except Exception as e:
    #     #     warnings.warn(f"Gram matrix construction failed: {e}; falling back to conservative behavior", UserWarning)
    #     #     G = None

    #     # if G is None:
    #     #     orthogonal = False
    #     # else:
    #     #     offdiag = np.max(np.abs(G - np.diag(np.diag(G))))
    #     #     orthogonal = offdiag < tol_orth
    #     #     print(f"Gram off-diag max: {offdiag:.3e}")
    #     #     if not orthogonal:
    #     #         warnings.warn('Basis not numerically orthogonal; Sobol/variance results may be invalid', UserWarning)

    #     # 2) Norms: prefer self.norms if provided, else compute
    #     if hasattr(self, "norms") and self.norms is not None:
    #         norms = np.asarray(self.norms)
    #         if norms.shape[0] != M:
    #             raise ValueError("Provided self.norms length does not match number of basis terms")
    #     else:
    #         print('COMPUTING NORMS')
    #         norms = np.asarray(cp.E(basis**2, dist))
    #     if np.any(norms < 0):
    #         raise ValueError("Found negative norm values; check basis/dist")

    #     # 3) Coefficient alignment check: inner products between fitted_poly and each basis term
    #     print('CHECKING COEFFICIENT ALIGNMENT')
    #     inner = np.asarray(cp.E(self.fitted_poly * basis, dist))   # shape (M,)
    #     # For orthogonal basis, inner[j] == coeffs[j] * norms[j]
    #     residual = inner - coeffs * norms
    #     max_resid = np.max(np.abs(residual))
    #     print(f"Max coefficient alignment residual: {max_resid:.3e}")
    #     aligned = (max_resid < tol_align)

    #     # If not aligned, attempt reprojection to recover coefficients that match the basis
    #     reproj_coeffs = None
    #     if not aligned:
    #         warnings.warn("Coefficient alignment check failed; attempting reprojection using inner/norms", UserWarning)
    #         # avoid division by zero
    #         zero_norms = norms == 0
    #         if np.any(zero_norms):
    #             raise RuntimeError("Some basis norms are zero; cannot reprojection. Investigate basis or distribution.")
    #         reproj_coeffs = inner / norms
    #         reproj_diff = np.max(np.abs(reproj_coeffs - coeffs))
    #         print(f"Max reprojection diff: {reproj_diff:.3e}")
    #         # if reprojection is closer to inner consistency, prefer reproj_coeffs
    #         if reproj_diff < max_resid:
    #             warnings.warn("Using reprojected coefficients for variance and Sobol calculations", UserWarning)
    #             coeffs_used = reproj_coeffs
    #         else:
    #             warnings.warn("Reprojection did not improve alignment; proceeding with original coefficients but treat results cautiously", UserWarning)
    #             coeffs_used = coeffs
    #     else:
    #         coeffs_used = coeffs

    #     # 4) Compute mean (expectation) via cp.E (safe)
    #     print('COMPUTING EXPECTATION')
    #     expectation = float(cp.E(self.fitted_poly, dist))

    #     # 5) Compute variance: try coefficient-based if orthogonality and alignment hold
    #     print('COMPUTING VARIANCE')
    #     safe_for_coeff = orthogonal and aligned
    #     if safe_for_coeff:
    #         # exclude constant term (assume index 0 is constant)
    #         total_var_coeff = float(np.sum(coeffs_used[1:]**2 * norms[1:]))
    #         # quick consistency check vs quadrature-based evaluation
    #         try:
    #             # choose quadrature order conservatively
    #             poly_order = getattr(self, "poly_order", None)
    #             if poly_order is None:
    #                 quad_order = max(6, quad_order_factor * 3)
    #             else:
    #                 quad_order = max(6, quad_order_factor * poly_order)
    #             nodes, weights = cp.generate_quadrature(quad_order, dist, rule="gaussian", sparse=True)
    #             vals = np.asarray(self.fitted_poly(*nodes))
    #             E_val = float(np.dot(vals, weights))
    #             E_val2 = float(np.dot(vals**2, weights))
    #             total_var_quad = E_val2 - E_val**2
    #             rel_err = abs(total_var_coeff - total_var_quad) / max(1e-16, abs(total_var_quad))
    #             print(f"Variance coeff: {total_var_coeff:.6e}, quad: {total_var_quad:.6e}, rel_err: {rel_err:.3e}")
    #             if rel_err > 1e-6:
    #                 warnings.warn("Coefficient-based variance differs from quadrature by relative error %.2e; falling back to quadrature" % rel_err, UserWarning)
    #                 variance = total_var_quad
    #             else:
    #                 variance = total_var_coeff
    #         except Exception as e:
    #             warnings.warn("Quadrature check failed (%s). Using coefficient variance result with caution." % (e,), UserWarning)
    #             variance = total_var_coeff
    #     else:
    #         # fallback to quadrature-based variance
    #         print("FALLING BACK TO QUADRATURE FOR VARIANCE")
    #         poly_order = getattr(self, "poly_order", None)
    #         if poly_order is None:
    #             quad_order = max(6, quad_order_factor * 3)
    #         else:
    #             quad_order = max(6, quad_order_factor * poly_order)
    #         nodes, weights = cp.generate_quadrature(quad_order, dist, rule="gaussian", sparse=True)
    #         vals = np.asarray(self.fitted_poly(*nodes))
    #         E_val = float(np.dot(vals, weights))
    #         E_val2 = float(np.dot(vals**2, weights))
    #         variance = E_val2 - E_val**2

    #     if variance <= 0:
    #         warnings.warn("Computed variance is non-positive; check fitted_poly and computations", UserWarning)
    #         variance = float(max(0.0, variance))

    #     # 6) Compute Sobol indices using coefficient-based decomposition when safe.
    #     #    If not safe, compute via chaospy Sens_m / Sens_t (slower but robust)
    #     print('COMPUTING FIRST ORDER SOBOL INDICES')
    #     if orthogonal and (aligned or reproj_coeffs is not None):
    #         c = coeffs_used
    #         norms_used = norms
    #         sobol_first = []
    #         for i, _param in enumerate(params):
    #             mask_first = (exponents[:, i] > 0) & (np.sum(exponents, axis=1) == exponents[:, i])
    #             numerator = np.sum((c[mask_first]**2) * norms_used[mask_first])
    #             sobol_first.append(float(numerator / variance) if variance > 0 else 0.0)
    #     else:
    #         warnings.warn("Coefficient-based first-order Sobol skipped; using Chaospy Sens_m fallback", UserWarning)
    #         sobol_first = list(cp.Sens_m(self.fitted_poly, dist))

    #     print('COMPUTING TOTAL ORDER SOBOL INDICES')
    #     if orthogonal and (aligned or reproj_coeffs is not None):
    #         c = coeffs_used
    #         norms_used = norms
    #         sobol_total = []
    #         for i, _param in enumerate(params):
    #             mask_total = exponents[:, i] > 0
    #             numerator = np.sum((c[mask_total]**2) * norms_used[mask_total])
    #             sobol_total.append(float(numerator / variance) if variance > 0 else 0.0)
    #     else:
    #         warnings.warn("Coefficient-based total-order Sobol skipped; using Chaospy Sens_t fallback", UserWarning)
    #         sobol_total = list(cp.Sens_t(self.fitted_poly, dist))

    #     # Final assertions / shapes
    #     assert len(sobol_first) == d
    #     assert len(sobol_total) == d
    #     assert coeffs.shape[0] == norms.shape[0] == exponents.shape[0]

    #     # Prepare output DataFrame
    #     sobol_first_dict = {param + '_sobolF': sf for param, sf in zip(params, sobol_first)}
    #     sobol_total_dict = {param + '_sobolT': st for param, st in zip(params, sobol_total)}

    #     batch_info = {
    #         'num_samples': [num_samples],
    #         'poly_order': getattr(self, 'poly_order', None),
    #         'mean': [expectation],
    #         'std': [np.sqrt(variance)]
    #     }
    #     batch_info.update(sobol_first_dict)
    #     batch_info.update(sobol_total_dict)

    #     df = pd.DataFrame(batch_info)
    #     # save batch csv
    #     df.to_csv(os.path.join(batch_dir, 'batch_info.csv'), index=False)

    #     # append to aggregate csv
    #     all_batch_info_path = os.path.join(os.path.dirname(batch_dir), 'batch_info.csv')
    #     if os.path.exists(all_batch_info_path):
    #         df.to_csv(all_batch_info_path, mode='a', header=False, index=False)
    #     else:
    #         df.to_csv(all_batch_info_path, mode='w', header=True, index=False)

    #     # store for later inspection
    #     if not hasattr(self, 'all_batch_info'):
    #         self.all_batch_info = []
    #     self.all_batch_info.append(df)

    #     return df

    
    def register_future(self, future):
        """ Doesn't matter for random sampler TODO: Probably? """
        return None

    def register_futures(self, futures):
        return None