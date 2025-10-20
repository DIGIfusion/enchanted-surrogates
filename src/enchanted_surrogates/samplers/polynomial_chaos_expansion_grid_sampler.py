import os
# from .base import Sampler
from enchanted_surrogates.samplers.base_sampler import Sampler

import numpy as np
from dask.distributed import print
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# plt.rcParams.update({'font.size': 24})
import chaospy as cp
import numpoly as npy


import pickle
import warnings

class PolynomialChaosExpansionGridSampler(Sampler):
    """
# ┌────────────────────┬────────────────────────────────────┬────────────────────────────────────────┐
# │ Rule Code          │ Description                        │ Notes                                  │
# ├────────────────────┼────────────────────────────────────┼────────────────────────────────────────┤
# │ "G"                │ Gaussian quadrature                │ Optimal for smooth functions           │
# │ "E"                │ Gauss–Legendre quadrature          │ Uniform weight over interval           │
# │ "C"                │ Clenshaw–Curtis quadrature         │ Nested nodes; Uses Chebyshev nodes; good for sparse  │
# │ "L"                │ Leja sequence                      │ Nested nodes; useful for adaptivity    │
# │ "P"                │ Gauss–Patterson quadrature         │ Nested nodes; Extended Gaussian with   │
# │ "K"                │ Gauss–Kronrod quadrature           │ Enhances Gaussian with error estimate  │
# │ "R"                │ Gauss–Radau quadrature             │ Includes one endpoint of interval      │
# │ "lobatto"          │ Gauss–Lobatto quadrature           │ Includes both endpoints                │
# │ "fejer_1"/"fejer_2"│ Fejér quadrature                   │ Based on Chebyshev points              │
# │ "newton_cotes"     │ Newton–Cotes quadrature            │ Equally spaced nodes; less stable      │
# └────────────────────┴────────────────────────────────────┴────────────────────────────────────────┘
    """

    # sampler_interface = S.SEQUENTIAL

    def __init__(self, bounds, parameters, *args, **kwargs):
        """
        """
        print('CHAOSPY VERSION')
        print(cp.__version__)
        
        self.parameters = parameters
        self.bounds = bounds
        print(self.parameters, self.bounds, len(self.parameters), len(self.bounds))
        if len(self.parameters) != len(self.bounds):
            raise ValueError('The number of bounds and parameters do not match. Please define the same number of bounds as parameters')
        # self.num_samples = self.num_initial_points = float(num_samples)

        # self.samples = self.generate_parameters()
        self.initial_poly_order = kwargs.get('initial_poly_order', 3)
        self.sparse = kwargs.get('sparse', False)
        self.current_index = 0
        self.batch_number = 0
        self.train = {}
        self.all_batch_info = []
        self.budget = kwargs.get('budget')
        
        # Define a 2D uniform distribution over [0, 1]^2
        self.dist = cp.J(*[cp.Uniform(b[0], b[1]) for b in self.bounds])
        self.polynomials = None

        self.nodes = None
        self.weights = None
        self.weights_dict=None
        self.coeffs = None
        self.poly_order = self.initial_poly_order
        self.run_dirs = {}
        self.base_run_dir = kwargs.get('base_run_dir')
        self.quadrature_rule = kwargs.get('quadrature_rule',"C")
        if not (self.quadrature_rule == 'C' or self.quadrature_rule == 'L' or self.quadrature_rule == 'P'):
            warnings.warn(f'ONLY NESTED QUADRATURE RULES ARE CURRENTLY SUPPORTED FOR BATCH SAMPLING. RULES C,L,P ARE KNOWN TO BE NESTED. YOU CHOSE RULE: {self.quadrature_rule}. IF THIS IS NOT NESTED THEN THE RESULTS CANNOT BE TRUSTED.')
        # when using sparse grid the nested quadrature rules are often not completly nested. This keeps track of how many are dropped
        self.depreciated = []
        # # Define the model function: f(x) = x0 + x1 + x0*x1
        # def model(x):
        #     return x[0] + x[1] + x[0]*x[1]
        
    def get_initial_samples(self, *args, **kwargs):
        """
        Gets the initial parameters.

        Returns:
            list[dict[str, float]]: The initial parameters.
        """
        
        # Use Gaussian quadrature to compute nodes and weights
        self.nodes, self.weights = cp.generate_quadrature(self.initial_poly_order, self.dist, rule=self.quadrature_rule, sparse=self.sparse)
        print('debug nodes, len', len(self.nodes), type(self.nodes), self.nodes)
        print('debug nodes.T, len', len(self.nodes.T), type(self.nodes.T), self.nodes.T)
        print('debug weights, len', len(self.weights), type(self.weights), self.nodes)
        self.weights_dict = {tuple(k):v for k,v in zip(self.nodes.T, self.weights)}
        print('debug self.weights_dict made')
        samples = [{key: value for key, value in zip(self.parameters, params)} for params in self.nodes.T]
        print('debug samples made')
        self.batch_number += 1
        self.submitted += len(samples) 
        return samples

    def get_next_samples(
        self,
        batch_dir:str=None,
        *args,
        **kwargs) -> list[dict[str, float]]:
        
        if self.batch_number == 0:
            return self.get_initial_samples()
        
        else:
            
            if not self.base_run_dir:
                raise RuntimeError('base_run_dir IS NOT SET IN SAMPLER. THIS IS PASSED IN THE CONFIG FOR THE EXECUTOR. THE EXECUTOR MUST THEN PASS IT TO THE SAMPLER SO IT CAN GRAB DATA FOR TRAINING. ENSURE THE EXECUTOR HAS THIS LINE IN start_runs --> self.sampler.base_run_dir = self.base_run_dir ')
            new_data_df = pd.read_csv(os.path.join(self.base_run_dir, f'batch_{self.batch_number-1}', f'enchanted_dataset.csv'))
            output_col = [col for col in new_data_df.columns if 'output' in col]
            if len(output_col) > 1:
                raise RuntimeError('StaticSparseGrid SAMPLER REQUIRES EXACTLY ONE OUTPUT VARIABLE. THE single_code_run IN THE RUNNER SHOULD RETURN A DICTIONARY OF OUTPUTS WHERE ONLY ONE HAS output IN THE KEY, eg \{growthrate_output\:5\}. THIS IS THE ONE THAT WILL BE USED FOR ACTIVE LEARNING PUTPOSES')
            train_df = new_data_df[self.parameters + output_col]
            new_train = {
                tuple(row[col] for col in self.parameters): float(row[output_col].iloc[0])
                for _, row in train_df.iterrows()
            }

            self.train.update(new_train)
            del train_df

            # Compute PCE coefficients
            weights = np.array([self.approx_lookup(key, self.weights_dict) for key in self.nodes.T])
            samples = np.array([list(k) for k in self.nodes.T], dtype=np.float64).T
            evaluations = np.array([self.approx_lookup(key, self.train) for key in self.nodes.T]).flatten() # np.array(list(self.train.values()), dtype=float).flatten()
            print('debug evaluations', type(evaluations), evaluations)
            if evaluations.ndim != 1:
                raise ValueError(f"Expected 1D evaluations array, got shape {evaluations.shape}")

            # print(f"number of polynomial terms: {len(np.flatten(self.polynomials))}")

            # print("Any NaNs in weights?", np.isnan(weights).any())
            self.depreciated.append(len(self.train) - len(self.nodes.T))
            self.polynomials = cp.generate_expansion(self.poly_order, self.dist)
            self.fitted_poly = cp.fit_quadrature(self.polynomials, self.nodes, np.array(weights), evaluations)

            # write previous batch info
            previous_batch_dir = os.path.join(self.base_run_dir,f'batch_{self.batch_number}')
            self.save_poly_grid(previous_batch_dir)
            print('debug, current batch dir:', batch_dir , '\nWRITING batch DIR:',previous_batch_dir)
            self.write_batch_info(previous_batch_dir)

            # Get next samples
            self.poly_order += 1

            # Use Gaussian quadrature to compute nodes and weights
            if self.sparse:
                self.nodes, self.weights = cp.generate_quadrature(self.poly_order, self.dist, rule="G", sparse=True)
            else:
                self.nodes, self.weights = cp.generate_quadrature(self.poly_order, self.dist, rule="G", sparse=False)
            print('debug nodes', type(self.nodes), self.nodes)
            self.weights_dict = {tuple(k):v for k,v in zip(self.nodes.T, self.weights)}

    
            op = 0
            newp = 0
            for params in self.nodes.T:
                if not self.approx_in(params, self.train):
                    newp+=1
                    print('found new point:',newp)
                else:
                    op+=1
                    print('old point:', op)
            
            samples = [{key: value for key, value in zip(self.parameters, params)} for params in self.nodes.T if not self.approx_in(params, self.train)]

            self.batch_number += 1
            self.submitted += len(samples)
            if not self.has_budget:
                print(f'Exceeding budget of {self.budget} samples. Stopping sampling.')
                return []
            return samples
        
    # pce_model: callable that evaluates PCE at points with shape (dim, n_points)
    # def pce_model(self, points):
    #     # cp.call(polynomials, points) -> shape (n_basis, n_points)
    #     vals = cp.call(self.polynomials, points)          # shape (n_basis, n_points)
    #     return (vals.T @ self.coeffs).ravel()             # shape (n_points,)
    
    def save_poly_grid(self, batch_dir):
        with open(os.path.join(batch_dir, 'train.pkl'), 'wb') as file:
            pickle.dump(self.train, file)
        
        with open(os.path.join(batch_dir, 'weights_dict.pkl'), 'wb') as file:
            pickle.dump(self.train, file)
        
        with open(os.path.join(batch_dir, 'poly_order.pkl'), 'wb') as file:
            pickle.dump(self.poly_order, file)
    
    def write_batch_info(self, batch_dir):
        num_samples = len(self.nodes.T)
        # Construct the PCE surrogate model
        
        # # Expectation is the first coefficient (constant term)
        # expectation = cp.E(pce_model, self.dist)

        # if len(expectation) > 1:
        #     expectation=expectation[0]
        # else:
        #     print('EXPECTATION IS A SCALER AS IT SHOULD BE', len(expectation), expectation)

        # # Variance is the sum of squared non-constant coefficients
        # variance = cp.Var(pce_model, self.dist)
        # if len(variance)>1:
        #     variance = np.sum(variance)
        # # First-order Sobol indices
        # sobol_first = cp.Sens_m(pce_model, self.dist)
        # if len(sobol_first.shape) > 1:
        #     sobol_first = np.sum(sobol_first,axis=1)
        # sobol_first_dict = {param+'_sobolF': sf for param, sf in zip(self.parameters, sobol_first)}
        # # Total-order Sobol indices
        # sobol_total = cp.Sens_t(pce_model, self.dist)
        # if len(sobol_total) > 1:
        #     sobol_total = np.sum(sobol_total, axis=1)
        # sobol_total_dict = {param+'_sobolT': st for param, st in zip(self.parameters, sobol_total)}
        
        expectation = cp.E(self.fitted_poly, self.dist)
        variance = cp.Var(self.fitted_poly, self.dist)
        sobol_first = cp.Sens_m(self.fitted_poly, self.dist)
        sobol_total = cp.Sens_t(self.fitted_poly, self.dist)

        sobol_first_dict = {param + '_sobolF': sf for param, sf in zip(self.parameters, sobol_first)}
        sobol_total_dict = {param + '_sobolT': st for param, st in zip(self.parameters, sobol_total)}
        
        all_batch_info = {}
        all_batch_info.update({'num_samples':[num_samples], 'num_depreciated':self.depreciated[-1], 'total_runs':len(self.train), 'poly_order':self.poly_order, 'mean':[expectation], 'std':[np.sqrt(variance)]})
        all_batch_info.update(sobol_total_dict)
        all_batch_info.update(sobol_first_dict)
        # if self.batch_number==0:
        #     mean_diff = np.nan
        # else:
        #     base_run_dir = os.path.dirname(batch_dir)
        #     df = pd.read_csv(os.path.join(base_run_dir,f'batch_{self.batch_number-1}', 'batch_info.csv'))
        #     mean_before = df['mean'].iloc[0]
        #     mean_diff = np.abs(expectation - mean_before)
        
        df = pd.DataFrame(all_batch_info)
        df.to_csv(os.path.join(batch_dir, 'batch_info.csv'))
        
        all_batch_info_path = os.path.join(os.path.dirname(batch_dir), 'batch_info.csv')
        if os.path.exists(all_batch_info_path):
            df.to_csv(all_batch_info_path, mode='a', header=False, index=False)
        else:
            df.to_csv(all_batch_info_path, mode='w', header=True, index=False)
        
        
        self.all_batch_info.append(df)
        
        

    def approx_lookup(self, query_key, dictionary, tol=1e-9, default=None):
        return next(
            (dictionary[key]
            for key in dictionary
            if all(abs(a - b) < tol for a, b in zip(query_key, key))),
            default
        )

    def approx_in(self, query_key, dictionary, tol=1e-9):
        return any(all(abs(a - b) < tol for a, b in zip(query_key, key)) for key in dictionary)

    def register_future(self, future):
        """ Doesn't matter for random sampler TODO: Probably? """
        return None

    def register_futures(self, futures):
        return None