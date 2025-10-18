import os
from enchanted_surrogates.samplers.base_sampler import Sampler
from enchanted_surrogates.utils.precise_imports import import_sampler
import numpy as np
import pandas as pd
import chaospy as cp
import pickle
import warnings
from enchanted_surrogates.utils.timeout import run_with_timeout, FunctionTimeoutError, FunctionExecutionError
from enchanted_surrogates.utils.print_stats_table import print_stats_table
import time

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
        self.do_brute_force_uq = kwargs.get('do_brute_force_uq', False)
        self.base_run_dir = kwargs.get('base_run_dir')
        self.brute_force_uq_tolerence = kwargs.get('brute_force_uq_tolerence', 0.01)
        self.seed = kwargs.get('seed',42)
        
        if self.do_brute_force_uq:
            from enchanted_surrogates.samplers.sobol_indices_sampler import SobolIndicesSampler
            brute_force_uq_dir = os.path.join(self.base_run_dir, 'brute_force_uq')
            sis_kwargs = {
                'type': 'sobol_indices_sampler',
                'bounds': self.bounds,
                'parameters': self.parameters,
                'start_num_samples': (len(self.parameters)+2)*16,
                'method': 'saltelli',
                'end_num_samples': int(1e7),
                'convergence_tolerence': self.brute_force_uq_tolerence,
                'do_write_batch_info': False,
                'seed': 19588,
                'base_run_dir': brute_force_uq_dir,
            }
            self.sis = SobolIndicesSampler(**sis_kwargs)
            
        self.train = {}
        self.sampling_strategy = kwargs.get('sampling_strategy', 'random')  # random, lhs, halton, sobol, grid
        if self.sampling_strategy=='grid':
            warnings.warn('GRID SAMPLING STRATEGIE IS NOT SUITED TO BATCH SAMPLING. PLEASE USE ANOTHER STRATEGIE IF MORE THAN 1 BATCH IS REQUIRED')
        self.batch_size = kwargs.get('batch_size',None)
        self.sub_sampler_kwargs = kwargs.get('sub_sampler_kwargs', None)
        if self.sub_sampler_kwargs:
            if not 'parameters' in self.sub_sampler_kwargs:
                self.sub_sampler_kwargs['parameters'] = self.parameters
        
            if not 'bounds' in self.sub_sampler_kwargs:
                self.sub_sampler_kwargs['bounds'] = self.bounds 
        
            if self.batch_size:
                warnings.warn('BOTH BATCH SIZE AND sub_sampler_kwargs IS SET. THE BATCH SIZE WILL BE DECIDED BY THE SUB SAMPLER. IT IS NOT REQUIRED TO SET THE BATCH SIZE IN THIS CASE')
            self.sub_sampler = import_sampler(self.sub_sampler_kwargs['type'], self.sub_sampler_kwargs)
            
        if not self.batch_size and not self.sub_sampler_kwargs:
            warnings.warn('BATCH SIZE NOT SET, USING DEFAULT VALUE OF 2')
            self.batch_size = 2
        
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
        
        self.custom_submitted = 0
        
        
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
        print('debug len(new_train)', len(new_train))
        print('debug len(self.train)', len(self.train))
        self.train = {**self.train, **new_train}
        print('debug len(self.train) after update', len(self.train))

        previous_samples = np.array([list(k) for k in self.train.keys()], dtype=np.float64).T
        print('debug evaluations')
        evaluations = np.array(list(self.train.values()), dtype=float).flatten()
        print('debug fit poly')
        self.fitted_poly = cp.fit_regression(self.polynomials, previous_samples, evaluations)
        
        print('debug save poly grid')
        previous_batch_dir = os.path.join(self.base_run_dir, f'batch_{self.batch_number-1}')
        self.save_poly_grid(previous_batch_dir)

        print('debug write batch info')        
        self.write_batch_info_timeout = kwargs.get('write_batch_info_timeout', 120)
        try:
            run_with_timeout(self.write_batch_info, self.write_batch_info_timeout, kwargs={'batch_dir':previous_batch_dir})
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
        # self.submitted += len(samples)
        if self.custom_submitted >= self.budget:
            return None
        if not samples is None:
            self.custom_submitted += len(samples)
        return samples

    def save_poly_grid(self, batch_dir):
        os.makedirs(batch_dir, exist_ok=True)
        with open(os.path.join(batch_dir, 'fitted_poly.pkl'), 'wb') as file:
            pickle.dump(self.fitted_poly, file)
            
    def surrogate_predict(self, samples):
        assert samples.shape[1] == len(self.parameters), 'Input samples must have the same number of dimensions as parameters.'
        if not self.fitted_poly:
            raise RuntimeError('No fitted polynomial found for prediction.')
        samples = np.ascontiguousarray(samples, dtype=float)   # shape (N, D)
        y = self.fitted_poly(*samples.T)                            # returns shape (N,) for scalar polynomial
        y = np.asarray(y).squeeze()
        return y

    def uq_analysis(self):
        num_samples = len(self.train)
        print('COMPUTING EXPECTATION')
        expectation = cp.E(self.fitted_poly, self.dist)

        print('COMPUTING VARIANCE')
        variance = cp.Var(self.fitted_poly, self.dist)

        print('COMPUTING FIRST ORDER SOBOL INDICES')
        sobol_first = cp.Sens_m(self.fitted_poly, self.dist)
        
        print('COMPUTING TOTAL ORDER SOBOL INDICES')
        sobol_total = cp.Sens_t(self.fitted_poly, self.dist)
        
        sobol_first_dict = {param + '_sobolF': sf for param, sf in zip(self.parameters, sobol_first)}
        sobol_total_dict = {param + '_sobolT': st for param, st in zip(self.parameters, sobol_total)}

        batch_info = {
            'num_samples': [num_samples],
            'poly_order': [self.poly_order],
            'mean': [expectation],
            'std': [np.sqrt(variance)]
        }
        batch_info.update(sobol_first_dict)
        batch_info.update(sobol_total_dict)
        return batch_info
    
    def brute_force_uq_analysis(self):
        num_samples = len(self.train)
        batch_info = {
            'num_samples': num_samples,
            'poly_order': self.poly_order,
        }
        if not self.fitted_poly:
            raise RuntimeError('No fitted polynomial found for UQ analysis.')
        start = time.time()
        
        self.sis.reset()
        
        df = None
        num_eval = 0
        while self.sis.has_budget:
            samples = self.sis.get_next_samples(data_df=df)
            if not samples or self.sis.is_converged:
                print('BRUTE FORCE UQ ANALYSIS CONVERGED:', self.sis.is_converged)
                break
            
            dfi = pd.DataFrame.from_records(samples)
            evaluations = self.surrogate_predict(dfi[self.parameters].to_numpy(dtype=float))
            num_eval += len(evaluations)
            dfi['output'] = evaluations
            if df is None:
                df = dfi
            else:
                df = pd.concat([df, dfi], ignore_index=True)
            
            interval = time.time()
            stats = {'header':'BRUTE FORCE UQ STATS',
                    'TIME SEC': interval-start,
                    'NUM EVALUATIONS': num_eval,
                    'TIME PER EVAL SEC': (interval-start)/num_eval,
                    'CONVERGED?': self.sis.check_convergence(df),
                    'TOLERENCE SET': self.sis.convergence_tolerence}
            print_stats_table(stats)
        
        
        print('BRUTE FORCE UQ ANALYSIS CONVERGED:', self.sis.is_converged)
        brute_batch_info = self.sis.analyze_from_df(df)
        brute_batch_info = {'brute_'+k:v for k,v in brute_batch_info.items()}
        batch_info.update(brute_batch_info)
        end = time.time()
        stats = {'header':'BRUTE FORCE UQ STATS',
                 'TIME SEC': end-start,
                 'NUM EVALUATIONS': num_eval,
                 'TIME PER EVAL SEC': (end-start)/num_eval,
                 'CONVERGED?': self.sis.check_convergence(df),
                 'TOLERENCE SET': self.sis.convergence_tolerence}
        print_stats_table(stats)
        return batch_info
        
    def write_batch_info(self, batch_dir):
        print('WRITING BATCH INFO')

        if self.do_brute_force_uq:
            print('PERFORMING BRUTE FORCE UQ ANALYSIS')
            batch_info = self.brute_force_uq_analysis()
        else:
            print('PERFORMING STANDARD UQ ANALYSIS')
            batch_info = self.uq_analysis()
        
        df = pd.DataFrame({k:[v] for k,v in batch_info.items()})
        df.to_csv(os.path.join(batch_dir, 'batch_info.csv'), index=False)

        all_batch_info_path = os.path.join(os.path.dirname(batch_dir), 'batch_info.csv')
        if os.path.exists(all_batch_info_path):
            df.to_csv(all_batch_info_path, mode='a', header=False, index=False)
        else:
            df.to_csv(all_batch_info_path, mode='w', header=True, index=False)
    
    def register_future(self, future):
        """ Doesn't matter for random sampler TODO: Probably? """
        return None

    def register_futures(self, futures):
        return None