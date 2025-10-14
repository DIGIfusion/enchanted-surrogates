import os
from enchanted_surrogates.samplers.base_sampler import Sampler
from enchanted_surrogates.utils.precise_imports import import_sampler
import numpy as np
import pandas as pd
import chaospy as cp
import pickle
import warnings

class PolynomialChaosExpansionRegressionSampler(Sampler):
    def __init__(self, bounds, parameters, *args, **kwargs):
        self.parameters = parameters
        self.bounds = bounds
        if len(self.parameters) != len(self.bounds):
            raise ValueError('The number of bounds and parameters must match.')

        self.sparse = kwargs.get('sparse', False)
        self.current_index = 0
        self.batch_number = 0
        self.train = {}
        self.all_batch_info = []
        self.budget = kwargs.get('budget')
        self.base_run_dir = kwargs.get('base_run_dir')
        self.sampling_strategy = kwargs.get('sampling_strategy', 'random')  # random, lhs, halton, sobol, grid
        if self.sampling_strategy=='grid':
            warnings.warn('GRID SAMPLING STRATEGIE IS NOT SUITED TO BATCH SAMPLING. PLEASE USE ANOTHER STRATEGIE IF MORE THAN 1 BATCH IS REQUIRED')
        self.batch_size = kwargs.get('batch_size',None)
        self.sub_sampler_kwargs = kwargs.get('sub_sampler_kwargs', None)
        if self.sub_sampler_kwargs:
            self.sub_sampler = import_sampler(self.sub_sampler_kwargs['type'], self.sub_sampler_kwargs)
        self.seed = kwargs.get('seed',42)
        self.dist = cp.J(*[cp.Uniform(b[0], b[1]) for b in self.bounds])
        self.polynomials = None
        self.nodes = None
        self.coeffs = None
        self.poly_order = kwargs.get('poly_order')
        self.submitted = 0
        
    def get_initial_samples(self, *args, **kwargs):
        self.polynomials = cp.generate_expansion(self.poly_order, self.dist)
        
        if self.sub_sampler_kwargs:
            samples = self.sub_sampler.get_next_samples()
        else:
            np.random.seed(self.seed)
            self.nodes = cp.sample(self.dist, size=self.batch_size, rule=self.sampling_strategy)
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
        new_train = {
            tuple(row[col] for col in self.parameters): float(row[output_col[0]])
            for _, row in train_df.iterrows()
        }
        self.train.update(new_train)

        previous_samples = np.array([list(k) for k in self.train.keys()], dtype=np.float64).T
        evaluations = np.array(list(self.train.values()), dtype=float).flatten()
        
        self.coeffs = cp.fit_regression(self.polynomials, previous_samples, evaluations)

        previous_batch_dir = os.path.join(self.base_run_dir, f'batch_{self.batch_number}')
        self.save_poly_grid(previous_batch_dir)
        self.write_batch_info(previous_batch_dir)

        self.polynomials = cp.generate_expansion(self.poly_order, self.dist)

        self.seed += 1
        if self.sub_sampler_kwargs:
            samples = self.sub_sampler.get_next_samples()
        else:
            np.random.seed(self.seed)
            self.nodes = cp.sample(self.dist, size=self.batch_size, rule=self.sampling_strategy)
            samples = [{key: value for key, value in zip(self.parameters, params)} for params in self.nodes.T
                       if tuple(params) not in self.train]

        self.batch_number += 1
        self.submitted += len(samples)
        return samples

    def save_poly_grid(self, batch_dir):
        os.makedirs(batch_dir, exist_ok=True)
        with open(os.path.join(batch_dir, 'train.pkl'), 'wb') as file:
            pickle.dump(self.train, file)
        with open(os.path.join(batch_dir, 'poly_order.pkl'), 'wb') as file:
            pickle.dump(self.poly_order, file)

    def write_batch_info(self, batch_dir):
        num_samples = len(self.train)
        pce_model = cp.call(self.polynomials, self.coeffs)
        expectation = cp.E(pce_model, self.dist)
        variance = cp.Var(pce_model, self.dist)
        sobol_first = cp.Sens_m(pce_model, self.dist)
        sobol_total = cp.Sens_t(pce_model, self.dist)

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

        self.all_batch_info.append(df)
