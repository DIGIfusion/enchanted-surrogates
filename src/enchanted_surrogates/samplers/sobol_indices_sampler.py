import os
import numpy as np
import pandas as pd
from scipy.stats.qmc import Sobol
from enchanted_surrogates.samplers.base_sampler import Sampler
from scipy.stats import sobol_indices

import numpy as np
import pandas as pd
from scipy.stats.qmc import Sobol
from scipy.stats import sobol_indices

class SobolIndicesSampler(Sampler):
    def __init__(self, parameters, bounds, start_size=8, end_size=32, seed=42,*args, **kwargs):
        """"
        SobolIndicesSampler generates structured samples for Sobol sensitivity analysis.

        This sampler produces input samples for computing first-order and total-order Sobol indices
        using the Saltelli sampling scheme. It generates three sets of samples:
        - A: base samples
        - B: independent base samples
        - AB_i: hybrid samples where column i of A is replaced with column i of B

        Each base sample (A or B) has dimensionality `d = len(parameters)`, and the total number of
        model evaluations required for Sobol analysis is:

            total_evals = (2 + d) * n

        Where:
        - `n` is the number of base samples (must be a power of 2)
        - `d` is the number of input parameters

        Sampling proceeds in batches, starting with `start_size` base samples and doubling the base
        sample count with each call to `get_next_samples()` until reaching the `budget`.

        Parameters:
        - parameters: list of parameter names (length d)
        - bounds: list of [min, max] pairs for each parameter (length d)
        - end_size: total number of base samples to generate (must be a power of 2)
        - start_size: initial number of base samples (must be a power of 2 and ≤ budget)
        - seed: random seed for reproducibility
        """

        self.parameters = parameters
        self.bounds = bounds
        self.d = len(parameters)
        self.end_size = end_size
        self.budget = end_size * (2+self.d)
        self.start_size = start_size
        self.seed = seed
        self.current_size = 0
        self.base_run_dir = kwargs['base_run_dir']

        if not self._is_power_of_two(self.end_size):
            raise ValueError("end_size must be a power of 2")
        if not self._is_power_of_two(start_size):
            raise ValueError("start_size must be a power of 2")

        # Generate full Sobol sequences
        self.A_full = self._scale(Sobol(d=self.d, scramble=True, seed=seed).random(self.end_size))
        self.B_full = self._scale(Sobol(d=self.d, scramble=True, seed=seed + 1).random(self.end_size))
        self.batch_number = 0
    def _is_power_of_two(self, x):
        return (x & (x - 1) == 0) and x > 0

    def _scale(self, samples):
        """Scale samples from [0,1] to bounds."""
        scaled = np.empty_like(samples)
        for i, (low, high) in enumerate(self.bounds):
            scaled[:, i] = samples[:, i] * (high - low) + low
        return scaled

    def get_next_samples(self):
        """
        Returns the next batch of samples as a list of dicts.
        Each dict includes 'source', 'index', and parameter values.
        """
        if self.batch_number > 0:
            previous_batch_dir = os.path.join(self.base_run_dir, f'batch_{self.batch_number-1}')
            self.write_batch_info(previous_batch_dir)

        if self.current_size >= self.end_size:
            return None

        # Determine next batch size
        next_batch_size = min(max(self.current_size, self.start_size), self.end_size - self.current_size)
        start = self.current_size
        end = start + next_batch_size

        A = self.A_full[start:end]
        B = self.B_full[start:end]

        AB_list = []
        for i in range(self.d):
            AB = A.copy()
            AB[:, i] = B[:, i]
            AB_list.append(AB)

        sample_dicts = []
        for i in range(end - start):
            global_index = start + i
            sample_dicts.append({
                'source': 'A',
                'index': global_index,
                **{name: float(val) for name, val in zip(self.parameters, A[i])}
            })
            sample_dicts.append({
                'source': 'B',
                'index': global_index,
                **{name: float(val) for name, val in zip(self.parameters, B[i])}
            })
            for dim in range(self.d):
                sample_dicts.append({
                    'source': f'AB_{dim}',
                    'index': global_index,
                    **{name: float(val) for name, val in zip(self.parameters, AB_list[dim][i])}
                })

        self.current_size = end
        self.batch_number += 1
        return sample_dicts

    def assemble_func_dict(self, df, output_column):
        """
        df: pandas DataFrame with columns matching self.parameters + 'source' + 'index' + output_column
        output_column: name of the column containing model outputs

        Returns: dict with keys 'f_A', 'f_B', 'f_AB' for use in sobol_indices()
        """
        s = 1
        n = df[df['source'] == 'A']['index'].nunique()
        if not self._is_power_of_two(n):
            raise ValueError("Number of samples must be a power of 2")

        f_A = df[df['source'] == 'A'].sort_values('index')[output_column].values.reshape(s, n)
        f_B = df[df['source'] == 'B'].sort_values('index')[output_column].values.reshape(s, n)
        f_AB = np.zeros((self.d, s, n))
        for dim in range(self.d):
            f_AB[dim] = df[df['source'] == f'AB_{dim}'].sort_values('index')[output_column].values.reshape(s, n)
        return {'f_A': f_A, 'f_B': f_B, 'f_AB': f_AB}

    def analyze_from_csv(self, csv_path, output_column='output'):
        """
        Reads a CSV file containing Sobol samples and model outputs,
        computes Sobol indices, mean, and variance.

        Parameters:
        - csv_path: path to the CSV file
        - output_column: name of the column containing model outputs

        Returns:
        - result: dict with Sobol indices, mean, and variance
        """
        df = pd.read_csv(csv_path)
        output_column = next((col for col in df.columns if output_column in col), output_column)
        func_dict = self.assemble_func_dict(df, output_column)
        n = func_dict['f_A'].shape[1]

        res = sobol_indices(func=func_dict, n=n)

        all_uniform_outputs = np.concatenate([
            func_dict['f_A'].flatten(),
            func_dict['f_B'].flatten()
        ])
        mean = np.mean(all_uniform_outputs)
        variance = np.var(all_uniform_outputs)

        result = {
            'num_samples': len(df),
            'num_uniform_samples':len(all_uniform_outputs), 
            'mean': mean,
            'std': np.sqrt(variance)
        }

        for i, param in enumerate(self.parameters):
            result[f'{param}_sobolF'] = res.first_order[i]
            result[f'{param}_sobolT'] = res.total_order[i]
        
        return result

    def write_batch_info(self, batch_dir):
        enchanted_dataset = os.path.join(self.base_run_dir, 'enchanted_dataset.csv')
        batch_info = self.analyze_from_csv(enchanted_dataset)
        df = pd.DataFrame({k:[v] for k,v in batch_info.items()})
        all_batch_info_path = os.path.join(os.path.dirname(batch_dir), 'batch_info.csv')
        if os.path.exists(all_batch_info_path):
            df.to_csv(all_batch_info_path, mode='a', header=False, index=False)
        else:
            df.to_csv(all_batch_info_path, mode='w', header=True, index=False)
        df.to_csv(os.path.join(batch_dir,'batch_info.csv'), index=False)         
        
    
    def register_future(self, future):
        """ Doesn't matter for random sampler TODO: Probably? """
        return None

    def register_futures(self, futures):
        return None
