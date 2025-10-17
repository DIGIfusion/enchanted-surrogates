import os
import numpy as np
import pandas as pd
from scipy.stats.qmc import Sobol
from enchanted_surrogates.samplers.base_sampler import Sampler
from scipy.stats import sobol_indices
import warnings
import numpy as np
import pandas as pd
from scipy.stats.qmc import Sobol
from scipy.stats import sobol_indices

class SobolIndicesSampler(Sampler):
    def __init__(self, parameters, bounds, seed=42,*args, **kwargs):
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
        print('INALIZING SOBOL INDICES SAMPLER')
        self.parameters = parameters
        self.bounds = bounds
        self.d = len(parameters)
        self.convergence_tolerence = kwargs.get('convergence_tolerence', None)
        self.previous_batch_info = None
        self.end_size = kwargs.get('end_size', None)
        self.start_size = kwargs.get('start_size', None)
        self.is_converged = False
        self.batch_dif = None
        self.do_write_batch_info = kwargs.get('do_write_batch_info', True)
                
        start_num_samples = kwargs.get('start_num_samples', None)
        end_num_samples = kwargs.get('end_num_samples', None)
        if not self.start_size:
            assert start_num_samples, "Either start_size or start_num_samples must be provided"
        if not self.end_size:
            assert end_num_samples, "Either end_size or end_num_samples must be provided"
        # start_size=8, end_size=32,
        
        if not self.end_size:
            print(f'debug choose_power_of_two. before {end_num_samples // (2 + self.d)} | after {choose_power_of_two(end_num_samples // (2 + self.d))}, log2={np.log2(choose_power_of_two(end_num_samples // (2 + self.d)))}')    
            self.end_size = end_num_samples // (2 + self.d)
            if not self._is_power_of_two(self.end_size):
                self.end_size = choose_power_of_two(self.end_size, mode="floor")
            if (2 + self.d) * self.end_size != end_num_samples:
                warnings.warn(f"Adjusted end_num_samples to {(2 + self.d) * self.end_size} to be compatible with Sobol sampling and to ensure uniformity.\n \
                                Adjusted end_size to {self.end_size} to be compatible with Sobol sampling and to ensure uniformity.")

        if not self.start_size:
            self.start_size = start_num_samples // (2 + self.d)
            if not self._is_power_of_two(self.start_size):
                self.start_size = choose_power_of_two(self.start_size, mode="floor")
            if (2 + self.d) * self.start_size != start_num_samples:
                warnings.warn(f"Adjusted end_num_samples to {(2 + self.d) * self.start_size} to be compatible with Sobol sampling and to ensure uniformity.\n \
                                Adjusted start_size to {self.start_size} to be compatible with Sobol sampling and to ensure uniformity.")

        self.budget = self.end_size * (2+self.d)
        self.seed = seed
        self.current_size = 0  # Current number of base samples generated
        self.base_run_dir = kwargs['base_run_dir']
        if not os.path.exists(self.base_run_dir):
            os.makedirs(self.base_run_dir)

        if not self._is_power_of_two(self.end_size):
            raise ValueError(f"end_size must be a power of 2, got {self.end_size}, log2(end_size)={np.log2(self.end_size)}")
        if not self._is_power_of_two(self.start_size):
            raise ValueError(f"start_size must be a power of 2, got {self.start_size}, log2(start_size)={np.log2(self.start_size)}")

        # Generate full Sobol sequences
        print('GENERATING SOBOL SEQUENCES')
        self.A_full = self._scale(Sobol(d=self.d, scramble=True, seed=seed).random(self.end_size))
        self.B_full = self._scale(Sobol(d=self.d, scramble=True, seed=seed + 1).random(self.end_size))
        self.batch_number = 0
        print('END INALIZING SOBOL INDICES SAMPLER')
    def _is_power_of_two(self, x):
        return (x & (x - 1) == 0) and x > 0

    def reset(self):
        """Reset the sampler to initial state."""
        self.current_size = 0
        self.batch_number = 0
        self.is_converged = False
        self.previous_batch_info = None
        self.batch_dif = None
    
    def _scale(self, samples):
        """Scale samples from [0,1] to bounds."""
        scaled = np.empty_like(samples)
        for i, (low, high) in enumerate(self.bounds):
            scaled[:, i] = samples[:, i] * (high - low) + low
        return scaled

    def get_next_samples(self, data_df=None):
        """
        Returns the next batch of samples as a list of dicts.
        Each dict includes 'source', 'index', and parameter values.
        """
        if self.batch_number > 0:
            if self.batch_number > 1:
                self.is_converged = self.check_convergence(data_df=data_df)
                if self.is_converged:
                    print(f"Convergence reached at batch {self.batch_number}. Stopping further sampling.")
                    return None
            previous_batch_dir = os.path.join(self.base_run_dir, f'batch_{self.batch_number-1}')
            if self.do_write_batch_info:
                self.write_batch_info(previous_batch_dir, data_df=data_df)
            if not data_df is None:
                self.previous_batch_info = self.analyze_from_df(df=data_df)
            else:
                self.previous_batch_info = self.analyze_from_df(csv_path=os.path.join(self.base_run_dir, 'enchanted_dataset.csv'))
            
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
        self.submitted += len(sample_dicts)
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

    def analyze_from_df(self, df=None, csv_path=None, output_column='output'):
        """
        Reads a CSV file containing Sobol samples and model outputs,
        computes Sobol indices, mean, and variance.

        Parameters:
        - csv_path: path to the CSV file
        - output_column: name of the column containing model outputs

        Returns:
        - result: dict with Sobol indices, mean, and variance
        """
        if (df is None) == (csv_path is None):
            raise ValueError("You must provide either `df` or `csv_path`, but not both.")
        if csv_path is not None:
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
    
    def check_convergence(self, data_df=None):
        if not self.previous_batch_info:
            return False
        if not data_df is None:
            batch_info = self.analyze_from_df(df=data_df)
        else:
            batch_info = self.analyze_from_df(csv_path=os.path.join(self.base_run_dir, 'enchanted_dataset.csv'))
        self.batch_dif = {key: abs(batch_info[key] - self.previous_batch_info[key]) for key in batch_info if key in ['mean','std'] or key.endswith('_sobolF') or key.endswith('_sobolT')}
        if not self.convergence_tolerence:
            raise ValueError("convergence_tolerence must be set to check convergence")
        if all(v == 0 for v in self.batch_dif.values()):
            raise ValueError("No change in batch info detected; cannot assess convergence. Likely previous_batch_info is identical to current batch_info and no new samples were added.")
        return all(v <= self.convergence_tolerence for v in self.batch_dif.values())
        
    def write_batch_info(self, batch_dir, data_df=None):
        if data_df:
            batch_info = self.analyze_from_df(df=data_df)
        else:
            enchanted_dataset = os.path.join(self.base_run_dir, 'enchanted_dataset.csv')
            batch_info = self.analyze_from_df(csv_path=enchanted_dataset)
        
        df = pd.DataFrame({k:[v] for k,v in batch_info.items()})
        all_batch_info_path = os.path.join(os.path.dirname(batch_dir), 'batch_info.csv')
                
        if os.path.exists(all_batch_info_path):
            df.to_csv(all_batch_info_path, mode='a', header=False, index=False)
        else:
            df.to_csv(all_batch_info_path, mode='w', header=True, index=False)
        df.to_csv(os.path.join(batch_dir,'batch_info.csv'), index=False)         
        return batch_info
    def register_future(self, future):
        """ Doesn't matter for random sampler TODO: Probably? """
        return None

    def register_futures(self, futures):
        return None
    

def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0

def next_power_of_two(n: int) -> int:
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()

def prev_power_of_two(n: int) -> int:
    if n <= 0:
        return 0
    return 1 << ((n).bit_length() - 1)

def choose_power_of_two(n: int, mode: str = "nearest") -> int:
    """
    mode: "floor" | "ceil" | "nearest"
    """
    if mode not in {"floor", "ceil", "nearest"}:
        raise ValueError("mode must be 'floor', 'ceil' or 'nearest'")
    if n <= 1:
        return 1
    if mode == "floor":
        return prev_power_of_two(n)
    if mode == "ceil":
        return next_power_of_two(n)
    # nearest
    lo = prev_power_of_two(n)
    hi = next_power_of_two(n)
    return lo if (n - lo) <= (hi - n) else hi

