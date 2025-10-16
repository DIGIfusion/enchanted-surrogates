import numpy as np
from enchanted_surrogates.samplers.base_sampler import Sampler
import numpy as np
from scipy.stats.qmc import Sobol
import warnings
import os
import pandas as pd

class ReuseOldRunSampler(Sampler):
    def __init__(self, parameters, old_run_dir, *args, **kwargs):
        """
        This is useful when combined with a sampler that trains a model and computes batch statistics, for example polynomial_chaos_expansion_regression_sampler.
        """
        self.old_run_dir = old_run_dir
        self.parameters = parameters
        self.output_parameter = kwargs.get('output_parameter',None)
        self.batch_number = 0
        self.base_run_dir = kwargs.get('base_run_dir')
        self.budget = self.line_count(os.path.join(self.old_run_dir,'enchanted_dataset.csv'))-1
    def get_next_samples(self) -> list[dict]:
        batch_path = os.path.join(self.old_run_dir, f'batch_{self.batch_number}','enchanted_dataset.csv')
        print('debug batch path:', batch_path)
        if os.path.exists(batch_path):
            batch_df = pd.read_csv(batch_path)
            samples_columns = self.parameters.copy()
            print('debug self.output_parameter', self.output_parameter)
            if self.output_parameter:
                print('debug there is an output parameter defined')
                output_col = [col for col in batch_df.columns if self.output_parameter in col]
                print('debug output col', output_col)
                if len(output_col)>1:
                    warnings.warn(f'THE DATASET HAS MORE THAN ONE PARAMETER CONTAINNG THE output_parameter [{self.output_parameter}]. HERE ARE THE OPTIONS: {output_col}. TAKING THE FIRST ONE: {output_col[0]}')
                output_col = output_col[0]
                print('debug output col 2', output_col)
                
                if not 'output' in output_col:
                    batch_df.rename(columns={output_col:output_col+'_output'}, inplace=True)
                    output_col = output_col+'_output'
                    samples_columns.append(output_col)
                    print('debug samples_columns', samples_columns)
            print('debug batch_df columns', batch_df.columns)
            samples_df = batch_df[samples_columns]
            samples = samples_df.to_dict(orient='records')
            print('sample', samples[0])    
            self.batch_number += 1
            self.submitted += len(samples)
            return samples
        else:
            return None

    def line_count(self,file_path):
        with open(file_path, 'r') as f:
            line_count = sum(1 for _ in f)
        return line_count
    
    def register_future(self, future):
        """ Doesn't matter for random sampler TODO: Probably? """
        return None

    def register_futures(self, futures):
        return None
