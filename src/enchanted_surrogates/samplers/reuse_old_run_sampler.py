import numpy as np
from enchanted_surrogates.samplers.base_sampler import Sampler
import numpy as np
from scipy.stats.qmc import Sobol
import warnings
import os
import pandas as pd

class ReuseOldRunSampler(Sampler):
    def __init__(self, parameters, old_run_dir, **kwargs):
        self.old_run_dir = old_run_dir
        self.parameters = parameters
        self.output_parameter = kwargs.get('output_paramter',None)
        self.batch_number = 0
        self.base_run_dir = kwargs.get('base_run_dir')
        self.budget = self.line_count(os.path.join(self.base_run_dir,'enchanted_dataset.csv'))-1
    def get_next_samples(self) -> list[dict]:
        batch_path = os.path.join(self.base_run_dir, f'batch_{self.batch_number}','enchanted_dataset.csv')
        if os.path.exists(batch_path):
            batch_df = pd.read_csv(batch_path)
            samples_columns = self.parameters
            if self.output_parameter:
                output_col = [col for col in batch_df.columns if self.output_parameter in col]
                if len(output_col)>1:
                    raise ValueError('THE DATASET HAS MORE THAN ONE PARAMETER CONTAINNG THE output_parameter STRING.')
                output_col = output_col[0]
                samples_columns.append(output_col)
            samples_df = batch_df[samples_columns]
            if not 'output' in output_col:
                samples_df.rename({output_col:output_col+'_output'})
            samples = samples_df.to_dict(orient='records')
            
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
