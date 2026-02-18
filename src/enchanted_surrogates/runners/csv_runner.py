import pandas as pd
import numpy as np
from .base_runner import Runner
import shutil
from time import time

_df_cache = None

class CsvRunner(Runner):
    """
    A runner that simply looks up outputs from a CSV table.
    """    
    def __init__(self, csv_path, tol=1e-8, type='CsvRunner', **kwargs):
        self.csv_path = csv_path    
        self.remove_run_dir = kwargs.get('remove_run_dir', None)    
        self.remove_from_cache = kwargs.get('remove_from_cache', False)
        global _df_cache

        if _df_cache is None:
            print("Loading CSV once on this worker...")
            _df_cache = pd.read_csv(self.csv_path)

        self.df = _df_cache
        
        print('debug df shape', self.df.shape)
        
        import os
        print("Runner init PID:", os.getpid())

        
        self.tol = tol
        self.runner_type = type

        self.output_col = kwargs.get('output_col', None)
        if self.output_col is None:
            output_col = [col for col in self.df.columns if 'output' in col]
            if len(output_col) > 1:
                warnings.warn(
                    f'WHEN SETTING OUTPUT COL THERE WERE MORE THAN ONE COLUMNS WITH '
                    f'"output" STRING: {output_col}, TAKING FIRST: {output_col[0]}'
                )
            self.output_col = output_col[0]

    def single_code_run(self, run_dir: str, params: dict = None) -> dict:
        """
        Look up the output corresponding to the given params.
        Uses tolerance-based matching for numeric columns.
        """
        start = time()
        # Validate keys
        if not all(k in self.df.columns for k in params.keys()):
            return {'success': False, self.output_col: np.nan}

        # Start with all rows valid
        mask = np.ones(len(self.df), dtype=bool)

        for k, v in params.items():
            col = self.df[k]

            # Numeric tolerance-based matching
            if np.issubdtype(col.dtype, np.number):
                mask &= np.isclose(col.values, float(v), atol=self.tol, rtol=0.0)

            # Exact match for non-numeric
            else:
                mask &= (col == v)

        # No match found
        if not mask.any():
            return {'success': False, self.output_col: np.nan}

        # Extract the output value (first match)
        value = self.df.loc[mask, self.output_col].iloc[0]
        
        if self.remove_from_cache:
            # remove the value from the df, ONLY REMOVES IT FROM THE CACHED VARIABLE ON ONE WORKER, THE CACHED VARIABLE ON OTHER WORKERS ARE NOT AFFECTED.
            _df_cache = _df_cache.loc[~mask].copy()
            self.df = _df_cache

        end = time()
        print('csv runner time taken, single code run: ', end-start)
        if self.remove_run_dir:
            shutil.rmtree(run_dir)
        return {'success': True, self.output_col: value}
