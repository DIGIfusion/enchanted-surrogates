import pandas as pd
import numpy as np
from .base_runner import Runner
import shutil
from time import time

from enchanted_surrogates.utils.logger import get_logger

log = get_logger(__name__)

_df_cache = None

class CsvRunner(Runner):
    """
    A runner that simply looks up outputs from a CSV table.
    """    
    def __init__(self, csv_path, tol=1e-5, **kwargs):
        self.csv_path = csv_path
        self.remove_from_cache = kwargs.get('remove_from_cache', False)
        global _df_cache

        if _df_cache is None:
            print("Loading CSV once on this worker...")
            _df_cache = pd.read_csv(self.csv_path)

        self.df = _df_cache
        
        self.tol = tol

        self.output_variables = kwargs.get('output_variables', None)
        if self.output_variables is None:
            self.output_variables = [col for col in self.df.columns if '_active_output' in col]

    def single_code_run(self, run_dir: str, params: dict = None) -> dict:
        """
        Look up the output corresponding to the given params.
        Uses tolerance-based matching for numeric columns.
        """
        global _df_cache

        start = time()
        # Validate keys
        missing = [k for k in params.keys() if k not in self.df.columns]
        if len(missing) > 0:
            log.error(f'''
Not all the parameters in the origional csv:
csv_columns: {_df_cache.columns}
parameters: {params.keys()}
missing: {missing}
                      ''')
            return {'success': False, **{out_var: np.nan for out_var in self.output_variables}}

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
            best_idx, best_dist, comparison = self._find_closest_match(params)
            er_msg = f'''
No exact match found.
Requested params vs closest row (index {best_idx}, distance {best_dist}):
            
            '''

            for k, info in comparison.items():
                er_msg = er_msg + f'''
                
{k}: requested={info['requested']} |
closest={info['closest']} |
difference={info['difference']}

'''         
            log.error(er_msg)
            return {'success': False, **{out_var: np.nan for out_var in self.output_variables}}

        # Extract the row index of the selected value
        row_index = self.df.loc[mask].index[0]

        # Extract the value
        output_values = self.df.loc[row_index, self.output_variables].to_dict()

        if self.remove_from_cache:
            # Remove ONLY that row
            _df_cache = _df_cache.drop(index=row_index)
            self.df = _df_cache

        end = time()
        log.debug(f'csv runner time taken, single code run: {end-start} seconds')
        return {'success': True, **output_values}
    
    def _find_closest_match(self, params):
        """
        Returns:
            best_idx: index of closest row
            best_dist: total distance score
            comparison: dict mapping each param -> (requested, closest_value, difference)
        """
        df = self.df
        distances = []

        # Compute total distance for each row
        for idx, row in df.iterrows():
            dist = 0.0
            for k, v in params.items():
                col_val = row[k]

                if np.issubdtype(df[k].dtype, np.number):
                    try:
                        diff = abs(float(col_val) - float(v))
                    except Exception:
                        diff = np.inf
                    dist += diff
                else:
                    diff = 0.0 if col_val == v else 1.0
                    dist += diff

            distances.append((dist, idx))

        # Pick best match
        distances.sort(key=lambda x: x[0])
        best_dist, best_idx = distances[0]
        best_row = df.loc[best_idx]

        # Build side‑by‑side comparison
        comparison = {}
        for k, v in params.items():
            row_val = best_row[k]

            if np.issubdtype(df[k].dtype, np.number):
                try:
                    diff = float(row_val) - float(v)
                except Exception:
                    diff = None
            else:
                diff = None if row_val == v else "≠"

            comparison[k] = {
                "requested": v,
                "closest": row_val,
                "difference": diff,
            }

        return best_idx, best_dist, comparison

