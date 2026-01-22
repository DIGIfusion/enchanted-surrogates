import pandas as pd
import numpy as np
from .base_runner import Runner

class CsvRunner(Runner):
    """
    A runner that simply looks up outputs from a CSV table.
    """    
    def __init__(self, csv_path, tol=1e-8, type='CsvRunner'):
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        self.tol = tol
        self.runner_type = type

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

        # Validate keys
        if not all(k in self.df.columns for k in params.keys()):
            return {'success': False, 'output': np.nan}

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
            return {'success': False, 'output': np.nan}

        # Extract the output value (first match)
        value = self.df.loc[mask, self.output_col].iloc[0]

        return {'success': True, 'output': value}
