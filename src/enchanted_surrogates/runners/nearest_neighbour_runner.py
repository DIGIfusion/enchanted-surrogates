from .base_runner import Runner
import pandas as pd
import warnings
import numpy as np
from dask.distributed import print

class NearestNeighbourRunner(Runner):
    """
    Nearest neighbour lookup with:
    - exact match detection
    - Euclidean nearest neighbour fallback
    - tolerance-based tie detection
    """

    def __init__(self, data_csv, output_label=None, *args, **kwargs):
        self.data_csv = data_csv
        
        self.data_df = pd.read_csv(self.data_csv)
        self.output_label = output_label
        
        if output_label is None:
            output_col = [col for col in self.data_df.columns if 'output' in col]
            if len(output_col) > 1:
                warnings.warn(
                    f'MORE THAN ONE OUTPUT DETECTED: {output_col}. '
                    f'TAKING FIRST: {output_col[0]}'
                )
            self.output_label = output_col[0]
        
        self.fixed_params = kwargs.get('fixed_params', {})
        
        self.tag = kwargs.get('tag', '')
             
    def single_code_run(self, run_dir: str, params: dict = None, tol: float = 1e-12) -> dict:
        """
        Find the nearest neighbour in self.data_df based on the columns
        provided in params, and return its output_label value.

        - If multiple rows match params exactly, warn and take the first.
        - If multiple nearest neighbours are within tolerance of the minimum
          distance, warn and take the first.
        """
        print('debug nearest neighbour runner. run_dir:', run_dir)
        params = {**params, **self.fixed_params}
        if params is None or len(params) == 0:
            raise ValueError("params must be a non-empty dict of column:value pairs")

        # Ensure all keys exist in the dataframe
        for key in params:
            if key not in self.data_df.columns:
                raise KeyError(f"Column '{key}' not found in dataframe")

        # Ensure all params columns are numeric
        for col in params:
            if not np.issubdtype(self.data_df[col].dtype, np.number):
                raise TypeError(
                    f"Column '{col}' is non-numeric; cannot compute distances."
                )

        # --- 1. Check for exact matches -----------------------------------------
        mask = np.ones(len(self.data_df), dtype=bool)
        for col, val in params.items():
            mask &= (self.data_df[col] == val)

        matching_rows = self.data_df[mask]

        if len(matching_rows) > 1:
            warnings.warn(
                f"Multiple exact matches found for params {params}. "
                f"Taking the first match (index {matching_rows.index[0]})."
            )
            nearest_neighbour = matching_rows.iloc[0][self.output_label]
            return {'success': True, self.output_label+self.tag: nearest_neighbour}

        elif len(matching_rows) == 1:
            nearest_neighbour = matching_rows.iloc[0][self.output_label]
            return {'success': True, self.output_label+self.tag: nearest_neighbour}

        # --- 2. No exact match → compute Euclidean nearest neighbour ------------
        df_sub = self.data_df[list(params.keys())]

        # Explicit float conversion for safety
        target = np.array([float(params[col]) for col in df_sub.columns])

        distances = np.linalg.norm(df_sub.values - target, axis=1)
        min_dist = np.min(distances)

        # Identify all rows within tolerance of the minimum distance
        nearest_indices = np.where(np.isclose(distances, min_dist, atol=tol))[0]

        idx = nearest_indices[0]
        nearest_neighbour = self.data_df.loc[idx, self.output_label]

        if len(nearest_indices) > 1:
            warnings.warn(
                f"Multiple nearest neighbours found within tolerance {tol} "
                f"of minimum distance {min_dist}. "
                f"Possible nearest neighbout outputs: {self.data_df.loc[nearest_indices, self.output_label].tolist()}"
                f"Taking the first (index {nearest_neighbour})."
            )
        return {
            'success': True,
            self.output_label+self.tag: nearest_neighbour
        }