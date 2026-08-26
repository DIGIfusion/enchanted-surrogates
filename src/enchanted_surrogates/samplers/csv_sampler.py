"""
## Overview

The CSV sampler loads samples from a CSV file and yields them in batches.
Each row in the CSV is converted to a dictionary representing a sample.

---
"""
import pandas as pd
from enchanted_surrogates.samplers.base_sampler import Sampler


class CsvSampler(Sampler):
    """
    ## Configuration

    To use the `CsvSampler`, specify it in the configuration file as in following example:

    ```yaml
    sampler:
        type: CsvSampler
        csv_path: path/to/file.csv
        input_columns: ['x', 'y']
        batch_size: 10
    ```

    Attributes:
        csv_path (str): Path to the CSV file.
        input_columns (list[str], optional): Columns to use as inputs. If None, all columns are used.
        batch_size (int): Number of samples returned per batch (defaults to full budget).
        budget (int): Total number of samples from the CSV.
        submitted (int): Counter tracking how many samples have been yielded so far.

    ---

    ## Assumptions and Notes

     - Each row in the CSV file is converted to a dictionary.
     - If input_columns is specified, only those columns are included in each sample dict.
     - If input_columns is not specified, all columns are included.
     - The sampler yields samples sequentially from the CSV.
     - The budget is determined by the number of rows in the CSV file.

    ---
    """

    def __init__(self, csv_path: str, **kwargs):
        """
        Initializes the CsvSampler.

        Args:
            csv_path (str): Path to the CSV file.
            input_columns (list[str], optional): Specific columns to extract as inputs.
                If None, all columns are used.
            batch_size (int, optional): Number of samples returned per call
                to `get_next_samples`. Defaults to the full sampling budget.
        """
        self.csv_path = csv_path
        self.input_columns = kwargs.get("input_columns", None)

        # Load CSV
        df = pd.read_csv(csv_path)

        # Filter to input columns if specified
        if self.input_columns is not None:
            df = df[self.input_columns]

        self.parameters = df.columns.tolist()

        # Convert to list of dicts (one dict per row)
        self.samples = df.to_dict(orient='records')
        self.budget = len(self.samples)
        self.batch_size = kwargs.get("batch_size", self.budget)
        self.current_index = 0

    def get_next_samples(self) -> list[dict]:
        """
        Generates the next batch of samples from the CSV.

        Returns:
            list[dict]: A batch of sample dictionaries from the CSV.
        """
        end_index = min(self.current_index + self.batch_size, len(self.samples))
        list_param_dicts = self.samples[self.current_index:end_index]
        self.submitted += len(list_param_dicts)
        self.current_index = end_index
        return list_param_dicts

    def register_future(self, _future):
        """
        Registers a completed or scheduled evaluation.

        This method is part of the sampler interface but is not used by
        the CsvSampler, as sampling does not depend on evaluation results.

        Args:
            _future: A future or handle representing an asynchronous evaluation.

        Returns:
            None
        """
        return None

    def register_futures(self, _futures):
        """
        Registers multiple completed or scheduled evaluations.

        This method is part of the sampler interface but is not used by
        the CsvSampler. It is implemented as a no-op.

        Args:
            _futures: An iterable of futures or handles representing asynchronous
                evaluations.

        Returns:
            None
        """
        return None
