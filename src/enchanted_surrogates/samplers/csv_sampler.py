from enchanted_surrogates.samplers.base_sampler import Sampler
import warnings
import pandas as pd

class CsvSampler(Sampler):
    """
    A sampler that loads data from a CSV file and returns batches of samples.

    Parameters:
        csv_path (str): Path to the CSV file.
        budget (int, optional): Maximum number of samples to return. Defaults to the number of rows in the CSV.
        batch_size (int, optional): Number of samples per batch. Defaults to the full budget.
        parameters (list, optional): List of column names to include in the samples. Defaults to all columns.
        **kwargs: Additional keyword arguments (currently unused).
    """

    def __init__(self, csv_path, budget=None, batch_size=None, parameters=None, **kwargs):
        """
        Initializes the CsvSampler by loading the CSV and preparing sample batches.
        """
        self.df = pd.read_csv(csv_path)

        self.budget = budget if budget is not None else len(self.df)
        self.batch_size = batch_size if batch_size is not None else self.budget
        self.parameters = parameters if parameters is not None else self.df.columns

        self.batch_number = 0
        self.samples = self.df[self.parameters].to_dict(orient='records')

    def get_next_samples(self) -> list[dict]:
        """
        Returns the next batch of samples based on the batch size.

        Returns:
            list[dict]: A list of dictionaries representing the next batch of samples.
        """
        start = self.batch_number * self.batch_size
        end = min((self.batch_number + 1) * self.batch_size, self.budget)
        samples = self.samples[start:end]
        self.batch_number += 1
        self.submitted += len(samples)
        return samples

    def register_future(self, future):
        """
        Placeholder for registering a future. Not used in this sampler.

        Parameters:
            future: A future object (ignored).

        Returns:
            None
        """
        return None

    def register_futures(self, futures):
        """
        Placeholder for registering multiple futures. Not used in this sampler.

        Parameters:
            futures: A list of future objects (ignored).

        Returns:
            None
        """
        return None