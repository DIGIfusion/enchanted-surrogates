import pandas as pd
from sklearn.model_selection import train_test_split
import torch


def data_split(
    df: pd.DataFrame, 
    valid_size: float, 
    test_size: float,
    *args, **kwargs,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """ Train should be of length 0 when starting, valid and test should be of size given in argument """

    if  valid_size > 1 or test_size > 1:
        raise ValueError("Allowed data_args must be <1")
    elif  valid_size + test_size >= 1:
        raise ValueError("Sum of allowed data_args must be <1")
    
    poolsize = 1 - valid_size - test_size
    validsizeprime = (valid_size) / (1 - poolsize)     
    pool, tmp = train_test_split(df, test_size=1 - poolsize)
    valid, test = train_test_split(tmp, test_size=1 - validsizeprime)
    train = valid.sample(0)
    return train, valid, test, pool


class Scaler:
    """
    Implements the sklearn StandardScaler functionality but in torch.
    """

    def __init__(self):
        pass

    def fit(self, data: torch.Tensor):
        self.mean_ = data.mean(dim=0)
        self.scale_ = data.std(dim=0)
        print('scales: ',self.scale_, "means: ", self.mean_)

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        return (data - self.mean_) / self.scale_

    def fit_transform(self, data: torch.Tensor) -> torch.Tensor:
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        return self.scale_ * data + self.mean_

    def find_zeros(self):
        return torch.any(self.scale_<1.e-10).item()


def apply_scaler(train, valid, test, pool, scaler=None, op="transform"):
    """
    Utility to apply data scaling. It will instantiate a scaler, if this is not passed as an argument.
    It can either apply the direct or inverse transform.
    """
    if scaler is None and op == "inverse_transform":
        raise ValueError(
            "`Scaler` is not instantiated. Running `inverse_transform` will have the opposite effect to what you are thinking."
        )
    if scaler is None:
        scaler = Scaler()
        scaler.fit(train)
    operation = getattr(scaler, op)
    train = operation(train)
    valid = operation(valid)
    test = operation(test)
    pool = operation(pool)
    return train, valid, test, pool, scaler


class Loader:
    def __init__(self, data_path: str):
        pass

    def load_data(self):
        pass


class PickleLoader(Loader):
    """
    Loads from different extensions
    """

    def __init__(self, data_path: str):
        """
        Args:
            data_path (str): The path where the data is stored. Accepted formats: csv, pkl, h5.
        """
        self.data_path = data_path

    def load_data(self):
        return pd.read_pickle(self.data_path)


class CSVLoader(Loader):
    """
    Loads from different extensions
    """

    def __init__(self, data_path: str):
        """
        Args:
            data_path (str): The path where the data is stored. Accepted formats: csv, pkl, h5.
        """
        self.data_path = data_path

    def load_data(self):
        return pd.read_csv(self.data_path)


class HDFLoader(Loader):
    # ToDo: some data might come with keys, this needs to be handled
    """
    Loads from different extensions
    """

    def __init__(self, data_path: str):
        """
        Args:
            data_path (str): The path where the data is stored. Accepted formats: csv, pkl, h5.
        """
        self.data_path = data_path

    def load_data(self):
        return pd.read_hdf(self.data_path)
