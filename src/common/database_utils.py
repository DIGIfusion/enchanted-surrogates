import pandas as pd
from sklearn.model_selection import train_test_split

def data_split(df: pd.DataFrame, train_size: float, valid_size: float, test_size: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if train_size > 1 or valid_size > 1 or test_size > 1:
        raise ValueError("Allowed data_args must be <1")
    if train_size+valid_size+test_size>=1:
        raise ValueError("Sum of allowed data_args must be <1")
    poolsize = 1- train_size-valid_size-test_size
    poolsizeprime = poolsize/(1-train_size)
    validsizeprime = (valid_size)/(1-poolsizeprime)/(1-train_size)  
    # TODO: replace train test split with own thing 
    train, tmp = train_test_split(df, test_size=1-train_size, random_state=42)
    pool, tmp = train_test_split(tmp, test_size=1-poolsizeprime, random_state=42)
    valid, test = train_test_split(tmp, test_size=1-validsizeprime, random_state=42)
    return train, valid, test, pool


class Loader:
    def __init__(self, data_path: str):
        pass

    def load_data(self):
        pass


class PickleLoader(Loader):
    """
    Loads from different extensions
    """
    def __init__(self,data_path: str):
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
    def __init__(self,data_path: str):
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
    def __init__(self,data_path: str):
        """
        Args:
            data_path (str): The path where the data is stored. Accepted formats: csv, pkl, h5.
        """        
        self.data_path = data_path

    def load_data(self):
        return pd.read_hdf(self.data_path)
