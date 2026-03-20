import os
import re
import pandas as pd
from pathlib import Path


def get_run_dir_count(path: str, pattern: str = r"^d\d+_b\d+_r\d+_s\d+$"):
    """
    Get the amount of subdirectories inside the given path that match the pattern filter.

    Arguments:
    path (str): Path to directory that is looked through.
    pattern (str): Regex filter pattern. Defaults to d#_b#_r# to find run directories
    """
    base = Path(path)
    regex = re.compile(pattern)

    return sum(1 for p in base.iterdir() if p.is_dir() and regex.match(p.name))

def read_summary_file(path: str, filename: str = "enchanted_dataset.csv"):
    file = os.path.join(path, filename)
    assert os.path.exists(file)    
    return pd.read_csv(os.path.join(path, filename))
