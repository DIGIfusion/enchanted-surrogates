import os
import re
from pathlib import Path
from enchanted_surrogates.supervisor.supervisor import Supervisor


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
