import importlib.util

def is_package_available(package_name: str) -> bool:
    """
    Checks whether a Python package or module is available in the current environment.

    Parameters
    ----------
    package_name : str
        The fully qualified name of the package or module to check (e.g., 'dask.distributed').

    Returns
    -------
    bool
        True if the package is available and importable, False otherwise.

    Notes
    -----
    This function uses `importlib.util.find_spec()` to avoid importing the module directly,
    making it safe for use in plugin systems or conditional imports.
    """
    return importlib.util.find_spec(package_name) is not None
