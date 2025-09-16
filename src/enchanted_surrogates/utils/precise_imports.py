import re
import importlib
import os, sys
from enchanted_surrogates import load_plugins
src_dir = os.path.dirname(os.sep.join(os.path.normpath(__file__).split(os.sep)[:__file__.split(os.sep).index("enchanted_surrogates") + 1]))
sys.path.append(src_dir)

def detect_case_style(s):
    """
    Detects the naming convention of a given string.

    Parameters
    ----------
    s : str
        The string to analyze.

    Returns
    -------
    str
        One of: 'snake_case', 'camelCase', 'PascalCase', 'kebab-case', or 'unknown'.
    """

    if "_" in s and s.lower() == s:
        return "snake_case"
    elif re.match(r'^[a-z]+(?:[A-Z][a-z]*)+$', s):
        return "camelCase"
    elif re.match(r'^[A-Z][a-z]+(?:[A-Z][a-z]*)*$', s):
        return "PascalCase"
    elif "-" in s and s.lower() == s:
        return "kebab-case"
    else:
        return "unknown"

def snake_to_pascal(s):
    """
    Converts a snake_case string to PascalCase.

    Parameters
    ----------
    s : str
        A string in snake_case format.

    Returns
    -------
    str
        The converted PascalCase string.
    """

    return ''.join(word.title() for word in s.split('_'))

def camel_or_pascal_to_snake(s):
    """
    Converts a camelCase or PascalCase string to snake_case.

    Parameters
    ----------
    s : str
        A string in camelCase or PascalCase format.

    Returns
    -------
    str
        The converted snake_case string.
    """
    return re.sub(r'(?<!^)(?=[A-Z])', '_', s).lower()

def get_snake_and_pascal(string):
    """
    Given a string in either snake_case or PascalCase, returns both formats.

    Parameters
    ----------
    string : str
        The input string to normalize.

    Returns
    -------
    tuple of str
        (snake_case version, PascalCase version)

    Notes
    -----
    If the input is already in snake_case or PascalCase, it is preserved.
    camelCase and other formats are not supported.
    """
    string_case = detect_case_style(string)
    if string_case == 'snake_case':
        string_snake = string
        string_pascal = snake_to_pascal(string)
    elif string_case == 'PascalCase':
        string_snake = camel_or_pascal_to_snake(string)
        string_pascal = string
    else:
        raise ValueError(f"Input string '{string}' must be in either snake_case or PascalCase format.")
    return string_snake, string_pascal

def import_sampler(type, sampler_kwargs):
    """
    Dynamically imports and instantiates a sampler class based on naming convention.

    Parameters
    ----------
    type : str
        The name of the sampler (in snake_case or PascalCase).
    sampler_kwargs : dict
        Keyword arguments to pass to the sampler constructor.

    Returns
    -------
    object
        An instance of the sampler class.

    Raises
    ------
    ImportError
        If the module or class cannot be found.
    """
    type_snake, type_pascal = get_snake_and_pascal(type)
    eps = load_plugins()

    if type_snake in eps:
        sampler = eps[type_snake](**sampler_kwargs)
    else:
        sampler = getattr(importlib.import_module(f'enchanted_surrogates.samplers.{type_snake}'),type_pascal)(**sampler_kwargs)
    return sampler

def import_runner(type, runner_kwargs):
    """
    Dynamically imports and instantiates a runner class based on naming convention.

    Parameters
    ----------
    type : str
        The name of the sampler (in snake_case or PascalCase).
    sampler_kwargs : dict
        Keyword arguments to pass to the sampler constructor.

    Returns
    -------
    object
        An instance of the sampler class.

    Raises
    ------
    ImportError
        If the module or class cannot be found.
    """
    type_snake, type_pascal = get_snake_and_pascal(type)
    eps = load_plugins()

    if type_snake in eps:
        sampler = eps[type_snake](**runner_kwargs)
    else:
        sampler = getattr(importlib.import_module(f'enchanted_surrogates.runners.{type_snake}'),type_pascal)(**runner_kwargs)
    return sampler
    
    
def import_executor(type, executor_kwargs):
    """
    Dynamically imports and instantiates a executor class based on naming convention.

    Parameters
    ----------
    type : str
        The name of the executor (in snake_case or PascalCase).
    executor_kwargs : dict
        Keyword arguments to pass to the executor constructor.

    Returns
    -------
    object
        An instance of the executor class.

    Raises
    ------
    ImportError
        If the module or class cannot be found.
    """
    type_snake, type_pascal = get_snake_and_pascal(type)
    eps = load_plugins()

    if type_snake in eps:
        cls = eps[type_snake](**executor_kwargs)
    else:
        cls = getattr(
            importlib.import_module(
                f'enchanted_surrogates.executors.{type_snake}'), type_pascal)(**executor_kwargs)
    return cls
