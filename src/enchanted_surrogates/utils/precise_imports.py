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

def import_parser(type, parser_kwargs):
    """
    Dynamically imports and instantiates a parser class based on naming convention.

    Parameters
    ----------
    type : str
        The name of the parser (in snake_case or PascalCase).
    parser_kwargs : dict
        Keyword arguments to pass to the sampler constructor.

    Returns
    -------
    object
        An instance of the parser class.

    Raises
    ------
    ImportError
        If the module or class cannot be found.
    """
    type_snake, type_pascal = get_snake_and_pascal(type)
    eps = load_plugins()

    if type_snake in eps:
        parser = eps[type_snake](**parser_kwargs)
    else:
        parser = getattr(importlib.import_module(f'enchanted_surrogates.parsers.{type_snake}'),type_pascal)(**parser_kwargs)
    return parser
    
if __name__ == "__main__":
    print(src_dir)
