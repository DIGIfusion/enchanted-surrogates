import re
import importlib
from enchanted_surrogates import load_plugins

# All imported module entry points are stored to ensure modules are only imported once.
# Mapping from [module_name (str) -> class]
__module_entry_points = None

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
        raise ValueError(
            f"Input string '{string}' must be in either snake_case or PascalCase format.")
    return string_snake, string_pascal

def clear_import_cache():
    """
    Clears everything from import cache.
    """
    global __module_entry_points
    __module_entry_points = None

def cached_import(type_name: str, base_module: str):
    """
    Imports a module entry point by name. All imports are cached, so overhead after first import is minimal.
    
    Parameters
    ----------
    type_name : str
        Class name of the type that should be imported.
    base_module : str
        For non-plugin modules, the module name. Eg. 'samplers'

    Returns
    -------
    class
        Class entry point. To construct an instance, use eg. cached_import(type_name, base_module)(**kwargs)

    Raises
    ------
    ImportError
        If the module or class cannot be found.
    """
    type_snake, type_pascal = get_snake_and_pascal(type_name)

    global __module_entry_points
    if not __module_entry_points:
        __module_entry_points = load_plugins()

    if type_snake in __module_entry_points:
        return __module_entry_points[type_snake]
    else:
        imported_type = getattr(importlib.import_module(f"enchanted_surrogates.{base_module}.{type_snake}"), type_pascal)
        __module_entry_points[type_snake] = imported_type
        return imported_type

def import_sampler(sampler_type, sampler_config):
    """
    Dynamically imports and instantiates a sampler class based on naming convention.

    Parameters
    ----------
    sampler_type : str
        The name of the sampler (in snake_case or PascalCase).
    sampler_config : dict
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
    config: dict = sampler_config.copy()
    config.pop("type", None)

    sampler = cached_import(sampler_type, "samplers")(**config)
    return sampler


def import_runner(runner_type, runner_config):
    """
    Dynamically imports and instantiates a runner class based on naming convention.

    Parameters
    ----------
    runner_type : str
        The name of the sampler (in snake_case or PascalCase).
    sampler_config : dict
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
    runner = cached_import(runner_type, "runners")(**runner_config)
    return runner


def import_executor(executor_type, executor_config):
    """
    Dynamically imports and instantiates a executor class based on naming convention.

    Parameters
    ----------
    executor_type : str
        The name of the executor (in snake_case or PascalCase).
    executor_config : dict
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
    executor = cached_import(executor_type, "executors")(**executor_config)
    return executor

def import_parser(parser_type, parser_config):
    """
    Dynamically imports and instantiates a parser class based on naming convention.

    Parameters
    ----------
    parser_type : str
        The name of the parser (in snake_case or PascalCase).
    parser_config : dict
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
    parser = cached_import(parser_type, "parsers")(**parser_config)
    return parser
