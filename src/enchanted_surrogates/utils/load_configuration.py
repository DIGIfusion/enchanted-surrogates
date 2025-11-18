import yaml
import argparse
import os
import warnings
from typing import Optional

def load_configuration(config_path: str) -> argparse.Namespace:
    """
    Loads configuration from a YAML file.

    Args:
        config_path (str): Path to the configuration YAML file.

    Returns:
        argparse.Namespace: Namespace containing the configuration parameters.
    """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    config = argparse.Namespace(**config)
    config.executor["config_filepath"] = config_path

    # In case sampler or runner is defined outside the executor.
    # This only works for non nested workflows.
    if 'sampler_config' not in config.executor:
        if getattr(config, 'sampler', None):
            config.executor['sampler_config'] = config.sampler
        elif 'sampler' in config.executor:
            config.executor['sampler_config'] = config.executor.pop('sampler')
    if 'runner_config' not in config.executor:
        if getattr(config, 'runner', None):
            config.executor['runner_config'] = config.runner
        elif 'runner' in config.executor:
            config.executor['runner_config'] = config.executor.pop('runner')
    print(config)
    return config


def load_from_dir(dir_path: str) -> Optional[argparse.Namespace]:
    """
    Loads configuration from a directory containing a single YAML file.

    Args:
        dir_path (str): Path to the directory to search for YAML files.

    Returns:
        argparse.Namespace or None: Loaded configuration if exactly one YAML file is found,
                                    otherwise None with a warning.
    """
    # Collect YAML files in the directory
    yaml_files = [
        f for f in os.listdir(dir_path)
        if f.lower().endswith((".yaml", ".yml"))
    ]

    if len(yaml_files) == 0:
        warnings.warn(f"No YAML files found in directory: {dir_path}")
        return None
    elif len(yaml_files) > 1:
        warnings.warn(f"Multiple YAML files found in directory: {dir_path}. Expected only one.")
        return None

    # Exactly one YAML file found
    config_path = os.path.join(dir_path, yaml_files[0])
    return load_configuration(config_path)
    
