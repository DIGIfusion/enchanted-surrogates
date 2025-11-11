import yaml
import argparse

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
