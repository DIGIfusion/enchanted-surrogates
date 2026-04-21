import yaml
import argparse
from enchanted_surrogates.utils.logger import get_logger

log = get_logger(__name__)

def load_configuration(config_path: str) -> argparse.Namespace:
    """
    Loads configuration from a YAML file.

    Args:
        config_path (str): Path to the configuration YAML file.

    Returns:
        argparse.Namespace: Namespace containing the configuration parameters.
    """
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    config = argparse.Namespace(**config)
    config.supervisor["config_filepath"] = config_path

    log.debug(config)
    return config
