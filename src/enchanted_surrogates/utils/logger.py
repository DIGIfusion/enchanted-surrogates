import os
import sys
import logging


def setup_logging(log_level: str, log_dir: str, log_file: str):
    """
    Configures a project-level logger that writes to stdout and to a file.

    Args:
        log_level (str): The logging level of the log file.

        log_dir (str): Path to the directory where the log file(s) will be created.

        log_file_name (str): Name of the log file.
    
    Note:
        Only run this function once (except dask workers run this also).
    """

    logger = logging.getLogger()

    file_path = os.path.join(log_dir, f"{log_file}")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger.setLevel(log_level)

    logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s] %(message)s")

    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(logFormatter)
    logger.addHandler(file_handler)

    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    stdout_handler.setFormatter(logFormatter)
    logger.addHandler(stdout_handler)

def get_logger(name: str) -> logging.Logger:
    """
    Returns a child instance of the project-level logger for use in logging.

    Args:
        name (str): Unique identifier for the logger, such as  __name__.

    Returns:
        logging.Logger: A configured logger.
    """
    return logging.getLogger().getChild(name)
    


