import os
import sys
import logging
import datetime

LOGGER_NAME = "enchanted_surrogates"

def setup_logging(log_level: str, log_dir: str):
    """
    Configures a project-level logger that writes to stdout and to a file.

    Args:
        log_level (str): The logging level of the log file.

        log_dir (str): Path to the directory where the log file will be created.
    
    Note:
        Only run this function once.
    """
    logger = logging.getLogger(LOGGER_NAME)

    time = datetime.datetime.now()

    file_name = time.strftime("%Y%m%d_%H%M%S_%f") # ISO 8601 with microseconds
    file_path = os.path.join(log_dir, f"{file_name}.log")

    file_handler = logging.FileHandler(file_path)
    file_handler.setLevel(log_level.upper())

    stdout_handler = logging.StreamHandler(stream=sys.stdout)

    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)

def get_logger(name: str) -> logging.Logger:
    """
    Returns a child instance of the project-level logger for use in logging.

    Args:
        name (str): Unique identifier for the logger, such as  __name__.

    Returns:
        logging.Logger: A configured logger.
    """
    return logging.getLogger(LOGGER_NAME).getChild(name)
