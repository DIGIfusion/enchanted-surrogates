import os
import sys
import logging

_log_dir=None

# TODO: Better solution for getting the log dir, give executors the log dir when supervisor is ready?
def get_log_dir():
    global _log_dir
    return _log_dir

def setup_logging(log_level: str, log_dir: str, log_file: str):
    """
    Configures a project-level logger that writes to stdout and to a file.

    Args:
        log_level (str): The logging level of the log file.

        log_dir (str): Path to the directory where the log file will be created.

        log_file_name (str): Name of the log file.
    
    Note:
        Only run this function once (except dask workers run this also).
    """

    # Store the log dir
    global _log_dir
    if not _log_dir:
        _log_dir = log_dir
    
    logger = logging.getLogger()

    # time = datetime.datetime.now()
    # file_name = time.strftime("%Y%m%d_%H%M%S_%f") # ISO 8601 with microseconds
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
    


