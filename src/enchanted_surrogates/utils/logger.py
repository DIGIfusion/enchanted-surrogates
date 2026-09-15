import logging
import os, sys

LOG_FORMAT = "%(asctime)s [%(levelname)-5.5s] %(message)s"


class Singleton(type):
    _instance = None

    def __call__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instance

    def reset(cls):
        cls._instance = None


# Stores log level, log_dir etc.
class LoggerConfig(metaclass=Singleton):
    def __init__(self, log_level=logging.INFO, log_dir=None, format=LOG_FORMAT):
        self.log_level = log_level
        self.log_dir = log_dir
        self.format = format


def setup_logging(
    config: LoggerConfig,
    console_handler: logging.Handler,
    file_handler: logging.Handler,
):
    """
    Sets up logging to console and file
    """
    logger = logging.getLogger()
    logger.propagate = False

    log_level = config.log_level
    logger.setLevel(log_level)

    logFormatter = logging.Formatter(config.format)

    logger.handlers.clear()

    console_handler.setFormatter(logFormatter)
    logger.addHandler(console_handler)

    file_handler.setFormatter(logFormatter)
    logger.addHandler(file_handler)


def setup_logger(base_run_dir: str, log_level: str, log_dir: str):
    """
    Prepares the logging module and creates log directory

    Args:
        base_run_dir (str): The base run directory from supervisor
        log_level (str): Logging level to use, for example INFO or DEBUG
    """
    log_dir = os.path.join(base_run_dir, log_dir)
    log_file = os.path.join(log_dir, "main.log")

    # Store to logger config
    config = LoggerConfig(log_level=log_level, log_dir=log_dir)

    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)

    setup_logging(
        config,
        logging.StreamHandler(stream=sys.stdout),
        logging.FileHandler(filename=log_file),
    )


def get_logger(name: str) -> logging.Logger:
    """
    Returns a child instance of the project-level logger for use in logging.

    Args:
        name (str): Unique identifier for the logger, such as  __name__.

    Returns:
        logging.Logger: A configured logger.
    """
    return logging.getLogger().getChild(name)
