import logging

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


def get_logger(name: str) -> logging.Logger:
    """
    Returns a child instance of the project-level logger for use in logging.

    Args:
        name (str): Unique identifier for the logger, such as  __name__.

    Returns:
        logging.Logger: A configured logger.
    """
    return logging.getLogger().getChild(name)
