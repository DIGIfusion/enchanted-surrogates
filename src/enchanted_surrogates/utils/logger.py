import multiprocessing
import os
import sys
import logging
import logging.handlers

import datetime
import threading

# Multiprocessing queue for QueueHandler, initialized in setup_logging()
_log_queue = None

# Listener thread for log events, initialized in setup_logging()
_logger_thread = None

def logger_thread(queue):
    """
    Listens for log events from multiple processes.

    Note:
        Should be spawned and terminated in run.py.
    """
    while True:
        record = queue.get()
        if not record:
            break

        logger = logging.getLogger(record.name)
        logger.handle(record)


def log_queue():
    global _log_queue
    return _log_queue

def setup_logging(log_level: str, log_dir: str):
    """
    Configures a project-level logger that writes to stdout and to a file.

    Args:
        log_level (str): The logging level of the log file.

        log_dir (str): Path to the directory where the log file will be created.
    
    Note:
        Only run this function once.
    """

    
    logger = logging.getLogger()

    time = datetime.datetime.now()
    file_name = time.strftime("%Y%m%d_%H%M%S_%f") # ISO 8601 with microseconds
    file_path = os.path.join(log_dir, f"{file_name}.log")

    logger.setLevel(log_level)

    logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s] %(message)s")


    # Initialize log queue
    mgr = multiprocessing.Manager()
    global _log_queue
    _log_queue = mgr.Queue()

    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(logFormatter)
    logger.addHandler(file_handler)

    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    stdout_handler.setFormatter(logFormatter)
    logger.addHandler(stdout_handler)

    global _logger_thread
    _logger_thread = threading.Thread(target=logger_thread, args=(log_queue(),))
    _logger_thread.start()

def shutdown_logging():
    """
    Terminates the logging listener thread.
    """
    log_queue().put(None)
    global _logger_thread
    _logger_thread.join()

def setup_subprocess_logging(queue):
    """
    Adds a queue handler for the logger. Dask client should run this to make logging work.
    
    :param queue: The shared log queue
    """
    logger = logging.getLogger()

    handler = logging.handlers.QueueHandler(queue)
    logger.addHandler(handler)

    logger.setLevel(logging.DEBUG)



def get_logger(name: str) -> logging.Logger:
    """
    Returns a child instance of the project-level logger for use in logging.

    Args:
        name (str): Unique identifier for the logger, such as  __name__.

    Returns:
        logging.Logger: A configured logger.
    """
    return logging.getLogger().getChild(name)
    


