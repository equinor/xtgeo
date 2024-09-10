from __future__ import annotations

import logging
import time
from functools import wraps


def null_logger(name: str) -> logging.Logger:
    """
    Create and return a logger with a NullHandler.

    This function creates a logger for the specified name and attaches a
    NullHandler to it. The NullHandler prevents logging messages from being
    automatically output to the console or other default handlers. This is
    particularly useful in library modules where you want to provide the
    users of the library the flexibility to configure their own logging behavior.

    Args:
        name (str): The name of the logger to be created. This is typically
                    the name of the module in which the logger is
                    created (e.g., using __name__).

    Returns:
        logging.Logger: A logger object configured with a NullHandler.

    Example:
        # In a library module
        logger = null_logger(__name__)
        logger.info("This info won't be logged to the console by default.")
    """

    logger = logging.getLogger(name)
    logger.addHandler(logging.NullHandler())
    return logger


def functimer(func):
    """A decorator function to measure the execution time of a function.

    Will emit a log message with the execution time in seconds, and is primarily
    for developer use.

    Usage is simple, just add the decorator to the function you want to measure:

    @functimer
    def my_function():
        pass

    """
    logger = null_logger(__name__)

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()  # Start the timer
        result = func(*args, **kwargs)  # Execute the function
        end_time = time.perf_counter()  # End the timer
        elapsed_time = f"{end_time - start_time: .5f}"  # Calculate the elapsed time
        logger.debug("Function %s executed in %s seconds", func.__name__, elapsed_time)
        return result

    return wrapper
