from __future__ import annotations

import logging
import time
from functools import wraps
from typing import Callable, Literal, Optional


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


def functimer(
    func: Optional[Callable] = None,
    *,
    output: Literal["debug", "info", "print"] = "debug",
    comment: str = "",
) -> Callable:
    """A decorator function to measure the execution time of a function.

    Will emit a print or log message with the execution time in seconds, and is
    primarily for developer use.

    Usage is simple, just add the decorator to the function you want to measure:

    @functimer(output="print", comment="My function took a while to run")
    def my_function():
        pass

    Or without arguments:

    @functimer
    def my_function():
        pass

    """
    if func is None:

        def decorator(func: Callable) -> Callable:
            return functimer(func, output=output, comment=comment)

        return decorator

    logger = null_logger(__name__)

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()  # Start the timer
        result = func(*args, **kwargs)  # Execute the function
        end_time = time.perf_counter()  # End the timer
        elapsed_time = f"{end_time - start_time: .5f}"

        # Build message with optional comment
        message = f"Function {func.__name__} executed in {elapsed_time} seconds"
        if comment:
            message += f" - {comment}"

        # Output according to specified method
        if output == "print":
            print(message)
        elif output == "info":
            logger.info(message)
        else:
            logger.debug(message)

        return result

    return wrapper
