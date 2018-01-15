"""Module for file utiltities, e.g. check if file exists."""
import os.path
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def file_exists(fname):
    """Check if file exists, and returns True of OK."""
    status = os.path.isfile(fname)

    if not status:
        logger.warning('File does not exist')

    return status
