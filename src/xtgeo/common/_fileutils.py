"""Module for file utiltities, e.g. check if file exists."""
import os.path
import logging

import xtgeo.cxtgeo.cxtgeo as _cxtgeo

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def file_exists(fname):
    """Check if file exists, and returns True of OK."""
    status = os.path.isfile(fname)

    if not status:
        logger.warning('File does not exist')

    return status


def _get_fhandle(pfile):
    """Examine for file or filehandle and return filehandle + a bool"""

    pclose = True
    if "Swig Object of type 'FILE" in str(pfile):
        fhandle = pfile
        pclose = False
    else:
        fhandle = _cxtgeo.xtg_fopen(pfile, 'rb')

    return fhandle, pclose


def _close_fhandle(fh, flag):
    """Close file if flag is True"""

    if flag:
        _cxtgeo.xtg_fclose(fh)
        logger.debug('File is now closed')
    else:
        logger.debug('File remains open')
