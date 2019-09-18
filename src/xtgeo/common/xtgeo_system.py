# -*- coding: utf-8 -*-
"""Module for basic XTGeo interaction with OS/system files and folders"""

from __future__ import division, absolute_import
from __future__ import print_function

import os
import os.path

import xtgeo.cxtgeo.cxtgeo as _cxtgeo
from xtgeo.common import XTGeoDialog

xtg = XTGeoDialog()
logger = xtg.functionlogger(__name__)


def file_exists(fname):
    """Check if file exists, and returns True of OK."""
    status = os.path.isfile(fname)

    if not status:
        logger.warning("File does not exist")

    return status


def check_folder(xfile, raiseerror=None):
    """Check if folder given in xfile exists and is writeable.

    The file itself may not exist (yet), only the folder is checked

    Args:
        xfile (str): Name of full path, including a file name normally
        raiseerror (excpetion): If none, then return True or False, else raise the
            given Exception if False

    Return:
        status: True, if folder exists and is writable, False if not.

    """

    # Here are issues here on Windows in particular

    status = True

    if os.path.isdir(xfile):
        folder = xfile
    else:
        folder = os.path.dirname(xfile)
        if folder == "":
            folder = "."

    if not os.path.exists(folder):
        if raiseerror:
            raise raiseerror("Folder does not exist: <{}>".format(folder))

        status = False

    if os.path.exists(folder) and not os.access(folder, os.W_OK):
        if raiseerror:
            raise raiseerror(
                "Folder does exist but is not writable: <{}>".format(folder)
            )

        status = False

    return status


def get_fhandle(pfile, mode="rb"):
    """Return a new filehandle if pfile is not a filehandle already; otherwise
    return as is
    """

    if "Swig Object of type 'FILE" in str(pfile):
        return pfile

    return _cxtgeo.xtg_fopen(pfile, mode)


def is_fhandle(pfile):
    """Return True if pfile is a filehandle, not a file"""

    if "Swig Object of type 'FILE" in str(pfile):
        return True

    return False


def close_fhandle(fh, cond=True):
    """Close file given that fh is is a filehandle (return True), otherwise do
    nothing (return False).

    If cond is False, othing is done (made in order to avoid ifs i callers)
    """
    if not cond:
        return True

    if is_fhandle(fh):
        _cxtgeo.xtg_fclose(fh)
        logger.debug("File is now closed")
        return True

    return False
