# -*- coding: utf-8 -*-
"""Module for basic XTGeo interaction with OS/system files and folders"""

from __future__ import division, absolute_import
from __future__ import print_function

import os


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


# def file_exists(fname):
#     """Check if file exists, and returns True of OK."""
#     status = os.path.isfile(fname)

#     if not status:
#         logger.warning('File does not exist')

#     return status


# def _get_fhandle(pfile):
#     """Examine for file or filehandle and return filehandle + a bool"""

#     pclose = True
#     if "Swig Object of type 'FILE" in str(pfile):
#         fhandle = pfile
#         pclose = False
#     else:
#         fhandle = _cxtgeo.xtg_fopen(pfile, 'rb')

#     return fhandle, pclose


# def _close_fhandle(fh, flag):
#     """Close file if flag is True"""

#     if flag:
#         _cxtgeo.xtg_fclose(fh)
#         logger.debug('File is now closed')
#     else:
#         logger.debug('File remains open')
