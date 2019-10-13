# -*- coding: utf-8 -*-
"""Module for basic XTGeo interaction with OS/system files and folders"""

from __future__ import division, absolute_import
from __future__ import print_function

import os
import os.path
import io
import platform as plf

from .xtgeo_dialog import XTGeoDialog
import xtgeo.cxtgeo.cxtgeo as _cxtgeo

xtg = XTGeoDialog()
logger = xtg.functionlogger(__name__)


class _XTGeoCFile(object):
    """A private class for file handling of files in/out of CXTGeo"""

    def __init__(self, fobj, mode="rb"):

        self._name = fobj
        self._delete_after = False  # delete file (e.g. tmp) afterwards
        self._fhandle = None
        self._mode = mode

    @property
    def fhandle(self):  # was get_handle
        """File handle for CXTgeo (read only)"""

        if self._fhandle and "Swig Object of type 'FILE" in str(self._fhandle):
            return self._fhandle

        fhandle = None
        if (
            isinstance(self._name, io.BytesIO)
            and self._mode == "rb"
            and plf.system() == "Linux"
        ):
            buf = self._name.getvalue()  # bytes type in Python3, str in Python2

            # note that the typemap in swig computes the length for the buf!
            fhandle = _cxtgeo.xtg_fopen_bytestream(buf, self._mode)

        # elif isinstance(pfile, io.BytesIO) and mode == "rb" and plf.system() == "Windows":
        #     # windows miss fmemopen; hence write buffer to a tmp instead as workaround
        #     for

        #     # note that the typemap in swig computes the length for the buf!
        #     return _cxtgeo.xtg_fopen_bytestream(buf, mode)

        else:
            fhandle = _cxtgeo.xtg_fopen(self._name, self._mode)

        self._fhandle = fhandle
        return self._fhandle

    def exists(self):  # was: file_exists
        """Check if file or memerory stream exists, and returns True of OK."""
        if ("r" in self._mode):
            if isinstance(self._name, str) and os.path.isfile(self._name):
                return True

            elif isinstance(self._name, io.BytesIO):
                return True

            return False

        return True

    def check_folder(self, raiseerror=None):
        """Check if folder given in xfile exists and is writeable.

        The file itself may not exist (yet), only the folder is checked

        Args:
            raiseerror (excpetion): If none, then return True or False, else raise the
                given Exception if False

        Return:
            status: True, if folder exists and is writable, False if not.

        """

        # Here are issues here on Windows in particular

        status = True

        if os.path.isdir(self._name):
            folder = self.filename
        else:
            folder = os.path.dirname(self._name)
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

    def has_fhandle(self):  # was is_fhandle
        """Return True if pfile is a filehandle, not a file"""

        if self._fhandle and "Swig Object of type 'FILE" in str(self._fhandle):
            return True

        return False

    def close(self, cond=True, force=False):  # was close_fhandle
        """Close file handle given that filehandle exists (return True), otherwise do
        nothing (return False).

        If cond is False, othing is done (made in order to avoid ifs i callers)
        """
        if not cond:
            return True

        if self._fhandle:
            _cxtgeo.xtg_fclose(self._fhandle)
            logger.debug("File is now closed")

            return True

        return False
