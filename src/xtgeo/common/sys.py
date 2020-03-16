# -*- coding: utf-8 -*-
"""Module for basic XTGeo interaction with OS/system files and folders"""

from __future__ import division, absolute_import
from __future__ import print_function

try:
    import pathlib
except ImportError:
    import pathlib2 as pathlib

import os
from os.path import join
import io
from platform import system as plfsys
from tempfile import mkstemp


import six

import xtgeo.cxtgeo._cxtgeo as _cxtgeo
from .xtgeo_dialog import XTGeoDialog

xtg = XTGeoDialog()
logger = xtg.functionlogger(__file__)


def check_folder(fname, raiseerror=None):
    """General function to check folder"""
    _nn = _XTGeoCFile(fname)
    status = _nn.check_folder(raiseerror=raiseerror)
    del _nn
    return status


class _XTGeoCFile(object):
    """A private class for file handling of files in/out of CXTGeo"""

    def __init__(self, fobj, mode="rb"):

        self._file = None  # Path instance
        self._tmpfile = None
        self._delete_after = False  # delete file (e.g. tmp) afterwards
        self._fhandle = None
        self._mode = mode
        self._memstream = False
        logger.debug("Init ran for _XTGeoFile")

        # The self._file must be a Pathlib or a BytesIO instance
        if isinstance(fobj, pathlib.Path):
            self._file = fobj
        elif isinstance(fobj, str):
            self._file = pathlib.Path(fobj)
        elif isinstance(fobj, io.BytesIO):
            self._file = fobj
            self._memstream = True

        logger.info("Ran init of %s", __name__)

    @property
    def memstream(self):
        """Read only: Get True if file object is a memory stream (BytesIO)"""
        return self._memstream

    @property
    def file(self):
        """Read only: Get Path object (if input was file) or BytesIO object"""
        return self._file

    @property
    def name(self):
        """The absolute path name of a file"""

        logger.info("Get absolute name of file...")

        try:
            logger.debug("Try resolve...")
            fname = str(self._file.resolve())
        except FileNotFoundError:
            try:
                logger.debug("Try resolve parent, then file...")
                fname = os.path.abspath(
                    join(str(self._file.parent.resolve()), str(self._file.name))
                )
            except FileNotFoundError:
                # means that also folder is invalid
                logger.debug("Last attempt of name resolving...")
                fname = os.path.abspath(str(self._file))
        return fname

    @property
    def fhandle(self):  # was get_handle
        """SWIG file handle for CXTgeo, the filehandle is a read only property"""

        logger.info("Get SWIG fhandle...")

        if self._fhandle and "Swig Object of type 'FILE" in str(self._fhandle):
            return self._fhandle

        fhandle = None
        if (
            isinstance(self._file, io.BytesIO)
            and self._mode == "rb"
            and plfsys() == "Linux"
        ):
            if six.PY2:
                raise NotImplementedError(
                    "Reading BytesIO not fully supported in Python 2"
                )

            fobj = self._file.getvalue()  # bytes type in Python3, str in Python2

            # note that the typemap in swig computes the length for the buf/fobj!
            self._memstream = True

        elif (
            isinstance(self._file, io.BytesIO)
            and self._mode == "wb"
            and plfsys() == "Linux"
        ):
            if six.PY2:
                raise NotImplementedError(
                    "Writing to BytesIO not supported in Python 2"
                )

            fobj = bytes()
            self._memstream = True

        elif (
            isinstance(self._file, io.BytesIO)
            and self._mode == "rb"
            and (plfsys() == "Windows" or plfsys() == "Darwin")
        ):
            # windows/mac miss fmemopen; write buffer to a tmp instead as workaround
            fds, fobj = mkstemp(prefix="tmpxtgeoio")
            os.close(fds)
            with open(fobj, "wb") as newfile:
                newfile.write(self._file.getvalue())

            self._tmpfile = fobj

        else:
            fobj = self.name

        if self._memstream:
            fhandle = _cxtgeo.xtg_fopen_bytestream(fobj, self._mode)

        else:
            try:
                fhandle = _cxtgeo.xtg_fopen(fobj, self._mode)
            except TypeError as err:
                reason = ""
                if six.PY2:
                    reason = "In Python 2, do not use __future__ UnicodeLiterals!"
                logger.critical("Cannot open file: %s. %s", err, reason)

        self._fhandle = fhandle
        return self._fhandle

    def exists(self):  # was: file_exists
        """Check if 'r' file, memory stream or folder exists, and returns True of OK."""
        if "r" in self._mode:
            if self._file.exists():
                return True

            if isinstance(self._file, io.BytesIO):
                return True

            return False

        return True

    def check_file(self, raiseerror=None, raisetext=None):
        """Check if a file exists, and raises an IOError if not. Only
        meaningful for 'r' files

        Args:
            raiseerror (Exception): Type of Exception, default is None, which means
                no Excpetion, just return False or True
            raisetext (str): Which message to display if raiseerror, None gives a
                default message.

        Return:
            status: True, if file exists and is readable, False if not.
        """
        logger.info("Checking file...")

        if raisetext is None:
            raisetext = "File {} does not exist or cannot be accessed".format(self.name)

        if "r" in self._mode:
            if not self._file.is_file() or not self.exists():
                if raiseerror is not None:
                    raise raiseerror(raisetext)

                return False

        return True

    def check_folder(self, raiseerror=None, raisetext=None):
        """Check if folder given in xfile exists and is writeable.

        The file itself may not exist (yet), only the folder is checked

        Args:
            raiseerror (exception): If none, then return True or False, else raise the
                given Exception if False

        Return:
            status: True, if folder exists and is writable, False if not.

        """
        logger.info("Checking folder...")

        status = True
        folder = self._file.parent
        if raisetext is None:
            raisetext = "Folder {} does not exist or cannot be accessed".format(
                folder.name
            )

        if not folder.exists():
            if raiseerror:
                raise raiseerror(raisetext)

            status = False

        return status

        # # Here are issues here on Windows/Mac in particular

        # status = True

        # if os.path.isdir(self._file):
        #     folder = self._file
        # else:
        #     folder = os.path.dirname(self._file)
        #     if folder == "":
        #         folder = "."

        # if not os.path.exists(folder):
        #     if raiseerror:
        #         raise raiseerror("Folder does not exist: <{}>".format(folder))

        #     status = False

        # if os.path.exists(folder) and not os.access(folder, os.W_OK):
        #     if raiseerror:
        #         raise raiseerror(
        #             "Folder does exist but is not writable: <{}>".format(folder)
        #         )

        #     status = False

        # return status

    def has_fhandle(self):  # was is_fhandle
        """Return True if pfile is a filehandle, not a file"""

        if self._fhandle and "Swig Object of type 'FILE" in str(self._fhandle):
            return True

        return False

    def splitext(self, lower=False):
        """Return file stem and suffix, always lowercase if lower is True"""

        logger.info("Run splitext to get stem and suffix...")

        stem = self._file.stem
        suffix = self._file.suffix
        suffix = suffix.replace(".", "")

        if lower:
            stem = stem.lower()
            suffix = suffix.lower()

        return stem, suffix

    def close(self, cond=True):
        """Close file handle given that filehandle exists (return True), otherwise do
        nothing (return False).

        If cond is False, nothing is done (made in order to avoid ifs in callers)
        """

        logger.info("Close SWIG fhandle...")

        if not cond:
            return True

        if self._memstream and self._fhandle and "w" in self._mode:
            # this assumes that the file pointer is in the end of the current filehandle
            npos = _cxtgeo.xtg_ftell(self._fhandle)
            buf = bytes(npos)
            ier = _cxtgeo.xtg_get_fbuffer(self._fhandle, buf)
            if ier == 0:
                self._file.write(buf)  # write to bytesIO instance
                _cxtgeo.xtg_fflush(self._fhandle)
            else:
                raise RuntimeError("Could not write stream for unknown reasons")

        if self._fhandle:
            ier = _cxtgeo.xtg_fclose(self._fhandle)
            if ier != 0:
                raise RuntimeError("Could not close C file, code {}".format(ier))

            logger.debug("File is now closed %s", self._fhandle)

            if self._tmpfile:
                try:
                    os.remove(self._tmpfile)
                except Exception as ex:  # pylint: disable=W0703
                    logger.error("Could not remove tempfile for some reason: %s", ex)

            return True

        return False
