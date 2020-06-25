# -*- coding: utf-8 -*-
"""Module for basic XTGeo interaction with OS/system files and folders"""

from __future__ import division, absolute_import
from __future__ import print_function


import os
from os.path import join
import io
from platform import system as plfsys
from tempfile import mkstemp


import six

import xtgeo.cxtgeo._cxtgeo as _cxtgeo
from xtgeo import pathlib

from .xtgeo_dialog import XTGeoDialog

xtg = XTGeoDialog()
logger = xtg.functionlogger(__file__)


def check_folder(fname, raiseerror=None):
    """General function to check folder"""
    _nn = _XTGeoFile(fname)
    status = _nn.check_folder(raiseerror=raiseerror)
    del _nn
    return status


class _XTGeoFile(object):
    """
    A private class for file handling of files in/out of XTGeo and possibly CXTGeo

    Interesting attributes:

    xfile = _XTGeoFile(..some Path or  str or BytesIO ...)

    xfile.name: The absolute path to the file (str)
    xfile.file: The pathlib.Path instance
    xfile.memstream: Is True if memory stream

    xfile.exist(): Returns True (provided mode 'r') if file exists, always True for 'w'
    xfile.check_file(...): As above but may raise an Excpetion
    xfile.check_folder(...): For folder; may raise an Excpetion
    xfile.splitext(): return file's stem and extension
    xfile.get_cfhandle(): Get C SWIG file handle
    xfile.cfclose(): Close current C SWIG filehandle


    """

    def __init__(self, fobj, mode="rb"):

        self._file = None  # Path instance
        self._tmpfile = None
        self._delete_after = False  # delete file (e.g. tmp) afterwards
        self._mode = mode
        self._memstream = False

        self._cfhandle = 0
        self._cfhandlecount = 0

        # for internal usage in tests; mimick window/mac with no fmemopen in C
        self._fake_nofmem = False

        logger.debug("Init ran for _XTGeoFile")

        # The self._file must be a Pathlib or a BytesIO instance
        if isinstance(fobj, pathlib.Path):
            self._file = fobj
        elif isinstance(fobj, str):
            self._file = pathlib.Path(fobj)
        elif isinstance(fobj, io.BytesIO):
            self._file = fobj
            self._memstream = True
        elif isinstance(fobj, _XTGeoFile):
            raise RuntimeError("Reinstancing object, not allowed", self.__class__)
        else:
            raise RuntimeError("Illegal input, cannot continue", self.__class__)

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

        if self._memstream:
            return self._file

        try:
            logger.debug("Try resolve...")
            fname = str(self._file.resolve())
        except OSError:
            try:
                logger.debug("Try resolve parent, then file...")
                fname = os.path.abspath(
                    join(str(self._file.parent.resolve()), str(self._file.name))
                )
            except OSError:
                # means that also folder is invalid
                logger.debug("Last attempt of name resolving...")
                fname = os.path.abspath(str(self._file))
        return fname

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
        """Check if a file exists, and raises an OSError if not. Only
        meaningful for 'r' files

        Args:
            raiseerror (Exception): Type of Exception, default is None, which means
                no Exception, just return False or True
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

    def get_cfhandle(self):  # was get_handle
        """
        Get SWIG C file handle for CXTgeo

        This is tied to cfclose() which closes the file.

        if _cfhandle already exists, then _cfhandlecount is increased with 1

        """

        logger.info("Get SWIG C fhandle...")

        # differ on Linux and other OS as Linux can use fmemopen() in C
        islinux = True
        if plfsys() != "Linux":
            islinux = False

        if self._cfhandle and "Swig Object of type 'FILE" in str(self._cfhandle):
            self._cfhandlecount += 1
            return self._cfhandle

        if isinstance(self._file, io.BytesIO) and self._mode == "rb" and islinux:
            if six.PY2:
                raise NotImplementedError(
                    "Reading BytesIO not fully supported in Python 2"
                )

            fobj = self._file.getvalue()  # bytes type in Python3, str in Python2

            # note that the typemap in swig computes the length for the buf/fobj!
            self._memstream = True

        elif isinstance(self._file, io.BytesIO) and self._mode == "wb" and islinux:
            if six.PY2:
                raise NotImplementedError(
                    "Writing to BytesIO not supported in Python 2"
                )

            fobj = bytes()
            self._memstream = True

        elif (
            isinstance(self._file, io.BytesIO)
            and self._mode == "rb"
            and not islinux  # Windows or Darwin
        ):
            # windows/mac miss fmemopen; write buffer to a tmp instead as workaround
            fds, self._tmpfile = mkstemp(prefix="tmpxtgeoio")
            os.close(fds)
            with open(self._tmpfile, "wb") as newfile:
                newfile.write(self._file.getvalue())

        else:
            fobj = self.name

        if self._memstream:
            if islinux:
                cfhandle = _cxtgeo.xtg_fopen_bytestream(fobj, self._mode)
            else:
                cfhandle = _cxtgeo.xtg_fopen(self._tmpfile, self._mode)

        else:
            try:
                cfhandle = _cxtgeo.xtg_fopen(fobj, self._mode)
            except TypeError as err:
                reason = ""
                if six.PY2:
                    reason = "In Python 2, do not use __future__ UnicodeLiterals!"
                logger.critical("Cannot open file: %s. %s", err, reason)

        self._cfhandle = cfhandle
        self._cfhandlecount = 1

        return self._cfhandle

    def cfclose(self, strict=True):
        """
        Close SWIG C file handle by keeping track of _cfhandlecount

        Return True if cfhandle is really closed.
        """

        logger.info("Request for closing SWIG fhandle...")

        if self._cfhandle is None or self._cfhandlecount == 0:
            if strict:
                raise RuntimeError("Ask to close a nonexisting C file handle")

            self._cfhandle = None
            self._cfhandlecount = 0
            return True

        if self._cfhandlecount > 1 or self._cfhandlecount == 0:
            self._cfhandlecount -= 1
            return False

        if self._memstream and self._cfhandle and "w" in self._mode:
            # this assures that the file pointer is in the end of the current filehandle
            npos = _cxtgeo.xtg_ftell(self._cfhandle)
            buf = bytes(npos)
            ier = _cxtgeo.xtg_get_fbuffer(self._cfhandle, buf)
            if ier == 0:
                self._file.write(buf)  # write to bytesIO instance
                _cxtgeo.xtg_fflush(self._cfhandle)
            else:
                raise RuntimeError("Could not write stream for unknown reasons")

        ier = _cxtgeo.xtg_fclose(self._cfhandle)
        if ier != 0:
            raise RuntimeError("Could not close C file, code {}".format(ier))

        logger.info("File is now closed for C io: %s", self.name)

        if self._tmpfile:
            try:
                os.remove(self._tmpfile)
            except Exception as ex:  # pylint: disable=W0703
                logger.error("Could not remove tempfile for some reason: %s", ex)

        self._cfhandle = None
        self._cfhandlecount = 0

        return True
