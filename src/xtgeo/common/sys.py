# -*- coding: utf-8 -*-
"""Module for basic XTGeo interaction with OS/system files and folders."""

import hashlib
import io
import os
import pathlib
import re
import struct
import uuid
from os.path import join
from platform import system as plfsys
from tempfile import mkstemp
from types import BuiltinFunctionType
from typing import Optional

import h5py
import numpy as np
import xtgeo.cxtgeo._cxtgeo as _cxtgeo

from .xtgeo_dialog import XTGeoDialog

xtg = XTGeoDialog()
logger = xtg.functionlogger(__name__)


SUPPORTED_FORMATS = {
    "rmswell": ["rmswell", "rmsw", "w", "bw"],
    "roff_binary": ["roff_binary", "roff", "roff_bin", "roff-bin", "roffbin", "roff.*"],
    "roff_ascii": ["roff_ascii", "roff_asc", "roff-asc", "roffasc", "asc"],
    "egrid": ["egrid"],
    "fegrid": ["fegrid"],
    "init": ["init"],
    "finit": ["finit"],
    "unrst": ["unrst"],
    "funrst": ["funrst"],
    "grdecl": ["grdecl"],
    "bgrdecl": ["bgrdecl"],
    "irap_binary": ["irap_binary", "irap_bin", "rms_binary", "irapbin", "gri"],
    "irap_ascii": ["irap_ascii", "irap_asc", "rms_ascii", "irapasc", "fgr"],
    "hdf": ["hdf", "hdf5", "h5"],
    "segy": ["segy", "sgy", "segy.*"],
    "storm": ["storm"],
    "zmap_ascii": ["zmap", "zmap+", "zmap_ascii", "zmap-ascii", "zmap-asc", "zmap.*"],
    "ijxyz": ["ijxyz"],
    "petromod": ["pmd", "petromod"],
    "xtg": ["xtg", "xtgeo", "xtgf", "xtg.*"],
    "xyz": ["xyz", "poi", "pol"],
    "rms_attr": ["rms_attr", "rms_attrs", "rmsattr.*"],
}

VALID_FILE_ALIASES = ["$fmu-v1", "$md5sum", "$random"]


def npfromfile(fname, dtype=np.float32, count=1, offset=0, mmap=False):
    """Wrapper round np.fromfile to be compatible with older np versions."""
    try:
        if mmap:
            vals = np.memmap(
                fname, dtype=dtype, shape=(count,), mode="r", offset=offset
            )
        else:
            vals = np.fromfile(fname, dtype=dtype, count=count, offset=offset)
    except TypeError as err:
        # offset keyword requires numpy >= 1.17, need this for backward compat.:
        if "'offset' is an invalid" in str(err):
            with open(fname, "rb") as buffer:
                buffer.seek(offset)
                vals = np.fromfile(buffer, dtype=dtype, count=count)
        else:
            raise
    return vals


def check_folder(fname, raiseerror=None):
    """General function to check folder."""
    _nn = _XTGeoFile(fname)
    status = _nn.check_folder(raiseerror=raiseerror)
    del _nn
    return status


def generic_hash(gid, hashmethod="md5"):
    """Return a unique hash ID for current instance.

    This hash can e.g. be used to compare two instances for equality.

    Args:
        gid (str): Any string as signature, e.g. cumulative attributes of an instance.
        hashmethod (str or function): Supported methods are "md5", "sha256", "blake2b"
            or use a full function signature e.g. hashlib.sha128.

    Returns:
        Hash signature.

    Raises:
        KeyError: String in hashmethod has an invalid option

    .. versionadded:: 2.14
    """
    validmethods = {
        "md5": hashlib.md5,
        "sha256": hashlib.sha256,
        "blake2b": hashlib.blake2b,
    }

    mhash = None
    if isinstance(hashmethod, str):
        mhash = validmethods[hashmethod]()
    elif isinstance(hashmethod, BuiltinFunctionType):
        mhash = hashmethod()

    mhash.update(gid.encode())
    return mhash.hexdigest()


class _XTGeoFile(object):
    """A private class for file/stream handling in/out of XTGeo and CXTGeo.

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

    def __init__(self, fobj, mode="rb", obj=None):

        self._file = None  # Path instance or BytesIO memory stream
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
        elif isinstance(fobj, io.StringIO):
            self._file = fobj
            self._memstream = True
        elif isinstance(fobj, _XTGeoFile):
            raise RuntimeError("Reinstancing object, not allowed", self.__class__)
        else:
            raise RuntimeError(
                "Illegal input, cannot continue ({}) {}: {}".format(
                    self.__class__, fobj, type(fobj)
                )
            )

        if obj and not self._memstream:
            self.resolve_alias(obj)

        logger.info("Ran init of %s, ID is %s", __name__, id(self))

    @property
    def memstream(self):
        """Read only: Get True if file object is a memory stream (BytesIO)."""
        return self._memstream

    @property
    def file(self):
        """Read only: Get Path object (if input was file) or BytesIO object."""
        return self._file

    @property
    def name(self):
        """The absolute path name of a file."""
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

    def resolve_alias(self, obj):
        """Change a file path name alias to autogenerated name, based on rules.

        Only the file stem name will be updated, not the file name extension. Any
        parent folders and file suffix/extension will be returned as is.

        Aliases supported so far are '$md5sum' '$random' '$fmu-v1'

        Args:
            obj (XTGeo instance): Instance of e.g. RegularSurface()

        Example::
            >>> import xtgeo
            >>> surf = xtgeo.surface_from_file(surface_dir + "/topreek_rota.gri")
            >>> xx = _XTGeoFile("/tmp/$md5sum.gri", "rb", surf)
            >>> print(xx.file)
            /tmp/c144fe19742adac8187b97e7976ac68c.gri

        .. versionadded:: 2.14
        """
        fileobj = self._file

        parent = fileobj.parent
        stem = fileobj.stem
        suffix = fileobj.suffix

        if "$" in stem and stem not in VALID_FILE_ALIASES:
            raise ValueError(
                "A '$' is present in file name but this is not a valid alias"
            )

        newname = stem
        if stem == "$md5sum":
            newname = obj.generate_hash()
        elif stem == "$random":
            newname = uuid.uuid4().hex  # random name
        elif stem == "$fmu-v1":
            # will make name such as topvalysar--avg_porosity based on metadata
            short = obj.metadata.opt.shortname.lower().replace(" ", "_")
            desc = obj.metadata.opt.description.lower().replace(" ", "_")
            date = obj.metadata.opt.datetime
            newname = short + "--" + desc
            if date:
                newname += "--" + date
        else:
            # return without modifications of self._file to avoid with_suffix() issues
            # if the file name stem itself contains multiple '.'
            return

        self._file = (parent / newname).with_suffix(suffix)

    def exists(self):  # was: file_exists
        """Check if 'r' file, memory stream or folder exists, and returns True of OK."""
        if "r" in self._mode:
            if isinstance(self._file, io.BytesIO):
                return True

            if self._file.exists():
                return True

            return False

        return True

    def check_file(self, raiseerror=None, raisetext=None):
        """Check if a file exists, and raises an OSError if not.

        This is only meaningful for 'r' files.

        Args:
            raiseerror (Exception): Type of Exception, default is None, which means
                no Exception, just return False or True
            raisetext (str): Which message to display if raiseerror, None gives a
                default message.

        Return:
            status: True, if file exists and is readable, False if not.
        """
        logger.info("Checking file...")

        if self.memstream:
            return True

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
            raisetext (str): Text to raise.

        Return:
            status: True, if folder exists and is writable, False if not.

        Raises:
            ValueError: If the file is a memstream

        """
        if self.memstream:
            raise ValueError("Tried to check folder status of a in-memory file")

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
        """Return file stem and suffix, always lowercase if lower is True."""
        logger.info("Run splitext to get stem and suffix...")

        stem = self._file.stem
        suffix = self._file.suffix
        suffix = suffix.replace(".", "")

        if lower:
            stem = stem.lower()
            suffix = suffix.lower()

        return stem, suffix

    def get_cfhandle(self):  # was get_handle
        """Get SWIG C file handle for CXTgeo.

        This is tied to cfclose() which closes the file.

        if _cfhandle already exists, then _cfhandlecount is increased with 1

        """
        # differ on Linux and other OS as Linux can use fmemopen() in C
        islinux = True
        fobj = None
        if plfsys() != "Linux":
            islinux = False

        if self._cfhandle and "Swig Object of type 'FILE" in str(self._cfhandle):
            self._cfhandlecount += 1
            logger.info("Get SWIG C fhandle no %s", self._cfhandlecount)
            return self._cfhandle

        if isinstance(self._file, io.BytesIO) and self._mode == "rb" and islinux:

            fobj = self._file.getvalue()  # bytes type in Python3, str in Python2

            # note that the typemap in swig computes the length for the buf/fobj!
            self._memstream = True

        elif isinstance(self._file, io.BytesIO) and self._mode == "wb" and islinux:
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
                raise IOError(f"Cannot open file: {fobj}") from err

        self._cfhandle = cfhandle
        self._cfhandlecount = 1

        logger.info("Get initial SWIG C fhandle no %s", self._cfhandlecount)
        return self._cfhandle

    def cfclose(self, strict=True):
        """Close SWIG C file handle by keeping track of _cfhandlecount.

        Return True if cfhandle is really closed.
        """
        logger.info("Request for closing SWIG fhandle no: %s", self._cfhandlecount)

        if self._cfhandle is None or self._cfhandlecount == 0:
            if strict:
                raise RuntimeError("Ask to close a nonexisting C file handle")

            self._cfhandle = None
            self._cfhandlecount = 0
            return True

        if self._cfhandlecount > 1 or self._cfhandlecount == 0:
            self._cfhandlecount -= 1
            logger.info(
                "Remaining SWIG cfhandles: %s, do not close...", self._cfhandlecount
            )
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
        logger.info("Remaining SWIG cfhandles: %s, return is True", self._cfhandlecount)

        return True

    def detect_fformat(
        self, details: Optional[bool] = False, suffixonly: Optional[bool] = False
    ):
        """Try to deduce format from looking at file signature.

        The file signature may be the initial part of the binary file/stream but if
        that fails, the file extension is used.

        Args:
            details: If True, more info is added to the return string (useful for some
                formats) e.g. "hdf RegularSurface xtgeo"
            suffixonly: If True, look at file suffix only.

        Returns:
            A string with format spesification, e.g. "hdf".
        """

        if not suffixonly:
            fformat = self._detect_fformat_by_contents(details)
            if fformat is not None:
                return fformat

        # if looking at contents failed, look at extension
        fmt = self._detect_format_by_extension()
        return self._validate_format(fmt)

    def _detect_fformat_by_contents(self, details: Optional[bool] = False):
        # Try the read the N first bytes
        maxbuf = 100

        if self.memstream:
            self.file.seek(0)
            buf = self.file.read(maxbuf)
            self.file.seek(0)
        else:
            if not self.exists():
                raise ValueError(f"File {self.name} does not exist")
            with open(self.file, "rb") as fhandle:
                buf = fhandle.read(maxbuf)

        if not isinstance(buf, bytes):
            return None

        # HDF format, different variants
        if len(buf) >= 4:
            _, hdf = struct.unpack("b 3s", buf[:4])
            if hdf == b"HDF":
                logger.info("Signature is hdf")

                main = self._validate_format("hdf")
                fmt = ""
                provider = ""
                if details:
                    with h5py.File(self.file, "r") as hstream:
                        for xtgtype in ["RegularSurface", "Well", "CornerPointGrid"]:
                            if xtgtype in hstream.keys():
                                fmt = xtgtype
                                grp = hstream.require_group(xtgtype)
                                try:
                                    provider = grp.attrs["provider"]
                                except KeyError:
                                    provider = "unknown"
                                break

                    return f"{main} {fmt} {provider}"
                else:
                    return main

        # Irap binary regular surface format
        if len(buf) >= 8:
            fortranblock, gricode = struct.unpack(">ii", buf[:8])
            if fortranblock == 32 and gricode == -996:
                logger.info("Signature is irap binary")
                return self._validate_format("irap_binary")

        # Petromod binary regular surface
        if b"Content=Map" in buf and b"DataUnitDistance" in buf:
            logger.info("Signature is petromod")
            return self._validate_format("petromod")

        # Eclipse binary 3D EGRID, look at FILEHEAD:
        #  'FILEHEAD'         100 'INTE'
        #   3        2016           0           0           0           0
        #  (ver)    (release)      (reserved)   (backw)    (gtype)      (dualporo)

        if len(buf) >= 24:
            fort1, name, num, _, fort2 = struct.unpack("> i 8s i 4s i", buf[:24])
            if fort1 == 16 and name == b"FILEHEAD" and num == 100 and fort2 == 16:
                # Eclipse corner point EGRID
                logger.info("Signature is egrid")
                return self._validate_format("egrid")
            # Eclipse binary 3D UNRST, look for SEQNUM:
            #  'SEQNUM'         1 'INTE'
            if fort1 == 16 and name == b"SEQNUM  " and num == 1 and fort2 == 16:
                # Eclipse UNRST
                logger.info("Signature is unrst")
                return self._validate_format("unrst")
            # Eclipse binary 3D INIT, look for INTEHEAD:
            #  'INTEHEAD'         411 'INTE'
            if fort1 == 16 and name == b"INTEHEAD" and num > 400 and fort2 == 16:
                # Eclipse INIT
                logger.info("Signature is init")

                return self._validate_format("init")

        if len(buf) >= 9:
            name, _ = struct.unpack("8s b", buf[:9])
            # ROFF binary 3D
            if name == b"roff-bin":
                logger.info("Signature is roff_binary")
                return self._validate_format("roff_binary")
            # ROFF ascii 3D
            if name == b"roff-asc":
                logger.info("Signature is roff_ascii")
                return self._validate_format("roff_ascii")

        # RMS well format (ascii)
        # 1.0
        # Unknown
        # WELL12 90941.63200000004 5506367.711 23.0
        # ...
        # The signature here is one float in first line with values 1.0; one string
        # in second line; and 3 or 4 items in the next (sometimes RKB is missing)
        try:
            xbuf = buf.decode().split("\n")
        except UnicodeDecodeError:
            return None

        if (
            len(xbuf) >= 3
            and xbuf[0] == "1.0"
            and len(xbuf[1]) >= 1
            and len(xbuf[2]) >= 10
        ):
            logger.info("Signature is rmswell")
            return self._validate_format("rmswell")

        return None

    def _detect_format_by_extension(self):
        """Detect format by extension."""
        if self.memstream:
            return "unknown"

        suffix = self.file.suffix[1:].lower()

        for fmt, variants in SUPPORTED_FORMATS.items():
            if suffix in variants:
                logger.info(f"Extension hints {fmt}")
                return fmt

        # if none of these above are accepted, check regular expression
        # (intentional to complete all variant in loop above first before trying re())
        for fmt, variants in SUPPORTED_FORMATS.items():
            for var in variants:
                if "*" in var:
                    pattern = re.compile(var)
                    if pattern.match(suffix):
                        logger.info(f"Extension by regexp hints {fmt}")
                        return fmt

        return "unknown"

    @staticmethod
    def _validate_format(fmt):
        """Validate format."""
        if fmt in SUPPORTED_FORMATS.keys() or fmt == "unknown":
            return fmt
        else:
            raise RuntimeError(f"Invalid format: {fmt}")

    @staticmethod
    def generic_format_by_proposal(propose):
        """Get generic format by proposal."""
        for fmt, variants in SUPPORTED_FORMATS.items():
            if propose in variants:
                return fmt

        # if none of these above are accepted, check regular expression
        for fmt, variants in SUPPORTED_FORMATS.items():
            for var in variants:
                if "*" in var:
                    pattern = re.compile(var)
                    if pattern.match(propose):
                        return fmt

        raise ValueError(f"Non-supportred file extension: {propose}")


def inherit_docstring(inherit_from):
    def decorator_set_docstring(func):
        if func.__doc__ is None and inherit_from.__doc__ is not None:
            func.__doc__ = inherit_from.__doc__
        return func

    return decorator_set_docstring
