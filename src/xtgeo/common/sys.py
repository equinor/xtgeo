"""Module for basic XTGeo interaction with OS/system files and folders."""

from __future__ import annotations

import hashlib
import io
import os
import pathlib
import platform
import re
import struct
import uuid
from collections.abc import Callable
from os.path import join
from tempfile import mkstemp
from types import BuiltinFunctionType
from typing import TYPE_CHECKING, Literal, Union

import h5py
import numpy as np

from xtgeo import _cxtgeo

from . import null_logger
from ._xyz_enum import _AttrType

if TYPE_CHECKING:
    import numpy.typing as npt
    import pandas as pd

    from xtgeo import (
        BlockedWell,
        BlockedWells,
        Cube,
        Grid,
        GridProperties,
        GridProperty,
        Points,
        Polygons,
        RegularSurface,
        Surfaces,
        Well,
        Wells,
    )

    XTGeoObject = Union[
        BlockedWell,
        BlockedWells,
        Cube,
        Grid,
        GridProperty,
        GridProperties,
        Points,
        Polygons,
        RegularSurface,
        Surfaces,
        Well,
        Wells,
    ]

logger = null_logger(__name__)


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


def npfromfile(
    fname: str | pathlib.Path | io.BytesIO | io.StringIO,
    dtype: npt.DTypeLike = np.float32,
    count: int = 1,
    offset: int = 0,
    mmap: bool = False,
) -> np.ndarray:
    """Wrapper round np.fromfile to be compatible with older np versions."""
    try:
        if mmap and not isinstance(fname, (io.BytesIO, io.StringIO)):
            vals = np.memmap(
                fname, dtype=dtype, shape=(count,), mode="r", offset=offset
            )
        else:
            vals = np.fromfile(fname, dtype=dtype, count=count, offset=offset)
    except TypeError as err:
        # offset keyword requires numpy >= 1.17, need this for backward compat.:
        if "'offset' is an invalid" not in str(err):
            raise
        if not isinstance(fname, (str, pathlib.Path)):
            raise
        with open(fname, "rb") as buffer:
            buffer.seek(offset)
            vals = np.fromfile(buffer, dtype=dtype, count=count)
    return vals


def check_folder(
    fname: str | pathlib.Path | io.BytesIO | io.StringIO,
    raiseerror: type[Exception] | None = None,
) -> bool:
    """General function to check folder."""
    _nn = _XTGeoFile(fname)
    status = _nn.check_folder(raiseerror=raiseerror)
    del _nn
    return status


def generic_hash(
    gid: str, hashmethod: Literal["md5", "sha256", "blake2d"] | Callable = "md5"
) -> str:
    """Return a unique hash ID for current instance.

    This hash can e.g. be used to compare two instances for equality.

    Args:
        gid: Any string as signature, e.g. cumulative attributes of an instance.
        hashmethod: Supported methods are "md5", "sha256", "blake2b"
            or use a full function signature e.g. hashlib.sha128.
            Defaults to md5.

    Returns:
        Hash signature.

    Raises:
        KeyError: String in hashmethod has an invalid option

    .. versionadded:: 2.14

    """
    validmethods: dict[str, Callable] = {
        "md5": hashlib.md5,
        "sha256": hashlib.sha256,
        "blake2b": hashlib.blake2b,
    }

    if isinstance(hashmethod, str) and hashmethod in validmethods:
        mhash = validmethods[hashmethod]()
    elif isinstance(hashmethod, BuiltinFunctionType):
        mhash = hashmethod()
    else:
        raise ValueError(f"Invalid hash method provided: {hashmethod}")

    mhash.update(gid.encode())
    return mhash.hexdigest()


class _XTGeoFile:
    """
    A private class for file/stream handling in/out of XTGeo and CXTGeo.

    Interesting attributes:

    xfile = _XTGeoFile(..some Path or str or BytesIO ...)

    xfile.name: The absolute path to the file (str)
    xfile.file: The pathlib.Path instance
    xfile.memstream: Is True if memory stream

    xfile.exists(): Returns True (provided mode 'r') if file exists, always True for 'w'
    xfile.check_file(...): As above but may raise an Excpetion
    xfile.check_folder(...): For folder; may raise an Excpetion
    xfile.splitext(): return file's stem and extension
    xfile.get_cfhandle(): Get C SWIG file handle
    xfile.cfclose(): Close current C SWIG filehandle

    """

    def __init__(
        self,
        filelike: str | pathlib.Path | io.BytesIO | io.StringIO,
        mode: Literal["rb", "wb"] = "rb",
        obj: XTGeoObject = None,
    ) -> None:
        logger.debug("Init ran for _XTGeoFile")

        if not isinstance(filelike, (str, pathlib.Path, io.BytesIO, io.StringIO)):
            raise RuntimeError(
                f"Cannot instantiate {self.__class__} from "
                f"{filelike} of type {type(filelike)}. Expected "
                f"a str, pathlib.Path, io.BytesIO, or io.StringIO."
            )

        self._tmpfile: str | None = None
        self._delete_after = False  # delete file (e.g. tmp) afterwards
        self._mode = mode

        self._cfhandle = 0
        self._cfhandlecount = 0

        # for internal usage in tests; mimick window/mac with no fmemopen in C
        self._fake_nofmem = False

        if isinstance(filelike, str):
            filelike = pathlib.Path(filelike)

        self._file: pathlib.Path | io.BytesIO | io.StringIO = filelike
        self._memstream = isinstance(self._file, (io.BytesIO, io.StringIO))

        if obj and not self._memstream:
            self.resolve_alias(obj)

        logger.debug("Ran init of %s, ID is %s", __name__, id(self))

    @property
    def memstream(self) -> bool:
        """Get whether or not this file is a io.BytesIO/StringIO memory stream."""
        return self._memstream

    @property
    def file(self) -> pathlib.Path | io.BytesIO | io.StringIO:
        """Get Path object (if input was file) or memory stream object."""
        return self._file

    @property
    def name(self) -> str | io.BytesIO | io.StringIO:
        """Get the absolute path name of the file, or the memory stream."""
        if isinstance(self.file, (io.BytesIO, io.StringIO)):
            return self.file

        try:
            logger.debug("Trying to resolve filepath")
            fname = str(self.file.resolve())
        except OSError:
            try:
                logger.debug("Trying to resolve parent, then file...")
                fname = os.path.abspath(
                    join(str(self.file.parent.resolve()), str(self.file.name))
                )
            except OSError:
                # means that also folder is invalid
                logger.debug("Last attempt of name resolving...")
                fname = os.path.abspath(str(self.file))
        return fname

    def resolve_alias(self, obj: XTGeoObject) -> None:
        """
        Change a file path name alias to autogenerated name, based on rules.

        Only the file stem name will be updated, not the file name extension. Any
        parent folders and file suffix/extension will be returned as is.

        Aliases supported so far are '$md5sum' '$random' '$fmu-v1'

        Args:
            obj: Instance of some XTGeo object e.g. RegularSurface()

        Example::
            >>> import xtgeo
            >>> surf = xtgeo.surface_from_file(surface_dir + "/topreek_rota.gri")
            >>> xx = _XTGeoFile("/tmp/$md5sum.gri", "rb", surf)
            >>> print(xx.file)
            /tmp/c144fe19742adac8187b97e7976ac68c.gri

        .. versionadded:: 2.14

        """
        if self.memstream or isinstance(self.file, (io.BytesIO, io.StringIO)):
            return

        parent = self.file.parent
        stem = self.file.stem
        suffix = self.file.suffix

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

    def exists(self) -> bool:
        """Returns True if 'r' file, memory stream, or folder exists."""
        if "r" in self._mode:
            if isinstance(self.file, (io.BytesIO, io.StringIO)):
                return True
            return self.file.exists()

        # Writes and appends will always exist after writing
        return True

    def check_file(
        self,
        raiseerror: type[Exception] | None = None,
        raisetext: str | None = None,
    ) -> bool:
        """
        Check if a file exists, and raises an OSError if not.

        This is only meaningful for 'r' files.

        Args:
            raiseerror: Type of exception to raise. Default is None, which means
                no Exception, just return False or True.
            raisetext: Which message to display if raiseerror. Defaults to None
                which gives a default message.

        Returns:
            True if file exists and is readable, False if not.

        """
        logger.debug("Checking file...")

        # Redundant but mypy can't follow when memstream is True
        if self.memstream or isinstance(self.file, (io.BytesIO, io.StringIO)):
            return True

        if raisetext is None:
            raisetext = f"File {self.name} does not exist or cannot be accessed"

        if "r" in self._mode:
            if not self.file.is_file() or not self.exists():
                if raiseerror is not None:
                    raise raiseerror(raisetext)
                return False

        return True

    def check_folder(
        self,
        raiseerror: type[Exception] | None = None,
        raisetext: str | None = None,
    ) -> bool:
        """
        Check if folder given in file exists and is writeable.

        The file itself may not exist (yet), only the folder is checked.

        Args:
            raiseerror: Type of exception to raise. Default is None, which means
                no Exception, just return False or True.
            raisetext: Which message to display if raiseerror. Defaults to None
                which gives a default message.

        Returns:
            True if folder exists and is writable, False if not.

        Raises:
            ValueError: If the file is a memstream

        """
        # Redundant but mypy can't follow when memstream is True
        if self.memstream or isinstance(self.file, (io.BytesIO, io.StringIO)):
            raise ValueError("Cannot check folder status of an in-memory file")

        logger.debug("Checking folder...")

        folder = self.file.parent
        if raisetext is None:
            raisetext = f"Folder {folder.name} does not exist or cannot be accessed"

        if not folder.exists():
            if raiseerror:
                raise raiseerror(raisetext)

            return False

        return True

    def splitext(self, lower: bool = False) -> tuple[str, str]:
        """Return file stem and suffix, always lowercase if lower is True."""
        if self.memstream or isinstance(self.file, (io.BytesIO, io.StringIO)):
            raise ValueError("Cannot split extension of an in-memory file")

        logger.debug("Run splitext to get stem and suffix...")

        stem = self.file.stem
        suffix = self.file.suffix
        suffix = suffix.replace(".", "")

        if lower:
            stem = stem.lower()
            suffix = suffix.lower()

        return stem, suffix

    def get_cfhandle(self) -> int:
        """
        Get SWIG C file handle for CXTGeo.

        This is tied to cfclose() which closes the file.

        if _cfhandle already exists, then _cfhandlecount is increased with 1

        Returns:
            int indicating the file handle number.

        """
        # Windows and pre-10.13 macOS lack fmemopen()
        islinux = platform.system() == "Linux"

        if self._cfhandle and "Swig Object of type 'FILE" in str(self._cfhandle):
            self._cfhandlecount += 1
            logger.debug("Get SWIG C fhandle no %s", self._cfhandlecount)
            return self._cfhandle

        fobj: bytes | str | io.BytesIO | io.StringIO = self.name
        if isinstance(self.file, io.BytesIO):
            if self._mode == "rb" and islinux:
                fobj = self.file.getvalue()
            elif self._mode == "wb" and islinux:
                fobj = b""  # Empty bytes obj.
            elif self._mode == "rb" and not islinux:
                # Write stream to a temporary file
                fds, self._tmpfile = mkstemp(prefix="tmpxtgeoio")
                os.close(fds)
                with open(self._tmpfile, "wb") as newfile:
                    newfile.write(self.file.getvalue())

        if self.memstream:
            if islinux:
                cfhandle = _cxtgeo.xtg_fopen_bytestream(fobj, self._mode)
            else:
                cfhandle = _cxtgeo.xtg_fopen(self._tmpfile, self._mode)
        else:
            try:
                cfhandle = _cxtgeo.xtg_fopen(fobj, self._mode)
            except TypeError as err:
                raise OSError(f"Cannot open file: {fobj!r}") from err

        self._cfhandle = cfhandle
        self._cfhandlecount = 1

        logger.debug("Get initial SWIG C fhandle no %s", self._cfhandlecount)
        return self._cfhandle

    def cfclose(self, strict: bool = True) -> bool:
        """
        Close SWIG C file handle by keeping track of _cfhandlecount.

        Returns:
            True if cfhandle is closed.

        """
        logger.debug("Request for closing SWIG fhandle no: %s", self._cfhandlecount)

        if self._cfhandle == 0 or self._cfhandlecount == 0:
            if strict:
                raise RuntimeError("Ask to close a nonexisting C file handle")

            self._cfhandle = 0
            self._cfhandlecount = 0
            return True

        if self._cfhandlecount > 1:
            self._cfhandlecount -= 1
            logger.debug(
                "Remaining SWIG cfhandles: %s, do not close...", self._cfhandlecount
            )
            return False

        if (
            isinstance(self.file, io.BytesIO)
            and self._cfhandle > 0
            and "w" in self._mode
        ):
            # this assures that the file pointer is in the end of the current filehandle
            npos = _cxtgeo.xtg_ftell(self._cfhandle)
            buf = bytes(npos)

            copy_code = _cxtgeo.xtg_get_fbuffer(self._cfhandle, buf)
            # Returns EXIT_SUCCESS = 0 from C
            if copy_code == 0:
                self.file.write(buf)
                _cxtgeo.xtg_fflush(self._cfhandle)
            else:
                raise RuntimeError("Could not write stream for unknown reasons")

        close_code = _cxtgeo.xtg_fclose(self._cfhandle)
        if close_code != 0:
            raise RuntimeError(f"Could not close C file, code {close_code}")

        logger.debug("File is now closed for C io: %s", self.name)

        if self._tmpfile:
            try:
                os.remove(self._tmpfile)
            except Exception as ex:  # pylint: disable=W0703
                logger.error("Could not remove tempfile for some reason: %s", ex)

        self._cfhandle = 0
        self._cfhandlecount = 0
        logger.debug(
            "Remaining SWIG cfhandles: %s, return is True", self._cfhandlecount
        )
        return True

    def detect_fformat(
        self, details: bool | None = False, suffixonly: bool | None = False
    ) -> str:
        """
        Try to deduce format from looking at file signature.

        The file signature may be the initial part of the binary file/stream but if
        that fails, the file extension is used.

        Args:
            details: If True, more info is added to the return string (useful for some
                formats) e.g. "hdf RegularSurface xtgeo". Defaults for False.
            suffixonly: If True, look at file suffix only. Defaults to False.

        Returns:
            A string with format specification, e.g. "hdf".

        """

        if not suffixonly:
            fformat = self._detect_fformat_by_contents(details)
            if fformat is not None:
                return fformat

        # if looking at contents failed, look at extension
        fmt = self._detect_format_by_extension()
        return self._validate_format(fmt)

    def _detect_fformat_by_contents(self, details: bool | None = False) -> str | None:
        # Try the read the N first bytes
        maxbuf = 100

        if isinstance(self.file, (io.BytesIO, io.StringIO)):
            self.file.seek(0)
            buf = self.file.read(maxbuf)
            self.file.seek(0)
        else:
            assert isinstance(self.file, pathlib.Path)
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
                logger.debug("Signature is hdf")

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
                logger.debug("Signature is irap binary")
                return self._validate_format("irap_binary")

        # Petromod binary regular surface
        if b"Content=Map" in buf and b"DataUnitDistance" in buf:
            logger.debug("Signature is petromod")
            return self._validate_format("petromod")

        # Eclipse binary 3D EGRID, look at FILEHEAD:
        #  'FILEHEAD'         100 'INTE'
        #   3        2016           0           0           0           0
        #  (ver)    (release)      (reserved)   (backw)    (gtype)      (dualporo)

        if len(buf) >= 24:
            fort1, name, num, _, fort2 = struct.unpack("> i 8s i 4s i", buf[:24])
            if fort1 == 16 and name == b"FILEHEAD" and num == 100 and fort2 == 16:
                # Eclipse corner point EGRID
                logger.debug("Signature is egrid")
                return self._validate_format("egrid")
            # Eclipse binary 3D UNRST, look for SEQNUM:
            #  'SEQNUM'         1 'INTE'
            if fort1 == 16 and name == b"SEQNUM  " and num == 1 and fort2 == 16:
                # Eclipse UNRST
                logger.debug("Signature is unrst")
                return self._validate_format("unrst")
            # Eclipse binary 3D INIT, look for INTEHEAD:
            #  'INTEHEAD'         411 'INTE'
            if fort1 == 16 and name == b"INTEHEAD" and num > 400 and fort2 == 16:
                # Eclipse INIT
                logger.debug("Signature is init")

                return self._validate_format("init")

        if len(buf) >= 9:
            name, _ = struct.unpack("8s b", buf[:9])
            # ROFF binary 3D
            if name == b"roff-bin":
                logger.debug("Signature is roff_binary")
                return self._validate_format("roff_binary")
            # ROFF ascii 3D
            if name == b"roff-asc":
                logger.debug("Signature is roff_ascii")
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
            logger.debug("Signature is rmswell")
            return self._validate_format("rmswell")

        return None

    def _detect_format_by_extension(self) -> str:
        """Detect format by extension."""
        if self.memstream or isinstance(self.file, (io.BytesIO, io.StringIO)):
            return "unknown"

        suffix = self.file.suffix[1:].lower()

        for fmt, variants in SUPPORTED_FORMATS.items():
            if suffix in variants:
                logger.debug("Extension hints: %s", fmt)
                return fmt

        # if none of these above are accepted, check regular expression
        # (intentional to complete all variant in loop above first before trying re())
        for fmt, variants in SUPPORTED_FORMATS.items():
            for var in variants:
                if "*" in var:
                    pattern = re.compile(var)
                    if pattern.match(suffix):
                        logger.debug("Extension by regexp hints %s", fmt)
                        return fmt

        return "unknown"

    @staticmethod
    def _validate_format(fmt: str) -> str:
        """Validate format."""
        if fmt in SUPPORTED_FORMATS.keys() or fmt == "unknown":
            return fmt
        raise RuntimeError(f"Invalid format: {fmt}")

    @staticmethod
    def generic_format_by_proposal(propose: str) -> str:
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


def inherit_docstring(inherit_from: Callable) -> Callable:
    def decorator_set_docstring(func: Callable) -> Callable:
        if func.__doc__ is None and inherit_from.__doc__ is not None:
            func.__doc__ = inherit_from.__doc__
        return func

    return decorator_set_docstring


# ----------------------------------------------------------------------------------
# Special methods for nerds, to be removed when not appplied any more
# ----------------------------------------------------------------------------------


def _convert_np_carr_int(length: int, np_array: np.ndarray) -> np.ndarray:
    """Convert numpy 1D array to C array, assuming int type.

    The numpy is always a double (float64), so need to convert first
    """
    carr = _cxtgeo.new_intarray(length)
    np_array = np_array.astype(np.int32)
    _cxtgeo.swig_numpy_to_carr_i1d(np_array, carr)
    return carr


def _convert_np_carr_double(length: int, np_array: np.ndarray) -> np.ndarray:
    """Convert numpy 1D array to C array, assuming double type."""
    carr = _cxtgeo.new_doublearray(length)
    _cxtgeo.swig_numpy_to_carr_1d(np_array, carr)
    return carr


def _convert_carr_double_np(
    length: int, carray: np.ndarray, nlen: int | None = None
) -> np.ndarray:
    """Convert a C array to numpy, assuming double type."""
    if nlen is None:
        nlen = length
    nparray = _cxtgeo.swig_carr_to_numpy_1d(nlen, carray)
    return nparray


def _get_carray(
    dataframe: pd.DataFrame, attributes: _AttrType, attrname: str
) -> np.ndarray | None:
    """
    Returns the C array pointer (via SWIG) for a given attr.

    Type conversion is double if float64, int32 if DISC attr.
    Returns None if log does not exist.
    """
    np_array = None
    if attrname in dataframe:
        np_array = dataframe[attrname].values
    else:
        return None

    nlen = len(dataframe.index)
    if attributes[attrname] == _AttrType.DISC.value:
        carr = _convert_np_carr_int(nlen, np_array)
    else:
        carr = _convert_np_carr_double(nlen, np_array)
    return carr
