"""A FileWrapper class to wrap around files and streams representing files."""

from __future__ import annotations

import io
import os
import pathlib
import platform
import re
import struct
import uuid
from enum import Enum
from os.path import join
from tempfile import mkstemp
from typing import TYPE_CHECKING, Literal, Union

import xtgeo._cxtgeo
from xtgeo.common.exceptions import InvalidFileFormatError
from xtgeo.common.log import null_logger

if TYPE_CHECKING:
    from xtgeo.common.types import FileLike
    from xtgeo.cube import Cube
    from xtgeo.grid3d import Grid, GridProperties, GridProperty
    from xtgeo.surface import RegularSurface, Surfaces
    from xtgeo.wells import BlockedWell, BlockedWells, Well, Wells
    from xtgeo.xyz import Points, Polygons

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

VALID_FILE_ALIASES = ["$fmu-v1", "$md5sum", "$random"]


class FileFormat(Enum):
    RMSWELL = ["rmswell", "rmsw", "w", "bw"]
    ROFF_BINARY = ["roff_binary", "roff", "roff_bin", "roff-bin", "roffbin", "roff.*"]
    ROFF_ASCII = ["roff_ascii", "roff_asc", "roff-asc", "roffasc", "asc"]
    EGRID = ["egrid", "eclipserun"]
    FEGRID = ["fegrid"]
    INIT = ["init"]
    FINIT = ["finit"]
    UNRST = ["unrst"]
    FUNRST = ["funrst"]
    GRDECL = ["grdecl"]
    BGRDECL = ["bgrdecl"]
    IRAP_BINARY = [
        "irap_binary",
        "irap_bin",
        "irapbinary",
        "irap",
        "rms_binary",
        "irapbin",
        "gri",
    ]
    IRAP_ASCII = [
        "irapascii",
        "irap_txt",
        "irap_ascii",
        "irap_asc",
        "rms_ascii",
        "irapasc",
        "fgr",
    ]
    HDF = ["hdf", "hdf5", "h5"]
    SEGY = ["segy", "sgy", "segy.*"]
    STORM = ["storm", "storm_binary"]
    ZMAP_ASCII = ["zmap", "zmap+", "zmap_ascii", "zmap-ascii", "zmap-asc", "zmap.*"]
    IJXYZ = ["ijxyz"]
    PETROMOD = ["pmd", "petromod"]
    XTG = ["xtg", "xtgeo", "xtgf", "xtgcpprop", "xtg.*"]
    XYZ = ["xyz", "poi", "pol"]
    RMS_ATTR = ["rms_attr", "rms_attrs", "rmsattr.*"]
    UNKNOWN = ["unknown"]

    @staticmethod
    def extensions_string(formats: list[FileFormat]) -> str:
        return ", ".join([f"'{item}'" for fmt in formats for item in fmt.value])


class FileWrapper:
    """
    A private class for file/stream handling in/out of XTGeo and CXTGeo.

    Interesting attributes:

    xfile = FileWrapper(..some Path or str or BytesIO ...)

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
        filelike: FileLike,
        mode: Literal["r", "w", "rb", "wb"] = "rb",
        obj: XTGeoObject = None,
    ) -> None:
        logger.debug("Init ran for FileWrapper")

        if not isinstance(filelike, (str, pathlib.Path, io.BytesIO, io.StringIO)):
            raise RuntimeError(
                f"Cannot instantiate {self.__class__} from "
                f"{filelike} of type {type(filelike)}. Expected "
                f"a str, pathlib.Path, io.BytesIO, or io.StringIO."
            )

        if isinstance(filelike, str):
            filelike = pathlib.Path(filelike)

        self._file: pathlib.Path | io.BytesIO | io.StringIO = filelike
        self._memstream = isinstance(self._file, (io.BytesIO, io.StringIO))
        self._mode = mode
        # String streams cannot be binary
        if isinstance(self._file, io.StringIO) and mode in ("rb", "wb"):
            self._mode = "r" if mode == "rb" else "w"

        if obj and not self._memstream:
            self.resolve_alias(obj)

        self._cfhandle = 0
        self._cfhandlecount = 0

        self._tmpfile: str | None = None

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
            >>> xx = FileWrapper("/tmp/$md5sum.gri", "rb", surf)
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
            newname = obj.generate_hash()  # type: ignore
        elif stem == "$random":
            newname = uuid.uuid4().hex  # random name
        elif stem == "$fmu-v1":
            # will make name such as topvalysar--avg_porosity based on metadata
            short = obj.metadata.opt.shortname.lower().replace(" ", "_")  # type: ignore
            desc = obj.metadata.opt.description.lower().replace(" ", "_")  # type: ignore
            date = obj.metadata.opt.datetime  # type: ignore
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

        if "r" in self._mode and (not self.file.is_file() or not self.exists()):
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
        logger.debug("Checking folder...")

        if self.memstream or isinstance(self.file, (io.BytesIO, io.StringIO)):
            logger.info(
                "Cannot check folder status of an in-memory file, just return True"
            )
            return True

        folder = self.file.parent
        if raisetext is None:
            raisetext = f"Folder {folder.name} does not exist or cannot be accessed"

        if not folder.exists():
            if raiseerror:
                raise raiseerror(raisetext)

            return False

        return True

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
                cfhandle = xtgeo._cxtgeo.xtg_fopen_bytestream(fobj, self._mode)
            else:
                cfhandle = xtgeo._cxtgeo.xtg_fopen(self._tmpfile, self._mode)
        else:
            try:
                cfhandle = xtgeo._cxtgeo.xtg_fopen(fobj, self._mode)
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
            npos = xtgeo._cxtgeo.xtg_ftell(self._cfhandle)
            buf = bytes(npos)

            copy_code = xtgeo._cxtgeo.xtg_get_fbuffer(self._cfhandle, buf)
            # Returns EXIT_SUCCESS = 0 from C
            if copy_code == 0:
                self.file.write(buf)
                xtgeo._cxtgeo.xtg_fflush(self._cfhandle)
            else:
                raise RuntimeError("Could not write stream for unknown reasons")

        close_code = xtgeo._cxtgeo.xtg_fclose(self._cfhandle)
        if close_code != 0:
            raise RuntimeError(f"Could not close C file, code {close_code}")

        logger.debug("File is now closed for C io: %s", self.name)

        if self._tmpfile:
            try:
                os.remove(self._tmpfile)
            except Exception as ex:
                logger.error("Could not remove tempfile for some reason: %s", ex)

        self._cfhandle = 0
        self._cfhandlecount = 0
        logger.debug(
            "Remaining SWIG cfhandles: %s, return is True", self._cfhandlecount
        )
        return True

    def fileformat(self, fileformat: str | None = None) -> FileFormat:
        """
        Try to deduce format from looking at file suffix or contents.

        The file signature may be the initial part of the binary file/stream but if
        that fails, the file extension is used.

        Args:
            fileformat (str, None): An optional user-provided string indicating what
              kind of file this is.

        Raises:
            A ValueError if an invalid or unsupported format is encountered.

        Returns:
            A FileFormat.
        """
        if fileformat:
            fileformat = fileformat.lower()
        self._validate_fileformat(fileformat)

        fmt = self._format_from_suffix(fileformat)
        if fmt == FileFormat.UNKNOWN:
            fmt = self._format_from_contents()
        if fmt == FileFormat.UNKNOWN:
            raise InvalidFileFormatError(
                f"File format {fileformat} is unknown or unsupported"
            )
        return fmt

    def _validate_fileformat(self, fileformat: str | None) -> None:
        """Validate that the pass format string is one XTGeo supports.

        Raises:
            ValueError: if format is unknown or unsupported
        """
        if not fileformat or fileformat == "guess":
            return
        for fmt in FileFormat:
            if fileformat in fmt.value:
                return
            for regex in fmt.value:
                if "*" in regex and re.compile(regex).match(fileformat):
                    return
        raise InvalidFileFormatError(
            f"File format {fileformat} is unknown or unsupported"
        )

    def _format_from_suffix(self, fileformat: str | None = None) -> FileFormat:
        """Detect format by the file suffix."""
        if not fileformat or fileformat == "guess":
            if isinstance(self.file, (io.BytesIO, io.StringIO)):
                return FileFormat.UNKNOWN
            fileformat = self.file.suffix[1:].lower()

        for fmt in FileFormat:
            if fileformat in fmt.value:
                logger.debug("Extension hints: %s", fmt)
                return fmt

        # Fall back to regex
        for fmt in FileFormat:
            for regex in fmt.value:
                if "*" in regex and re.compile(regex).match(fileformat):
                    logger.debug("Extension by regexp hints %s", fmt)
                    return fmt

        return FileFormat.UNKNOWN

    def _format_from_contents(self) -> FileFormat:
        BUFFER_SIZE = 128
        buffer = bytearray(BUFFER_SIZE)

        if isinstance(self.file, (io.BytesIO, io.StringIO)):
            mark = self.file.tell()
            # Encode to bytes if string
            if isinstance(self.file, io.StringIO):
                strbuf = self.file.read(BUFFER_SIZE)
                buffer = bytearray(strbuf.encode())
            else:
                self.file.readinto(buffer)
            self.file.seek(mark)
        else:
            if not self.exists():
                raise FileNotFoundError(f"File {self.name} does not exist")
            with open(self.file, "rb") as fhandle:
                fhandle.readinto(buffer)

        # HDF format, different variants
        if len(buffer) >= 4:
            _, hdf = struct.unpack("b 3s", buffer[:4])
            if hdf == b"HDF":
                logger.debug("Signature is hdf")
                return FileFormat.HDF

        # Irap binary regular surface format
        if len(buffer) >= 8:
            fortranblock, gricode = struct.unpack(">ii", buffer[:8])
            if fortranblock == 32 and gricode == -996:
                logger.debug("Signature is irap binary")
                return FileFormat.IRAP_BINARY

        # Petromod binary regular surface
        if b"Content=Map" in buffer and b"DataUnitDistance" in buffer:
            logger.debug("Signature is petromod")
            return FileFormat.PETROMOD

        # Eclipse binary 3D EGRID, look at FILEHEAD:
        #  'FILEHEAD'         100 'INTE'
        #   3        2016           0           0           0           0
        #  (ver)    (release)      (reserved)   (backw)    (gtype)      (dualporo)
        if len(buffer) >= 24:
            fort1, name, num, _, fort2 = struct.unpack("> i 8s i 4s i", buffer[:24])
            if fort1 == 16 and name == b"FILEHEAD" and num == 100 and fort2 == 16:
                logger.debug("Signature is egrid")
                return FileFormat.EGRID
            # Eclipse binary 3D UNRST, look for SEQNUM:
            #  'SEQNUM'         1 'INTE'
            if fort1 == 16 and name == b"SEQNUM  " and num == 1 and fort2 == 16:
                logger.debug("Signature is unrst")
                return FileFormat.UNRST
            # Eclipse binary 3D INIT, look for INTEHEAD:
            #  'INTEHEAD'         411 'INTE'
            if fort1 == 16 and name == b"INTEHEAD" and num > 400 and fort2 == 16:
                logger.debug("Signature is init")
                return FileFormat.INIT

        if len(buffer) >= 9:
            name, _ = struct.unpack("8s b", buffer[:9])
            if name == b"roff-bin":
                logger.debug("Signature is roff_binary")
                return FileFormat.ROFF_BINARY
            if name == b"roff-asc":
                logger.debug("Signature is roff_ascii")
                return FileFormat.ROFF_ASCII

        # RMS well format (ascii)
        # 1.0
        # Unknown
        # WELL12 90941.63200000004 5506367.711 23.0
        # ...
        # The signature here is one float in first line with values 1.0; one string
        # in second line; and 3 or 4 items in the next (sometimes RKB is missing)
        try:
            xbuf = buffer.decode().split("\n")
        except UnicodeDecodeError:
            return FileFormat.UNKNOWN

        if (
            len(xbuf) >= 3
            and xbuf[0] == "1.0"
            and len(xbuf[1]) >= 1
            and len(xbuf[2]) >= 10
        ):
            logger.debug("Signature is rmswell")
            return FileFormat.RMSWELL

        return FileFormat.UNKNOWN
