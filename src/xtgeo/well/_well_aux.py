"""Auxillary functions for the well class

'self' is a Well() instance

"""

from __future__ import annotations

from xtgeo.common.exceptions import InvalidFileFormatError
from xtgeo.common.log import null_logger
from xtgeo.io._file import FileFormat

from . import _well_io

logger = null_logger(__name__)


def _data_reader_factory(file_format: FileFormat):
    if file_format in (FileFormat.RMSWELL, FileFormat.IRAP_ASCII):
        return _well_io.import_rms_ascii
    if file_format == FileFormat.HDF:
        return _well_io.import_hdf5_well

    extensions = FileFormat.extensions_string(
        [FileFormat.RMSWELL, FileFormat.IRAP_ASCII, FileFormat.HDF]
    )
    raise InvalidFileFormatError(
        f"File format {file_format} is invalid for Well types. "
        f"Supported formats are {extensions}."
    )
