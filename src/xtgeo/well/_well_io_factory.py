"""Auxiliary factory functions for the well class versus the new I/O system."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from xtgeo.common._xyz_enum import _AttrName, _AttrType
from xtgeo.common.exceptions import InvalidFileFormatError
from xtgeo.common.log import null_logger
from xtgeo.io._file import FileFormat
from xtgeo.io._welldata import WellData, WellFileFormat

if TYPE_CHECKING:
    from xtgeo.well.well1 import Well

logger = null_logger(__name__)


def select_requested_logs(
    welldata: Any,
    lognames: str | list[str] | None = "all",
    lognames_strict: bool = False,
) -> tuple[Any, ...]:
    """Return selected logs from WellData-like object.

    The input object is expected to provide ``get_log(name)``, ``log_names`` and
    ``name`` attributes.
    """
    if lognames in (None, "all"):
        return tuple(welldata.logs)

    requested = [lognames] if isinstance(lognames, str) else list(lognames)

    filtered_logs: list[Any] = []
    missing_logs: list[str] = []
    for logname in requested:
        log = welldata.get_log(logname)
        if log is None:
            missing_logs.append(logname)
        else:
            filtered_logs.append(log)

    if missing_logs and lognames_strict:
        available = ", ".join(welldata.log_names)
        missing = ", ".join(missing_logs)
        raise ValueError(
            f"Requested log(s) not found in well {welldata.name}: {missing}. "
            f"Available logs: {available}"
        )

    return tuple(filtered_logs)


def _welldata_to_well_dict(
    welldata: WellData,
    mdlogname: str | None = None,
    zonelogname: str | None = None,
    strict: bool = False,
) -> dict:
    """Convert WellData object to dictionary format expected by Well.__init__.

    Args:
        welldata: WellData object from new I/O system
        mdlogname: Name of measured depth log to use
        zonelogname: Name of zone log to use
        strict: If True, raise error if mdlogname/zonelogname not found

    Returns:
        Dictionary with keys: wname, xpos, ypos, rkb, df, wlogtypes, wlogrecords,
        mdlogname, zonelogname
    """
    # Build wlogtypes and wlogrecords dictionaries
    wlogtypes = {}
    wlogrecords = {}

    for log in welldata.logs:
        if log.is_discrete:
            wlogtypes[log.name] = _AttrType.DISC.value
            if isinstance(log.code_names, dict):
                wlogrecords[log.name] = log.code_names
            else:
                # For discrete logs without code names, create a default
                wlogrecords[log.name] = {}
        else:
            wlogtypes[log.name] = _AttrType.CONT.value
            if isinstance(log.code_names, tuple):
                wlogrecords[log.name] = log.code_names
            else:
                wlogrecords[log.name] = ("CONT", "UNK", "lin")

    # Build dataframe with XYZ and logs
    df_data = {
        _AttrName.XNAME.value: welldata.survey_x,
        _AttrName.YNAME.value: welldata.survey_y,
        _AttrName.ZNAME.value: welldata.survey_z,
    }

    for log in welldata.logs:
        df_data[log.name] = log.values

    df = pd.DataFrame(df_data)

    # Check and validate mdlogname and zonelogname
    if mdlogname is not None and mdlogname not in df.columns:
        msg = (
            f"mdlogname={mdlogname} was requested but no such log found for "
            f"well {welldata.name}"
        )
        if strict:
            raise ValueError(msg)
        logger.warning(msg)
        mdlogname = None

    if zonelogname is not None and zonelogname not in df.columns:
        msg = (
            f"zonelogname={zonelogname} was requested but no such log found "
            f"for well {welldata.name}"
        )
        if strict:
            raise ValueError(msg)
        logger.warning(msg)
        zonelogname = None

    return {
        "wname": welldata.name,
        "xpos": welldata.xpos,
        "ypos": welldata.ypos,
        "rkb": welldata.zpos,
        "df": df,
        "wlogtypes": wlogtypes,
        "wlogrecords": wlogrecords,
        "mdlogname": mdlogname,
        "zonelogname": zonelogname,
    }


def _well_to_welldata(well: Well) -> WellData:
    """Convert Well object to WellData object for export.

    Args:
        well: Well object to convert

    Returns:
        WellData object suitable for export via new I/O system
    """
    from xtgeo.io._welldata._well_io import WellLog

    # Extract XYZ coordinates
    df = well.get_dataframe()
    survey_x = df[well.xname].values.astype(np.float64)
    survey_y = df[well.yname].values.astype(np.float64)
    survey_z = df[well.zname].values.astype(np.float64)

    # Convert logs to WellLog objects
    logs = []
    for logname in well.lognames:
        log_values = df[logname].values.astype(np.float64)
        logtype = well.get_logtype(logname)
        is_discrete = logtype == _AttrType.DISC.value

        # Get code_names/records
        code_names = None
        if logname in well.wlogrecords:
            code_names = well.wlogrecords[logname]

        logs.append(
            WellLog(
                name=logname,
                values=log_values,
                is_discrete=is_discrete,
                code_names=code_names,
            )
        )

    return WellData(
        name=well.name,
        xpos=well.xpos,
        ypos=well.ypos,
        zpos=well.rkb if well.rkb is not None else 0.0,
        survey_x=survey_x,
        survey_y=survey_y,
        survey_z=survey_z,
        logs=tuple(logs),
    )


def _filter_welldata_logs(
    welldata: WellData,
    lognames: str | list[str] | None = "all",
    lognames_strict: bool = False,
) -> WellData:
    """Return WellData with logs filtered according to requested log names."""
    if lognames in (None, "all"):
        return welldata

    filtered_logs = select_requested_logs(welldata, lognames, lognames_strict)

    return WellData(
        name=welldata.name,
        xpos=welldata.xpos,
        ypos=welldata.ypos,
        zpos=welldata.zpos,
        survey_x=welldata.survey_x,
        survey_y=welldata.survey_y,
        survey_z=welldata.survey_z,
        logs=filtered_logs,
    )


def _import_welldata_rms_ascii(
    wfile,
    mdlogname=None,
    zonelogname=None,
    strict=False,
    lognames="all",
    lognames_strict=False,
):
    """Import regular well from RMS ASCII using new I/O system."""
    welldata = WellData.from_file(wfile.file, fformat=WellFileFormat.RMS_ASCII)

    welldata = _filter_welldata_logs(welldata, lognames, lognames_strict)

    return _welldata_to_well_dict(welldata, mdlogname, zonelogname, strict)


def _import_welldata_csv(
    wfile,
    mdlogname=None,
    zonelogname=None,
    strict=False,
    lognames="all",
    lognames_strict=False,
    **kwargs,
):
    """Import regular well from CSV using new I/O system."""
    welldata = WellData.from_file(wfile.file, fformat=WellFileFormat.CSV, **kwargs)

    welldata = _filter_welldata_logs(welldata, lognames, lognames_strict)

    return _welldata_to_well_dict(welldata, mdlogname, zonelogname, strict)


def _import_welldata_hdf5(
    wfile,
    mdlogname=None,
    zonelogname=None,
    strict=False,
    lognames="all",
    lognames_strict=False,
    **kwargs,
):
    """Import regular well from HDF5 using new I/O system."""
    welldata = WellData.from_file(wfile.file, fformat=WellFileFormat.HDF5, **kwargs)
    welldata = _filter_welldata_logs(welldata, lognames, lognames_strict)

    return _welldata_to_well_dict(welldata, mdlogname, zonelogname, strict)


def _data_reader_factory(file_format: FileFormat):
    """Return import function for regular Well (uses WellData)."""
    if file_format in (FileFormat.RMSWELL, FileFormat.IRAP_ASCII):
        return _import_welldata_rms_ascii
    if file_format == FileFormat.CSV:
        return _import_welldata_csv
    if file_format == FileFormat.HDF:
        return _import_welldata_hdf5

    extensions = FileFormat.extensions_string(
        [FileFormat.RMSWELL, FileFormat.CSV, FileFormat.HDF]
    )
    raise InvalidFileFormatError(
        f"File format {file_format} is invalid for Well types. "
        f"Supported formats are {extensions}."
    )
