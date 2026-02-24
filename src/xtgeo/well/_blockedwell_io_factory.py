"""Auxiliary factory functions for the BlockedWell class versus the new I/O system."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from xtgeo.common._xyz_enum import _AttrName, _AttrType
from xtgeo.common.exceptions import InvalidFileFormatError
from xtgeo.common.log import null_logger
from xtgeo.io._file import FileFormat
from xtgeo.io._welldata import BlockedWellData, WellFileFormat, WellLog
from xtgeo.well._well_io_factory import select_requested_logs

if TYPE_CHECKING:
    from xtgeo.well.blocked_well import BlockedWell

logger = null_logger(__name__)


def _blockedwelldata_to_blockedwell_dict(
    welldata: BlockedWellData,
    mdlogname: str | None = None,
    zonelogname: str | None = None,
    strict: bool = False,
) -> dict:
    """Convert BlockedWellData to dictionary for BlockedWell.__init__.

    Args:
        welldata: BlockedWellData object from new I/O system
        mdlogname: Name of measured depth log to use
        zonelogname: Name of zone log to use
        strict: If True, raise error if mdlogname/zonelogname not found

    Returns:
        Dictionary with keys: wname, xpos, ypos, rkb, df, wlogtypes,
        wlogrecords, mdlogname, zonelogname
    """
    if not isinstance(welldata, BlockedWellData):
        raise TypeError(
            f"Expected BlockedWellData, got {type(welldata).__name__}. "
            "Use Well class for regular wells without grid indices."
        )

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

    # Add grid indices
    df_data[_AttrName.I_INDEX.value] = welldata.i_index
    df_data[_AttrName.J_INDEX.value] = welldata.j_index
    df_data[_AttrName.K_INDEX.value] = welldata.k_index

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


def _blockedwell_to_blockedwelldata(blockedwell: BlockedWell):
    """Convert BlockedWell object to BlockedWellData object for export.

    Args:
        blockedwell: BlockedWell object to convert

    Returns:
        BlockedWellData object suitable for export via new I/O system
    """

    # Extract XYZ coordinates
    df = blockedwell.get_dataframe()
    survey_x = df[blockedwell.xname].values.astype(np.float64)
    survey_y = df[blockedwell.yname].values.astype(np.float64)
    survey_z = df[blockedwell.zname].values.astype(np.float64)

    # Exclude grid index columns from logs
    excluded_logs = {
        _AttrName.I_INDEX.value,
        _AttrName.J_INDEX.value,
        _AttrName.K_INDEX.value,
    }

    # Convert logs to WellLog objects
    logs = []
    for logname in blockedwell.lognames:
        if logname in excluded_logs:
            continue  # Skip grid indices - they're handled separately

        log_values = df[logname].values.astype(np.float64)
        logtype = blockedwell.get_logtype(logname)
        is_discrete = logtype == _AttrType.DISC.value

        # Get code_names/records
        code_names = None
        if logname in blockedwell.wlogrecords:
            code_names = blockedwell.wlogrecords[logname]

        logs.append(
            WellLog(
                name=logname,
                values=log_values,
                is_discrete=is_discrete,
                code_names=code_names,
            )
        )

    # Extract grid indices
    i_index = df[_AttrName.I_INDEX.value].values.astype(np.float64)
    j_index = df[_AttrName.J_INDEX.value].values.astype(np.float64)
    k_index = df[_AttrName.K_INDEX.value].values.astype(np.float64)

    return BlockedWellData(
        name=blockedwell.name,
        xpos=blockedwell.xpos,
        ypos=blockedwell.ypos,
        zpos=blockedwell.rkb if blockedwell.rkb is not None else 0.0,
        survey_x=survey_x,
        survey_y=survey_y,
        survey_z=survey_z,
        logs=tuple(logs),
        i_index=i_index,
        j_index=j_index,
        k_index=k_index,
    )


def _filter_blockedwelldata_logs(
    welldata: BlockedWellData,
    lognames: str | list[str] | None = "all",
    lognames_strict: bool = False,
) -> BlockedWellData:
    """Return BlockedWellData with logs filtered according to requested log names."""
    if lognames in (None, "all"):
        return welldata

    filtered_logs = select_requested_logs(welldata, lognames, lognames_strict)

    return BlockedWellData(
        name=welldata.name,
        xpos=welldata.xpos,
        ypos=welldata.ypos,
        zpos=welldata.zpos,
        survey_x=welldata.survey_x,
        survey_y=welldata.survey_y,
        survey_z=welldata.survey_z,
        logs=filtered_logs,
        i_index=welldata.i_index,
        j_index=welldata.j_index,
        k_index=welldata.k_index,
    )


def _import_blockedwelldata_rms_ascii(
    wfile,
    mdlogname=None,
    zonelogname=None,
    strict=False,
    lognames="all",
    lognames_strict=False,
):
    """Import blocked well from RMS ASCII using new I/O system."""

    welldata = BlockedWellData.from_file(wfile.file, fformat=WellFileFormat.RMS_ASCII)

    welldata = _filter_blockedwelldata_logs(welldata, lognames, lognames_strict)

    return _blockedwelldata_to_blockedwell_dict(
        welldata, mdlogname, zonelogname, strict
    )


def _import_blockedwelldata_csv(
    wfile,
    mdlogname=None,
    zonelogname=None,
    strict=False,
    lognames="all",
    lognames_strict=False,
    **kwargs,
):
    """Import blocked well from CSV using new I/O system."""
    welldata = BlockedWellData.from_file(
        wfile.file, fformat=WellFileFormat.CSV, **kwargs
    )

    welldata = _filter_blockedwelldata_logs(welldata, lognames, lognames_strict)

    return _blockedwelldata_to_blockedwell_dict(
        welldata, mdlogname, zonelogname, strict
    )


def _import_blockedwelldata_hdf5(
    wfile,
    mdlogname=None,
    zonelogname=None,
    strict=False,
    lognames="all",
    lognames_strict=False,
    **kwargs,
):
    """Import blocked well from HDF5 using new I/O system."""
    welldata = BlockedWellData.from_file(
        wfile.file, fformat=WellFileFormat.HDF5, **kwargs
    )
    welldata = _filter_blockedwelldata_logs(welldata, lognames, lognames_strict)

    return _blockedwelldata_to_blockedwell_dict(
        welldata, mdlogname, zonelogname, strict
    )


def _blocked_data_reader_factory(file_format: FileFormat):
    """Return import function for BlockedWell (uses BlockedWellData)."""
    if file_format in (FileFormat.RMSWELL, FileFormat.IRAP_ASCII):
        return _import_blockedwelldata_rms_ascii
    if file_format == FileFormat.CSV:
        return _import_blockedwelldata_csv
    if file_format == FileFormat.HDF:
        return _import_blockedwelldata_hdf5

    extensions = FileFormat.extensions_string(
        [FileFormat.RMSWELL, FileFormat.CSV, FileFormat.HDF]
    )
    raise InvalidFileFormatError(
        f"File format {file_format} is invalid for BlockedWell types. "
        f"Supported formats are {extensions}."
    )
