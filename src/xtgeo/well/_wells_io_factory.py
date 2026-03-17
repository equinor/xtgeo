"""Factory functions for the Wells (plural) class I/O operations."""

from __future__ import annotations

import io
import re
from typing import TYPE_CHECKING

import pandas as pd

from xtgeo.common.exceptions import InvalidFileFormatError
from xtgeo.common.log import null_logger
from xtgeo.io._file import FileFormat, FileWrapper

from . import _well_io_factory

if TYPE_CHECKING:
    from xtgeo.well.well1 import Well
    from xtgeo.well.wells import Wells

logger = null_logger(__name__)


def _export_rms_ascii_wells(wells: Wells, wfile_wrapper: FileWrapper) -> None:
    """Export multiple wells to RMS ASCII format.

    Each well is written with its full header and data section, separated by
    blank lines for readability.

    Args:
        wells: Wells instance containing multiple Well objects
        wfile_wrapper: FileWrapper for the output file
    """
    from xtgeo.io._welldata._fformats._rms_ascii import (
        _write_rms_ascii_data,
        _write_rms_ascii_header,
    )

    logger.debug("Exporting %d wells to RMS ASCII format", len(wells._wells))

    with wfile_wrapper.get_text_stream_write() as fhandle:
        for i, well in enumerate(wells._wells):
            if i > 0:
                fhandle.write("\n")  # Add blank line between wells for readability

            # Convert Well to WellData and write directly to file handle
            welldata = _well_io_factory._well_to_welldata(well)
            _write_rms_ascii_header(fhandle, welldata)
            _write_rms_ascii_data(fhandle, welldata)

    logger.debug("Successfully exported %d wells to RMS ASCII", len(wells._wells))


def _export_csv_wells(wells: Wells, wfile_wrapper: FileWrapper) -> None:
    """Export multiple wells to CSV format.

    All wells are combined into a single table with WELLNAME column.

    Args:
        wells: Wells instance containing multiple Well objects
        wfile_wrapper: FileWrapper for the output file
    """
    logger.debug("Exporting %d wells to CSV format", len(wells._wells))

    # Use the existing get_dataframe method which already adds WELLNAME column
    combined_df = wells.get_dataframe(filled=False)

    with wfile_wrapper.get_text_stream_write() as fhandle:
        combined_df.to_csv(fhandle, index=False)

    logger.debug(
        "Successfully exported %d wells to CSV with %d total records",
        len(wells._wells),
        len(combined_df),
    )


def wells_to_stacked_file(
    wells: Wells,
    wfile_wrapper: FileWrapper,
    fformat: str | None,
) -> None:
    """Export Wells instance to file based on format.

    Args:
        wells: Wells instance to export
        wfile_wrapper: FileWrapper for the output
        fformat: File format string

    Raises:
        InvalidFileFormatError: For unsupported formats.
    """
    # Detect format
    fmt = wfile_wrapper.fileformat(fformat)

    if fmt == FileFormat.RMSWELL:
        _export_rms_ascii_wells(wells, wfile_wrapper)
    elif fmt == FileFormat.CSV:
        _export_csv_wells(wells, wfile_wrapper)
    else:
        supported = FileFormat.extensions_string([FileFormat.RMSWELL, FileFormat.CSV])
        raise InvalidFileFormatError(
            f"File format {fformat} is not supported for Wells.to_stacked_file(). "
            f"Supported formats: {supported}"
        )


def _import_rms_ascii_wells(wfile_wrapper: FileWrapper) -> list[Well]:
    """Import multiple wells from RMS ASCII format.

    Reads a file containing multiple wells in RMS ASCII format, each with
    its own header and data section. Wells are separated by blank lines.

    Args:
        wfile_wrapper: FileWrapper for the input file

    Returns:
        List of Well objects
    """
    from xtgeo.io._welldata._fformats._rms_ascii import read_rms_ascii_well
    from xtgeo.well.well1 import Well

    logger.debug("Importing wells from RMS ASCII format: %s", wfile_wrapper.name)

    with wfile_wrapper.get_text_stream_read() as fhandle:
        text = fhandle.read()

    wells: list[Well] = []

    # Split on blank lines — each non-empty chunk is one complete well
    for chunk in re.split(r"\n[ \t]*\n", text):
        chunk = chunk.strip()
        if not chunk:
            continue

        welldata = read_rms_ascii_well(io.StringIO(chunk))
        well_dict = _well_io_factory._welldata_to_well_dict(welldata)
        well = Well(**well_dict)
        wells.append(well)

        logger.debug(
            "Read well '%s' with %d records and %d logs",
            welldata.name,
            welldata.n_records,
            len(welldata.logs),
        )

    logger.debug("Successfully imported %d wells from RMS ASCII", len(wells))
    return wells


def _import_csv_wells(wfile_wrapper: FileWrapper) -> list[Well]:
    """Import multiple wells from CSV format.

    Reads a CSV file with a WELLNAME column and splits into individual wells.

    Args:
        wfile_wrapper: FileWrapper for the input file

    Returns:
        List of Well objects

    Raises:
        ValueError: If WELLNAME column is not found
    """
    logger.debug("Importing wells from CSV format: %s", wfile_wrapper.name)

    with wfile_wrapper.get_text_stream_read() as fhandle:
        df = pd.read_csv(fhandle)

    if "WELLNAME" not in df.columns:
        raise ValueError(
            "CSV file must contain a 'WELLNAME' column to import multiple wells"
        )

    wells: list[Well] = []
    for wellname, group_df in df.groupby("WELLNAME"):  # Group by well name
        # Remove WELLNAME column from the well's dataframe
        well_df = group_df.drop(columns=["WELLNAME"]).reset_index(drop=True)

        # Assume standard column names
        xname = "X_UTME"
        yname = "Y_UTMN"
        zname = "Z_TVDSS"

        if xname not in well_df.columns or yname not in well_df.columns:
            # Try to infer coordinate columns
            possible_x = [c for c in well_df.columns if "X" in c.upper()]
            possible_y = [c for c in well_df.columns if "Y" in c.upper()]
            if possible_x and possible_y:
                xname = possible_x[0]
                yname = possible_y[0]
            else:
                raise ValueError(
                    f"Cannot find X/Y coordinate columns for well {wellname}"
                )

        if zname not in well_df.columns:
            possible_z = [
                c for c in well_df.columns if "Z" in c.upper() or "TVD" in c.upper()
            ]
            if possible_z:
                zname = possible_z[0]
        if zname not in well_df.columns:
            raise ValueError(
                "CSV file must contain a depth column (e.g. Z_TVDSS or TVD) "
                "to import wells"
            )

        # Get well head position (first point)
        xpos = float(well_df[xname].iloc[0])
        ypos = float(well_df[yname].iloc[0])
        rkb = 0.0  # Default RKB

        # Determine log types (discrete vs continuous)
        # For now, assume all logs are continuous unless they have integer values
        wlogtypes = {}
        wlogrecords = {}
        for col in well_df.columns:
            if col not in [xname, yname, zname]:
                # Check if values look like integers
                if pd.api.types.is_integer_dtype(well_df[col]) or (
                    well_df[col].notna().all()
                    and (well_df[col] == well_df[col].astype(int)).all()
                ):
                    wlogtypes[col] = "DISC"
                    # Get unique values as codes
                    unique_vals = well_df[col].dropna().astype(int).unique()
                    wlogrecords[col] = {int(v): str(v) for v in sorted(unique_vals)}
                else:
                    wlogtypes[col] = "CONT"
                    wlogrecords[col] = ("CONT", "UNK", "lin")

        from xtgeo.well.well1 import Well

        well = Well(
            wname=str(wellname),
            xpos=xpos,
            ypos=ypos,
            rkb=rkb,
            df=well_df,
            wlogtypes=wlogtypes,
            wlogrecords=wlogrecords,
        )
        wells.append(well)

        logger.debug(
            "Read well '%s' with %d records and %d logs",
            wellname,
            len(well_df),
            len([c for c in well_df.columns if c not in [xname, yname, zname]]),
        )

    logger.debug("Successfully imported %d wells from CSV", len(wells))
    return wells


def wells_from_stacked_file(
    wfile_wrapper: FileWrapper,
    fformat: str | None,
) -> list[Well]:
    """Import Wells from stacked file based on format.

    Args:
        wfile_wrapper: FileWrapper for the input
        fformat: File format string

    Returns:
        List of Well objects

    Raises:
        InvalidFileFormatError: For unsupported formats.
    """
    # Detect format
    fmt = wfile_wrapper.fileformat(fformat)

    if fmt == FileFormat.RMSWELL:
        return _import_rms_ascii_wells(wfile_wrapper)
    if fmt == FileFormat.CSV:
        return _import_csv_wells(wfile_wrapper)

    supported = FileFormat.extensions_string([FileFormat.RMSWELL, FileFormat.CSV])
    raise InvalidFileFormatError(
        f"File format {fformat} is not supported for wells_from_stacked_file(). "
        f"Supported formats: {supported}"
    )
