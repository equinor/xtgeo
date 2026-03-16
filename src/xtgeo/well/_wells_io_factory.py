"""Factory functions for the Wells (plural) class I/O operations."""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

import pandas as pd

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


def wells_to_file(
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
        NotImplementedError: For HDF5 format
        ValueError: For unsupported formats
    """
    # Detect format
    fmt = wfile_wrapper.fileformat(fformat)

    if fmt == FileFormat.RMSWELL:
        _export_rms_ascii_wells(wells, wfile_wrapper)
    elif fmt == FileFormat.CSV:
        _export_csv_wells(wells, wfile_wrapper)
    elif fmt == FileFormat.HDF:
        raise NotImplementedError(
            "HDF5 format is not supported for Wells.to_file(). "
            "Export individual wells using Well.to_file() instead."
        )
    else:
        raise ValueError(
            f"File format {fformat} is not supported for Wells.to_file(). "
            "Supported formats: 'rms_ascii', 'csv'"
        )


def _import_rms_ascii_wells(wfile_wrapper: FileWrapper) -> list[Well]:
    """Import multiple wells from RMS ASCII format.

    Reads a file containing multiple wells in RMS ASCII format, each with
    its own header and data section.

    Args:
        wfile_wrapper: FileWrapper for the input file

    Returns:
        List of Well objects
    """
    from xtgeo.io._welldata._fformats._rms_ascii import _read_rms_ascii_header
    from xtgeo.io._welldata._well_io import WellData, WellLog

    logger.debug("Importing wells from RMS ASCII format: %s", wfile_wrapper.name)

    wells: list[Well] = []

    with wfile_wrapper.get_text_stream_read() as fhandle:
        # Read all lines from the file
        all_lines = fhandle.readlines()

    # Split into individual wells based on "1.0" header lines
    i = 0
    while i < len(all_lines):
        line = all_lines[i].strip()

        # Skip blank lines
        if not line:
            i += 1
            continue

        # Check if this is the start of a well (version line should be "1.0")
        if line != "1.0":
            logger.warning(
                "Expected '1.0' at start of well at line %d, got '%s'. Skipping.",
                i + 1,
                line,
            )
            i += 1
            continue

        # Collect lines for this well
        well_lines = [all_lines[i]]  # Start with "1.0"
        i += 1

        # Read header lines (description, well header, num logs)
        if i < len(all_lines):
            well_lines.append(all_lines[i])  # description
            i += 1
        if i < len(all_lines):
            well_lines.append(all_lines[i])  # well header
            i += 1
        if i < len(all_lines):
            num_logs_line = all_lines[i]
            well_lines.append(num_logs_line)
            num_logs = int(num_logs_line.strip())
            i += 1

            # Read log definition lines
            for _ in range(num_logs):
                if i < len(all_lines):
                    well_lines.append(all_lines[i])
                    i += 1

            # Read data lines until blank line or end
            while i < len(all_lines):
                if all_lines[i].strip():
                    well_lines.append(all_lines[i])
                    i += 1
                else:
                    # Blank line signals end of this well
                    i += 1
                    break

        # Parse this well from the collected lines
        well_text = "".join(well_lines)
        well_stream = io.StringIO(well_text)

        # Read using the RMS ASCII parser
        wname, xpos, ypos, rkb, lognames, wlogtype, wlogrecords = (
            _read_rms_ascii_header(well_stream)
        )

        # Read data section
        column_names = ["X_UTME", "Y_UTMN", "Z_TVDSS"] + lognames
        import numpy as np

        RMS_ASCII_UNDEF = -999.0
        dfr = pd.read_csv(
            well_stream,
            sep=r"\s+",
            header=None,
            names=column_names,
            dtype=np.float64,
            na_values=RMS_ASCII_UNDEF,
        )

        # Build WellData
        survey_x = dfr["X_UTME"].to_numpy(dtype=np.float64)
        survey_y = dfr["Y_UTMN"].to_numpy(dtype=np.float64)
        survey_z = dfr["Z_TVDSS"].to_numpy(dtype=np.float64)

        logs = []
        for log_name in lognames:
            values = dfr[log_name].to_numpy(dtype=np.float64)
            is_discrete = wlogtype[log_name] == "DISC"
            code_names = wlogrecords.get(log_name)

            log = WellLog(
                name=log_name,
                values=values,
                is_discrete=is_discrete,
                code_names=code_names,
            )
            logs.append(log)

        welldata = WellData(
            name=wname,
            xpos=xpos,
            ypos=ypos,
            zpos=rkb,
            survey_x=survey_x,
            survey_y=survey_y,
            survey_z=survey_z,
            logs=tuple(logs),
        )

        # Convert WellData to Well and add to list
        well_dict = _well_io_factory._welldata_to_well_dict(welldata)
        from xtgeo.well.well1 import Well

        well = Well(**well_dict)
        wells.append(well)

        logger.debug(
            "Read well '%s' with %d records and %d logs",
            wname,
            welldata.n_records,
            len(logs),
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


def wells_from_file(
    wfile_wrapper: FileWrapper,
    fformat: str | None,
) -> list[Well]:
    """Import Wells from file based on format.

    Args:
        wfile_wrapper: FileWrapper for the input
        fformat: File format string

    Returns:
        List of Well objects

    Raises:
        NotImplementedError: For HDF5 format
        ValueError: For unsupported formats
    """
    # Detect format
    fmt = wfile_wrapper.fileformat(fformat)

    if fmt == FileFormat.RMSWELL:
        return _import_rms_ascii_wells(wfile_wrapper)
    if fmt == FileFormat.CSV:
        return _import_csv_wells(wfile_wrapper)
    if fmt == FileFormat.HDF:
        raise NotImplementedError(
            "HDF5 format is not supported for wells_from_stacked_file(). "
            "Import individual wells using well_from_file() instead."
        )

    raise ValueError(
        f"File format {fformat} is not supported for wells_from_stacked_file(). "
        "Supported formats: 'rms_ascii', 'csv'"
    )
