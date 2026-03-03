"""Factory functions for the BlockedWells (plural) class I/O operations."""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

import pandas as pd

from xtgeo.common.log import null_logger
from xtgeo.io._file import FileFormat, FileWrapper

from . import _blockedwell_io_factory

if TYPE_CHECKING:
    from xtgeo.well.blocked_well import BlockedWell
    from xtgeo.well.blocked_wells import BlockedWells

logger = null_logger(__name__)


def _export_rms_ascii_blockedwells(
    blockedwells: BlockedWells, bwfile_wrapper: FileWrapper
) -> None:
    """Export multiple blocked wells to RMS ASCII format.

    Each blocked well is written with its full header and data section, separated by
    blank lines for readability.

    Args:
        blockedwells: BlockedWells instance containing multiple BlockedWell objects
        bwfile_wrapper: FileWrapper for the output file
    """
    from xtgeo.io._welldata._fformats._rms_ascii import (
        _write_rms_ascii_data,
        _write_rms_ascii_header,
    )

    logger.debug(
        "Exporting %d blocked wells to RMS ASCII format", len(blockedwells._wells)
    )

    with bwfile_wrapper.get_text_stream_write() as fhandle:
        for i, blockedwell in enumerate(blockedwells._wells):
            if i > 0:
                # Add blank line between wells for readability
                fhandle.write("\n")

            # Convert BlockedWell to BlockedWellData and write directly to file handle
            blockedwelldata = _blockedwell_io_factory._blockedwell_to_blockedwelldata(
                blockedwell
            )
            _write_rms_ascii_header(fhandle, blockedwelldata)
            _write_rms_ascii_data(fhandle, blockedwelldata)

    logger.debug(
        "Successfully exported %d blocked wells to RMS ASCII", len(blockedwells._wells)
    )


def _export_csv_blockedwells(
    blockedwells: BlockedWells, bwfile_wrapper: FileWrapper
) -> None:
    """Export multiple blocked wells to CSV format.

    All blocked wells are combined into a single table with WELLNAME column.

    Args:
        blockedwells: BlockedWells instance containing multiple BlockedWell objects
        bwfile_wrapper: FileWrapper for the output file
    """
    logger.debug("Exporting %d blocked wells to CSV format", len(blockedwells._wells))

    # Use the existing get_dataframe method which already adds WELLNAME column
    combined_df = blockedwells.get_dataframe(filled=False)

    with bwfile_wrapper.get_text_stream_write() as fhandle:
        combined_df.to_csv(fhandle, index=False)

    logger.debug(
        "Successfully exported %d blocked wells to CSV with %d total records",
        len(blockedwells._wells),
        len(combined_df),
    )


def blockedwells_to_file(
    blockedwells: BlockedWells,
    bwfile_wrapper: FileWrapper,
    fformat: str | None,
    compression: str | None = "lzf",
) -> None:
    """Export BlockedWells instance to file based on format.

    Args:
        blockedwells: BlockedWells instance to export
        bwfile_wrapper: FileWrapper for the output
        fformat: File format string
        compression: Compression for HDF5 format (not used, kept for API compatibility)

    Raises:
        NotImplementedError: For HDF5 format
        ValueError: For unsupported formats
    """
    # Detect format
    fmt = bwfile_wrapper.fileformat(fformat)

    if fmt == FileFormat.RMSWELL:
        _export_rms_ascii_blockedwells(blockedwells, bwfile_wrapper)
    elif fmt == FileFormat.CSV:
        _export_csv_blockedwells(blockedwells, bwfile_wrapper)
    elif fmt == FileFormat.HDF:
        raise NotImplementedError(
            "HDF5 format is not supported for BlockedWells.to_file(). "
            "Export individual blocked wells using BlockedWell.to_file() instead."
        )
    else:
        raise ValueError(
            f"File format {fformat} is not supported for BlockedWells.to_file(). "
            "Supported formats: 'rms_ascii', 'csv'"
        )


def _import_rms_ascii_blockedwells(
    bwfile_wrapper: FileWrapper,
    mdlogname: str | None,
    zonelogname: str | None,
    strict: bool,
) -> list[BlockedWell]:
    """Import multiple blocked wells from RMS ASCII format.

    Reads a file containing multiple blocked wells in RMS ASCII format, each with
    its own header and data section.

    Args:
        bwfile_wrapper: FileWrapper for the input file
        mdlogname: Name of measured depth log
        zonelogname: Name of zone log
        strict: If True, raise error if requested logs not found

    Returns:
        List of BlockedWell objects
    """
    from xtgeo.io._welldata import BlockedWellData, WellLog
    from xtgeo.io._welldata._fformats._rms_ascii import _read_rms_ascii_header

    logger.debug(
        "Importing blocked wells from RMS ASCII format: %s", bwfile_wrapper.name
    )

    blockedwells: list[BlockedWell] = []

    with bwfile_wrapper.get_text_stream_read() as fhandle:
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
                "Expected '1.0' at start of blocked well at line %d, got '%s'. "
                "Skipping.",
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

        # Build BlockedWellData
        survey_x = dfr["X_UTME"].to_numpy(dtype=np.float64)
        survey_y = dfr["Y_UTMN"].to_numpy(dtype=np.float64)
        survey_z = dfr["Z_TVDSS"].to_numpy(dtype=np.float64)

        # Extract grid indices (required for BlockedWellData)
        i_index = dfr.get("I_INDEX", pd.Series([np.nan] * len(dfr))).to_numpy(
            dtype=np.float64
        )
        j_index = dfr.get("J_INDEX", pd.Series([np.nan] * len(dfr))).to_numpy(
            dtype=np.float64
        )
        k_index = dfr.get("K_INDEX", pd.Series([np.nan] * len(dfr))).to_numpy(
            dtype=np.float64
        )

        logs = []
        for log_name in lognames:
            # Skip index columns - they're handled separately
            if log_name in ["I_INDEX", "J_INDEX", "K_INDEX"]:
                continue
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

        blockedwelldata = BlockedWellData(
            name=wname,
            xpos=xpos,
            ypos=ypos,
            zpos=rkb,
            survey_x=survey_x,
            survey_y=survey_y,
            survey_z=survey_z,
            logs=tuple(logs),
            i_index=i_index,
            j_index=j_index,
            k_index=k_index,
        )

        # Convert BlockedWellData to BlockedWell and add to list
        bwell_dict = _blockedwell_io_factory._blockedwelldata_to_blockedwell_dict(
            blockedwelldata, mdlogname, zonelogname, strict
        )
        from xtgeo.well.blocked_well import BlockedWell

        blockedwell = BlockedWell(**bwell_dict)
        blockedwells.append(blockedwell)

        logger.debug(
            "Read blocked well '%s' with %d records and %d logs",
            wname,
            blockedwelldata.n_records,
            len(logs),
        )

    logger.debug(
        "Successfully imported %d blocked wells from RMS ASCII", len(blockedwells)
    )
    return blockedwells


def _import_csv_blockedwells(
    bwfile_wrapper: FileWrapper,
    mdlogname: str | None,
    zonelogname: str | None,
    strict: bool,
) -> list[BlockedWell]:
    """Import multiple blocked wells from CSV format.

    Reads a CSV file with a WELLNAME column and splits into individual blocked wells.

    Args:
        bwfile_wrapper: FileWrapper for the input file
        mdlogname: Name of measured depth log
        zonelogname: Name of zone log
        strict: If True, raise error if requested logs not found

    Returns:
        List of BlockedWell objects

    Raises:
        ValueError: If WELLNAME column is not found
    """
    import numpy as np

    from xtgeo.io._welldata import BlockedWellData, WellLog

    logger.debug("Importing blocked wells from CSV format: %s", bwfile_wrapper.name)

    with bwfile_wrapper.get_text_stream_read() as fhandle:
        df = pd.read_csv(fhandle)

    if "WELLNAME" not in df.columns:
        raise ValueError(
            "CSV file must contain a 'WELLNAME' column to import multiple blocked wells"
        )

    # Group by well name
    blockedwells: list[BlockedWell] = []
    for wellname, group_df in df.groupby("WELLNAME"):
        # Remove WELLNAME column from the well's dataframe
        bwell_df = group_df.drop(columns=["WELLNAME"]).reset_index(drop=True)

        # Assume standard column names
        xname = "X_UTME"
        yname = "Y_UTMN"
        zname = "Z_TVDSS"

        if xname not in bwell_df.columns or yname not in bwell_df.columns:
            # Try to infer coordinate columns
            possible_x = [c for c in bwell_df.columns if "X" in c.upper()]
            possible_y = [c for c in bwell_df.columns if "Y" in c.upper()]
            if possible_x and possible_y:
                xname = possible_x[0]
                yname = possible_y[0]

        if xname not in bwell_df.columns or yname not in bwell_df.columns:
            raise ValueError(
                "CSV file must contain horizontal coordinate columns "
                "(e.g. X_UTME/Y_UTMN)"
            )

        if zname not in bwell_df.columns:
            possible_z = [
                c for c in bwell_df.columns if "Z" in c.upper() or "TVD" in c.upper()
            ]
            if possible_z:
                zname = possible_z[0]

        if zname not in bwell_df.columns:
            raise ValueError(
                "CSV file must contain a depth column (e.g. Z_TVDSS) "
                "to import blocked wells"
            )

        # Extract position from first row (assuming constant for well header)
        xpos = bwell_df[xname].iloc[0]
        ypos = bwell_df[yname].iloc[0]
        rkb = 0.0  # Default RKB

        # Extract grid indices (required for BlockedWellData)
        i_index = bwell_df.get("I_INDEX", pd.Series([np.nan] * len(bwell_df))).to_numpy(
            dtype=np.float64
        )
        j_index = bwell_df.get("J_INDEX", pd.Series([np.nan] * len(bwell_df))).to_numpy(
            dtype=np.float64
        )
        k_index = bwell_df.get("K_INDEX", pd.Series([np.nan] * len(bwell_df))).to_numpy(
            dtype=np.float64
        )

        # Detect log types - assume all non-coordinate columns are logs
        log_columns = [
            c
            for c in bwell_df.columns
            if c
            not in [xname, yname, zname, "WELLNAME", "I_INDEX", "J_INDEX", "K_INDEX"]
        ]

        survey_x = bwell_df[xname].to_numpy(dtype=np.float64)
        survey_y = bwell_df[yname].to_numpy(dtype=np.float64)
        survey_z = bwell_df[zname].to_numpy(dtype=np.float64)

        logs = []
        for log_name in log_columns:
            values = bwell_df[log_name].to_numpy(dtype=np.float64)

            # Simple heuristic: if all non-NaN values are integers, treat as discrete
            non_nan_values = values[~np.isnan(values)]
            is_discrete = (
                len(non_nan_values) > 0
                and np.allclose(non_nan_values, np.round(non_nan_values))
                and len(np.unique(non_nan_values)) < len(non_nan_values) * 0.5
            )

            log = WellLog(
                name=log_name,
                values=values,
                is_discrete=is_discrete,
                code_names=None,
            )
            logs.append(log)

        blockedwelldata = BlockedWellData(
            name=str(wellname),
            xpos=xpos,
            ypos=ypos,
            zpos=rkb,
            survey_x=survey_x,
            survey_y=survey_y,
            survey_z=survey_z,
            logs=tuple(logs),
            i_index=i_index,
            j_index=j_index,
            k_index=k_index,
        )

        # Convert BlockedWellData to BlockedWell
        bwell_dict = _blockedwell_io_factory._blockedwelldata_to_blockedwell_dict(
            blockedwelldata, mdlogname, zonelogname, strict
        )
        from xtgeo.well.blocked_well import BlockedWell

        blockedwell = BlockedWell(**bwell_dict)
        blockedwells.append(blockedwell)

        logger.debug(
            "Read blocked well '%s' with %d records and %d logs",
            wellname,
            len(bwell_df),
            len(log_columns),
        )

    logger.debug("Successfully imported %d blocked wells from CSV", len(blockedwells))
    return blockedwells


def blockedwells_from_file(
    bwfile_wrapper: FileWrapper,
    fformat: str | None,
    mdlogname: str | None,
    zonelogname: str | None,
    strict: bool,
) -> list[BlockedWell]:
    """Import BlockedWells from file based on format.

    Args:
        bwfile_wrapper: FileWrapper for the input
        fformat: File format string
        mdlogname: Name of measured depth log
        zonelogname: Name of zone log
        strict: If True, raise error if requested logs not found

    Returns:
        List of BlockedWell objects

    Raises:
        NotImplementedError: For HDF5 format
        ValueError: For unsupported formats
    """
    # Detect format
    fmt = bwfile_wrapper.fileformat(fformat)

    if fmt == FileFormat.RMSWELL:
        return _import_rms_ascii_blockedwells(
            bwfile_wrapper, mdlogname, zonelogname, strict
        )
    if fmt == FileFormat.CSV:
        return _import_csv_blockedwells(bwfile_wrapper, mdlogname, zonelogname, strict)
    if fmt == FileFormat.HDF:
        raise NotImplementedError(
            "HDF5 format is not supported for blockedwells_from_stacked_file(). "
            "Import individual blocked wells using blockedwell_from_file() instead."
        )

    raise ValueError(
        f"File format {fformat} is not supported for blockedwells_from_stacked_file(). "
        "Supported formats: 'rms_ascii', 'csv'"
    )
