"""Well collection input/output functions for multiple wells from/to files."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from xtgeo.common.log import null_logger
from xtgeo.io._file import FileWrapper

if TYPE_CHECKING:
    from xtgeo.common.types import FileLike
    from xtgeo.well.blocked_well import BlockedWell
    from xtgeo.well.well1 import Well

logger = null_logger(__name__)


def import_stacked_rms_ascii(
    wfile: FileLike,
    well_class_name: str = "Well",
) -> list[Well] | list[BlockedWell]:
    """Import multiple wells from a stacked RMS ASCII file.

    This format contains multiple wells in a single file, where each well
    starts with a header block (version, undefined marker, well info, log definitions)
    followed by data rows.

    Args:
        wfile: Path to the stacked RMS ASCII file
        well_class_name: Name of the well class to instantiate ("Well" or "BlockedWell")

    Returns:
        List of Well or BlockedWell instances

    Example file format:
        1.0
        Undefined
        55_33-A-6 461292.7 5931883.3
        7
        i_index unit1 scale1
        j_index unit1 scale1
        ...
        461253.783997 5931879.317139 1693.960205 33 83 1 0 0.100390 ...
        ...

        1.0
        Undefined
        55_33-A-5 461519.2 5935692.6
        7
        ...
    """
    wfile_obj = FileWrapper(wfile, mode="r")

    wells = []
    current_well_data = []

    if wfile_obj.memstream:
        # StringIO object - use directly
        wfile_obj.file.seek(0)
        lines = wfile_obj.file.readlines()
    else:
        # File path - open and read
        with open(wfile_obj.file, "r", encoding="UTF-8") as f:
            lines = f.readlines()

    for line in lines:
        stripped = line.strip()

        # Check if this is the start of a new well (version line)
        # Version lines have only one field (the version number)
        if stripped and current_well_data and _is_version_line(stripped):
            # If we have accumulated data for a previous well, process it
            well = _parse_single_well(current_well_data, well_class_name)
            if well:
                wells.append(well)
            current_well_data = []

        # Accumulate lines for current well
        if stripped:  # Skip empty lines
            current_well_data.append(line)

    # Process the last well
    if current_well_data:
        well = _parse_single_well(current_well_data, well_class_name)
        if well:
            wells.append(well)

    logger.info(
        f"Imported {len(wells)} {well_class_name} instances from {wfile_obj.file}"
    )
    return wells


def _is_version_line(line: str) -> bool:
    """Check if a line is a version line (e.g., '1.0', '1.01').

    A version line is a single number with a decimal point, typically
    at the start of a well definition.

    Args:
        line: Stripped line to check

    Returns:
        True if this appears to be a version line
    """
    parts = line.split()
    if len(parts) != 1:
        return False

    # Check if it's a number with a decimal point
    try:
        value = float(parts[0])
        return "." in parts[0] and 0 < value < 10  # Version numbers are typically small
    except ValueError:
        return False


def _parse_single_well(
    lines: list[str],
    well_class_name: str,
) -> Well | BlockedWell | None:
    """Parse a single well from a list of lines.

    Args:
        lines: List of lines representing one well
        well_class_name: Name of the well class to instantiate

    Returns:
        Well or BlockedWell instance or None if parsing fails
    """
    import tempfile

    if len(lines) < 5:
        logger.warning("Insufficient lines for well data, skipping")
        return None

    # Create a temporary file with just this well's data
    # This allows us to reuse the existing RMS ASCII reader
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".rmswell", delete=False, encoding="UTF-8"
        ) as tmp_file:
            tmp_file.writelines(lines)
            tmp_path = tmp_file.name

        # Use the existing well reader based on class name
        if well_class_name == "BlockedWell":
            from xtgeo.well.blocked_well import BlockedWell

            well = BlockedWell._read_file(tmp_path, fformat="rms_ascii")
        else:
            from xtgeo.well.well1 import Well

            well = Well._read_file(tmp_path, fformat="rms_ascii")

        # Clean up temp file
        Path(tmp_path).unlink()

        return well

    except Exception as e:
        logger.warning(f"Failed to parse {well_class_name}: {e}")
        return None


def import_csv_wells(
    wfile: FileLike,
    well_class_name: str = "Well",
    wellname_col: str = "WELLNAME",
    xname: str = "X_UTME",
    yname: str = "Y_UTMN",
    zname: str = "Z_TVDSS",
) -> list[Well] | list[BlockedWell]:
    """Import multiple wells from a CSV file with a well name column.

    The CSV file should have a column identifying which well each row belongs to.
    This function will automatically detect unique well names and create separate
    Well or BlockedWell objects for each.

    Args:
        wfile: Path to the CSV file
        well_class_name: Name of the well class to instantiate ("Well" or "BlockedWell")
        wellname_col: Column name for well identifier (default: "WELLNAME")
        xname: Column name for X coordinates (default: "X_UTME")
        yname: Column name for Y coordinates (default: "Y_UTMN")
        zname: Column name for Z coordinates (default: "Z_TVDSS")

    Returns:
        List of Well or BlockedWell instances

    Example CSV format:
        X_UTME,Y_UTMN,Z_TVDSS,I_INDEX,J_INDEX,K_INDEX,WELLNAME,PHIT
        464789.0625,6553551.625,1620.5,109,115,0,99/19-16,0.326
        464789.0625,6553551.625,1621.5,109,115,1,99/19-16,0.316
        ...
        464790.0625,6553552.625,1625.5,109,115,5,99/19-18,0.312
        ...
    """
    import pandas as pd

    wfile_obj = FileWrapper(wfile, mode="r")

    # For StringIO, ensure we're at the beginning
    if wfile_obj.memstream:
        wfile_obj.file.seek(0)

    # Read the CSV file
    df = pd.read_csv(wfile_obj.file)

    # Check required columns
    required_cols = [wellname_col, xname, yname, zname]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns in CSV file: {', '.join(missing_cols)}"
        )

    # Get unique well names
    well_names = df[wellname_col].unique()
    logger.info(f"Found {len(well_names)} unique wells in CSV file: {list(well_names)}")

    wells = []

    # Import each well separately using the existing CSV reader
    for well_name in well_names:
        try:
            # For StringIO, seek back to beginning before each read
            if wfile_obj.memstream:
                wfile_obj.file.seek(0)

            if well_class_name == "BlockedWell":
                from xtgeo.well.blocked_well import BlockedWell

                well = BlockedWell._read_file(
                    wfile_obj.file,
                    fformat="csv",
                    wellname=well_name,
                    wellname_col=wellname_col,
                    xname=xname,
                    yname=yname,
                    zname=zname,
                )
            else:
                from xtgeo.well.well1 import Well

                well = Well._read_file(
                    wfile_obj.file,
                    fformat="csv",
                    wellname=well_name,
                    wellname_col=wellname_col,
                    xname=xname,
                    yname=yname,
                    zname=zname,
                )
            wells.append(well)
            logger.info(
                f"Imported {well_class_name} '{well_name}' "
                f"with {len(well.get_dataframe())} records"
            )
        except Exception as e:
            logger.warning(f"Failed to import well '{well_name}': {e}")

    logger.info(
        f"Imported {len(wells)} {well_class_name} instances from {wfile_obj.file}"
    )
    return wells


def export_stacked_rms_ascii(
    wells: list[Well] | list[BlockedWell],
    wfile: FileLike,
) -> None:
    """Export multiple wells to a single stacked RMS ASCII file.

    This format contains multiple wells in a single file, where each well
    starts with a header block (version, undefined marker, well info, log definitions)
    followed by data rows.

    Args:
        wells: List of Well or BlockedWell instances to export
        wfile: Path to the output stacked RMS ASCII file

    Example file format:
        1.0
        Undefined
        55_33-A-6 461292.7 5931883.3
        7
        i_index unit1 scale1
        j_index unit1 scale1
        ...
        461253.783997 5931879.317139 1693.960205 33 83 1 0 0.100390 ...
        ...

        1.0
        Undefined
        55_33-A-5 461519.2 5935692.6
        7
        ...
    """
    if not wells:
        raise ValueError("Cannot export empty wells list")

    wfile_obj = FileWrapper(wfile, mode="w")

    # Export each well to a temporary file, then concatenate
    if wfile_obj.memstream:
        out_file = wfile_obj.file
        out_file.seek(0)
        out_file.truncate()
        _export_wells_to_stream(wells, out_file)
    else:
        with open(wfile_obj.file, "w", encoding="UTF-8") as out_file:
            _export_wells_to_stream(wells, out_file)

    logger.debug(f"Exported {len(wells)} wells to {wfile_obj.file}")


def _export_wells_to_stream(wells, out_file) -> None:
    """Helper function to export wells to a file stream.

    Args:
        wells: List of Well or BlockedWell instances
        out_file: Open file stream to write to
    """
    import tempfile

    for i, well in enumerate(wells):
        # Create a temporary file for this well
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".rmswell", delete=False, encoding="UTF-8"
        ) as tmp_file:
            tmp_path = tmp_file.name

        # Export the well to the temporary file
        well.to_file(tmp_path, fformat="rms_ascii")

        # Read the temporary file and write to the output
        with open(tmp_path, "r", encoding="UTF-8") as tmp_file:
            content = tmp_file.read()
            out_file.write(content)
            # Add newline between wells (except after last well)
            if i < len(wells) - 1:
                out_file.write("\n")

        # Clean up temp file
        Path(tmp_path).unlink()


def export_csv_wells(
    wells: list[Well] | list[BlockedWell],
    wfile: FileLike,
    xname: str = "X_UTME",
    yname: str = "Y_UTMN",
    zname: str = "Z_TVDSS",
    wellname_col: str = "WELLNAME",
    **kwargs,
) -> None:
    """Export multiple wells to a CSV file with a well name column.

    The CSV file will have a column identifying which well each row belongs to.

    Args:
        wells: List of Well or BlockedWell instances to export
        wfile: Path to the output CSV file
        xname: Column name for X coordinates (default: "X_UTME")
        yname: Column name for Y coordinates (default: "Y_UTMN")
        zname: Column name for Z coordinates (default: "Z_TVDSS")
        wellname_col: Column name for well identifier (default: "WELLNAME")
        **kwargs: Additional parameters passed to pandas.DataFrame.to_csv()

    Example CSV format:
        X_UTME,Y_UTMN,Z_TVDSS,WELLNAME,PHIT,PERM
        464789.0625,6553551.625,1620.5,WELL-A,0.326,100.5
        464789.0625,6553551.625,1621.5,WELL-A,0.316,95.3
        ...
        464790.0625,6553552.625,1620.5,WELL-B,0.300,80.1
        ...
    """
    import pandas as pd

    if not wells:
        raise ValueError("Cannot export empty wells list")

    wfile_obj = FileWrapper(wfile, mode="w")

    # Collect all well dataframes with well name added
    dfs = []
    for well in wells:
        df = well.get_dataframe()
        # Rename coordinate columns if needed
        if well.xname != xname and well.xname in df.columns:
            df = df.rename(columns={well.xname: xname})
        if well.yname != yname and well.yname in df.columns:
            df = df.rename(columns={well.yname: yname})
        if well.zname != zname and well.zname in df.columns:
            df = df.rename(columns={well.zname: zname})

        # Add wellname column
        df[wellname_col] = well.name
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True, sort=False)

    # Reorder columns to put wellname and coordinates first
    coord_cols = [wellname_col, xname, yname, zname]
    other_cols = [col for col in combined_df.columns if col not in coord_cols]
    ordered_cols = coord_cols + other_cols
    combined_df = combined_df[ordered_cols]

    csv_kwargs = {"index": False}
    csv_kwargs.update(kwargs)

    if wfile_obj.memstream:
        wfile_obj.file.seek(0)
        wfile_obj.file.truncate()
        combined_df.to_csv(wfile_obj.file, **csv_kwargs)
    else:
        combined_df.to_csv(wfile_obj.file, **csv_kwargs)

    logger.info(f"Exported {len(wells)} wells to {wfile_obj.file}")
