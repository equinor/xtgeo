"""CSV I/O for well data and blocked well data."""

from __future__ import annotations

import io
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from xtgeo.common.log import null_logger
from xtgeo.io.welldata._well_io import WellData, WellLog

if TYPE_CHECKING:
    from xtgeo.common.types import FileLike
    from xtgeo.io.welldata._blockedwell_io import BlockedWellData

logger = null_logger(__name__)


def _read_and_filter_csv(
    filepath: FileLike,
    wellname: str | None,
    wellname_col: str,
    required_cols: list[str],
) -> pd.DataFrame:
    """Helper function to read CSV and filter by wellname.

    Args:
        filepath: Path to CSV file or a file-like stream object
        wellname: Name of the well to extract. If None, uses the first well found.
        wellname_col: Column name for well name
        required_cols: List of required column names

    Returns:
        Filtered DataFrame containing only the specified well's data

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If required columns are missing or wellname not found
    """
    # Handle file paths vs streams
    is_stream = isinstance(filepath, (io.StringIO, io.BytesIO, io.TextIOBase))

    if not is_stream:
        assert isinstance(filepath, (str, Path))
        path_obj = Path(filepath)
        if not path_obj.exists():
            raise FileNotFoundError(f"File not found: {path_obj}")

    # Read CSV file
    df = pd.read_csv(filepath)

    # Validate required columns
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns in CSV file: {', '.join(missing_cols)}"
        )

    # If wellname not specified, use the first well in the file in case the CSV
    # contains multiple wells (i.e. a stacked file).
    if wellname is None:
        available_wells = df[wellname_col].unique().tolist()
        if len(available_wells) == 0:
            raise ValueError(f"No wells found in CSV file (column '{wellname_col}')")
        wellname = available_wells[0]
        logger.info(
            f"No wellname specified, using first well: '{wellname}' "
            f"(available: {available_wells})"
        )

    well_data = df[df[wellname_col] == wellname]

    if well_data.empty:
        available_wells = df[wellname_col].unique().tolist()
        raise ValueError(
            f"Well '{wellname}' not found in CSV file. "
            f"Available wells: {', '.join(map(str, available_wells))}"
        )

    return well_data


def _extract_logs(
    well_data: pd.DataFrame,
    excluded_cols: list[str],
) -> tuple[WellLog, ...]:
    """Helper function to extract well logs from DataFrame.

    Args:
        well_data: DataFrame containing well data
        excluded_cols: List of column names to exclude (coordinates, indices, etc.)

    Returns:
        Tuple of WellLog objects
    """
    # Identify log columns (all columns except the excluded ones)
    log_cols = [col for col in well_data.columns if col not in excluded_cols]

    # Create WellLog objects for each log column
    logs = []
    for log_name in log_cols:
        values = well_data[log_name].to_numpy(dtype=np.float64)

        # Try to determine if this is a discrete or continuous log
        # If all non-NaN values are integers, treat as discrete
        is_discrete = False
        if not np.all(np.isnan(values)):
            non_nan_values = values[~np.isnan(values)]
            if np.allclose(non_nan_values, np.round(non_nan_values)):
                is_discrete = True

        log = WellLog(name=log_name, values=values, is_discrete=is_discrete)
        logs.append(log)

    return tuple(logs)


def read_csv_well(
    filepath: FileLike,
    wellname: str | None = None,
    xname: str = "X_UTME",
    yname: str = "Y_UTMN",
    zname: str = "Z_TVDSS",
    wellname_col: str = "WELLNAME",
) -> WellData:
    """Read well data from CSV file or stream.

    The CSV file can contain data for multiple wells. This function extracts data
    for a specific well by filtering on the wellname column.

    Args:
        filepath: Path to CSV file or a file-like stream object
        wellname: Name of the well to extract from the CSV file. If None, uses the
            first well found in the file.
        xname: Column name for X coordinates (default: "X_UTME")
        yname: Column name for Y coordinates (default: "Y_UTMN")
        zname: Column name for Z coordinates (default: "Z_TVDSS")
        wellname_col: Column name for well name (default: "WELLNAME")

    Returns:
        WellData object

    Raises:
        ValueError: If the specified wellname is not found in the CSV file

    Example:
        >>> from xtgeo.io.welldata._io_csv import read_csv_well
        >>> well = read_csv_well("well.csv", wellname="WELL-1")
        >>> print(f"Well: {well.name}, Records: {well.n_records}")
    """
    is_stream = isinstance(filepath, (io.StringIO, io.BytesIO, io.TextIOBase))
    if is_stream:
        logger.info("Reading well data from CSV stream")
    else:
        logger.info("Reading well data from CSV: %s", filepath)

    required_cols = [xname, yname, zname, wellname_col]
    well_data = _read_and_filter_csv(filepath, wellname, wellname_col, required_cols)

    # Get the actual wellname used (in case it was None and auto-selected)
    actual_wellname = well_data[wellname_col].iloc[0] if wellname is None else wellname

    survey_x = well_data[xname].to_numpy(dtype=np.float64)
    survey_y = well_data[yname].to_numpy(dtype=np.float64)
    survey_z = well_data[zname].to_numpy(dtype=np.float64)

    xpos = float(survey_x[0])
    ypos = float(survey_y[0])
    zpos = float(survey_z[0])

    excluded_cols = [xname, yname, zname, wellname_col]
    logs = _extract_logs(well_data, excluded_cols)

    well = WellData(
        name=actual_wellname,
        xpos=xpos,
        ypos=ypos,
        zpos=zpos,
        survey_x=survey_x,
        survey_y=survey_y,
        survey_z=survey_z,
        logs=logs,
    )

    logger.info(
        "Successfully read well '%s' with %d records and %d logs",
        actual_wellname,
        well.n_records,
        len(logs),
    )

    return well


def write_csv_well(
    well: WellData,
    filepath: FileLike,
    xname: str = "X_UTME",
    yname: str = "Y_UTMN",
    zname: str = "Z_TVDSS",
    wellname_col: str = "WELLNAME",
    include_header: bool = True,
) -> None:
    """Write well data to CSV file or stream.

    Args:
        well: WellData object to write
        filepath: Output CSV file path or a file-like stream object
        xname: Column name for X coordinates (default: "X_UTME")
        yname: Column name for Y coordinates (default: "Y_UTMN")
        zname: Column name for Z coordinates (default: "Z_TVDSS")
        wellname_col: Column name for well name (default: "WELLNAME")
        include_header: Whether to include column headers (default: True)

    Example:
        >>> from xtgeo.io.welldata._io_csv import write_csv_well
        >>> write_csv_well(well, "output.csv")
    """
    is_stream = isinstance(filepath, (io.StringIO, io.BytesIO, io.TextIOBase))

    if is_stream:
        logger.info("Writing well data to CSV stream")
    else:
        assert isinstance(filepath, (str, Path))
        path_obj = Path(filepath)
        logger.info("Writing well data to CSV: %s", path_obj)

    # Create DataFrame with coordinates and well name
    data = {
        xname: well.survey_x,
        yname: well.survey_y,
        zname: well.survey_z,
        wellname_col: [well.name] * well.n_records,
    }

    # Add other logs
    for log in well.logs:
        data[log.name] = log.values

    df = pd.DataFrame(data)

    df.to_csv(filepath, index=False, header=include_header)

    logger.debug(
        "Successfully wrote well '%s' with %d records to %s",
        well.name,
        well.n_records,
        filepath,
    )


def read_csv_blockedwell(
    filepath: FileLike,
    wellname: str | None = None,
    xname: str = "X_UTME",
    yname: str = "Y_UTMN",
    zname: str = "Z_TVDSS",
    i_indexname: str = "I_INDEX",
    j_indexname: str = "J_INDEX",
    k_indexname: str = "K_INDEX",
    wellname_col: str = "WELLNAME",
) -> BlockedWellData:
    """Read blocked well data from CSV file or stream.

    The CSV file can contain data for multiple wells. This function extracts data
    for a specific well by filtering on the wellname column.

    Args:
        filepath: Path to CSV file or a file-like stream object
        wellname: Name of the well to extract from the CSV file. If None, uses the
            first well found in the file.
        xname: Column name for X coordinates (default: "X_UTME")
        yname: Column name for Y coordinates (default: "Y_UTMN")
        zname: Column name for Z coordinates (default: "Z_TVDSS")
        i_col: Column name for I-indices (default: "I_INDEX")
        j_col: Column name for J-indices (default: "J_INDEX")
        k_col: Column name for K-indices (default: "K_INDEX")
        wellname_col: Column name for well name (default: "WELLNAME")

    Returns:
        BlockedWellData object

    Raises:
        ValueError: If the specified wellname is not found in the CSV file

    Example:
        >>> from xtgeo.io.welldata._io_csv import read_csv_blockedwell
        >>> well = read_csv_blockedwell("blocked_well.csv", wellname="WELL-1")
        >>> print(f"Well: {well.name}, Records: {well.n_records}")

    Notes:
        - The well header position (xpos, ypos, zpos) is taken from the first record
          of the filtered well data
        - All columns except the required coordinate and index columns are treated
          as well logs
        - Numeric columns become continuous logs, unless they contain only integers
          (which are treated as discrete logs)
    """
    is_stream = isinstance(filepath, (io.StringIO, io.BytesIO, io.TextIOBase))
    if is_stream:
        logger.info("Reading blocked well data from CSV stream")
    else:
        logger.info("Reading blocked well data from CSV: %s", filepath)

    # Read and filter CSV data
    required_cols = [
        xname,
        yname,
        zname,
        i_indexname,
        j_indexname,
        k_indexname,
        wellname_col,
    ]
    well_data = _read_and_filter_csv(filepath, wellname, wellname_col, required_cols)

    actual_wellname = well_data[wellname_col].iloc[0] if wellname is None else wellname

    survey_x = well_data[xname].to_numpy(dtype=np.float64)
    survey_y = well_data[yname].to_numpy(dtype=np.float64)
    survey_z = well_data[zname].to_numpy(dtype=np.float64)

    i_index = well_data[i_indexname].to_numpy(dtype=np.float64)
    j_index = well_data[j_indexname].to_numpy(dtype=np.float64)
    k_index = well_data[k_indexname].to_numpy(dtype=np.float64)

    # well header position from first record
    xpos = float(survey_x[0])
    ypos = float(survey_y[0])
    zpos = float(survey_z[0])

    excluded_cols = [
        xname,
        yname,
        zname,
        i_indexname,
        j_indexname,
        k_indexname,
        wellname_col,
    ]
    logs = _extract_logs(well_data, excluded_cols)

    # Import BlockedWellData lazily to avoid circular import
    from xtgeo.io.welldata._blockedwell_io import BlockedWellData

    blocked_well = BlockedWellData(
        name=actual_wellname,
        xpos=xpos,
        ypos=ypos,
        zpos=zpos,
        survey_x=survey_x,
        survey_y=survey_y,
        survey_z=survey_z,
        logs=logs,
        i_index=i_index,
        j_index=j_index,
        k_index=k_index,
    )

    logger.debug(
        "Successfully read blocked well '%s' with %d records and %d logs",
        actual_wellname,
        blocked_well.n_records,
        len(logs),
    )

    return blocked_well


def write_csv_blockedwell(
    blocked_well: BlockedWellData,
    filepath: FileLike,
    xname: str = "X_UTME",
    yname: str = "Y_UTMN",
    zname: str = "Z_TVDSS",
    i_indexname: str = "I_INDEX",
    j_indexname: str = "J_INDEX",
    k_indexname: str = "K_INDEX",
    wellname_col: str = "WELLNAME",
    include_header: bool = True,
) -> None:
    """Write blocked well data to CSV file or stream.

    Args:
        blocked_well: BlockedWellData object to write
        filepath: Output CSV file path or a file-like stream object
        xname: Column name for X coordinates (default: "X_UTME")
        yname: Column name for Y coordinates (default: "Y_UTMN")
        zname: Column name for Z coordinates (default: "Z_TVDSS")
        i_col: Column name for I-indices (default: "I_INDEX")
        j_col: Column name for J-indices (default: "J_INDEX")
        k_col: Column name for K-indices (default: "K_INDEX")
        wellname_col: Column name for well name (default: "WELLNAME")
        include_header: Whether to include column headers (default: True)

    Example:
        >>> from xtgeo.io.welldata._io_csv import write_csv_blockedwell
        >>> write_csv_blockedwell(blocked_well, "output.csv")
    """
    is_stream = isinstance(filepath, (io.StringIO, io.BytesIO, io.TextIOBase))

    if is_stream:
        logger.info("Writing blocked well data to CSV stream")
    else:
        assert isinstance(filepath, (str, Path))
        path_obj = Path(filepath)
        logger.info("Writing blocked well data to CSV: %s", path_obj)

    data = {
        xname: blocked_well.survey_x,
        yname: blocked_well.survey_y,
        zname: blocked_well.survey_z,
        i_indexname: blocked_well.i_index,
        j_indexname: blocked_well.j_index,
        k_indexname: blocked_well.k_index,
        wellname_col: [blocked_well.name] * blocked_well.n_records,
    }

    for log in blocked_well.logs:
        data[log.name] = log.values

    df = pd.DataFrame(data)

    df.to_csv(filepath, index=False, header=include_header)

    logger.debug(
        "Successfully wrote blocked well '%s' with %d records to %s",
        blocked_well.name,
        blocked_well.n_records,
        filepath,
    )
