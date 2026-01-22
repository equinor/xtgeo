"""CSV I/O for well data and blocked well data."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from xtgeo.common.log import null_logger
from xtgeo.io._file import FileWrapper
from xtgeo.io._welldata._blockedwell_io import BlockedWellData
from xtgeo.io._welldata._well_io import WellData, WellLog

if TYPE_CHECKING:  # pragma: no cover
    from xtgeo.common.types import FileLike

logger = null_logger(__name__)


def _read_and_filter_csv(
    wrapper: FileWrapper,
    wellname: str | None,
    wellname_col: str,
    required_cols: list[str],
) -> tuple[pd.DataFrame, str]:
    """Helper function to read CSV and filter by wellname.

    Args:
        wrapper: FileWrapper for the CSV file
        wellname: Name of the well to extract. If None, uses the first well found.
        wellname_col: Column name for well name
        required_cols: List of required column names

    Returns:
        Tuple of (filtered DataFrame, actual wellname used)

    Raises:
        ValueError: If required columns are missing or wellname not found
    """
    with wrapper.get_text_stream_read() as fwell:
        df = pd.read_csv(fwell)

    has_wellname_col = wellname_col in df.columns  # Check if wellname_col exists

    # Validate required columns (excluding wellname_col if not present)
    required_cols_to_check = [
        col for col in required_cols if col != wellname_col or has_wellname_col
    ]
    missing_cols = [col for col in required_cols_to_check if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns in CSV file: {', '.join(missing_cols)}"
        )

    # If wellname column doesn't exist, treat as single-well file
    if not has_wellname_col:
        if wellname is not None:
            logger.warning(
                f"Column '{wellname_col}' not found in CSV. "
                f"Treating as single-well file and using provided wellname '{wellname}'"
            )
            actual_wellname = wellname
        else:
            actual_wellname = str(wrapper.name) if wrapper.name else "UNKNOWN"
            logger.debug(
                f"Column '{wellname_col}' not found. Using filename as wellname: "
                f"'{actual_wellname}'"
            )
        return df, actual_wellname

    if wellname is None:
        available_wells = df[wellname_col].unique().tolist()
        if len(available_wells) == 0:
            raise ValueError(f"No wells found in CSV file (column '{wellname_col}')")
        wellname = available_wells[0]
        logger.debug(
            f"No wellname specified, using first well: '{wellname}' "
            f"(available: {available_wells})"
        )

    # Filter by wellname
    well_data = df[df[wellname_col] == wellname]

    if well_data.empty:
        available_wells = df[wellname_col].unique().tolist()
        raise ValueError(
            f"Well '{wellname}' not found in CSV file. "
            f"Available wells: {', '.join(map(str, available_wells))}"
        )

    return well_data, wellname


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
    log_cols = [col for col in well_data.columns if col not in excluded_cols]

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
    for a specific well by filtering on the wellname column. If the wellname column
    doesn't exist, treats it as a single-well file.

    Args:
        filepath: Path to CSV file or a file-like stream object
        wellname: Name of the well to extract from the CSV file. If None, uses the
            first well found in the file (or filename if no wellname column exists).
        xname: Column name for X coordinates (default: "X_UTME")
        yname: Column name for Y coordinates (default: "Y_UTMN")
        zname: Column name for Z coordinates (default: "Z_TVDSS")
        wellname_col: Column name for well name (default: "WELLNAME")

    Returns:
        WellData object

    Raises:
        ValueError: If required columns are missing or wellname not found

    Example:
        >>> from xtgeo.io._welldata._fformats._csv_table import read_csv_well
        >>> well = read_csv_well("well.csv", wellname="WELL-1")
        >>> print(f"Well: {well.name}, Records: {well.n_records}")
    """
    wrapper = FileWrapper(filepath, mode="r")

    logger.debug("Reading well data from CSV: %s", wrapper.name)

    # Read and filter CSV data
    required_cols = [xname, yname, zname, wellname_col]
    df, actual_wellname = _read_and_filter_csv(
        wrapper, wellname, wellname_col, required_cols
    )

    survey_x = df[xname].to_numpy(dtype=np.float64)
    survey_y = df[yname].to_numpy(dtype=np.float64)
    survey_z = df[zname].to_numpy(dtype=np.float64)

    xpos = float(survey_x[0])
    ypos = float(survey_y[0])
    zpos = float(survey_z[0])

    excluded_cols = [xname, yname, zname, wellname_col]
    logs = _extract_logs(df, excluded_cols)

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

    logger.debug(
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
        >>> from xtgeo.io._welldata._fformats._csv_table import write_csv_well
        >>> write_csv_well(well, "output.csv")
    """
    wrapper = FileWrapper(filepath, mode="w")

    logger.debug("Writing well data to CSV: %s", wrapper.name)

    data = {
        xname: well.survey_x,
        yname: well.survey_y,
        zname: well.survey_z,
        wellname_col: [well.name] * well.n_records,
    }

    for log in well.logs:
        data[log.name] = log.values

    df = pd.DataFrame(data)

    with wrapper.get_text_stream_write() as fwell_write:
        df.to_csv(fwell_write, index=False, header=include_header)

    logger.debug(
        "Successfully wrote well '%s' with %d records to %s",
        well.name,
        well.n_records,
        wrapper.name,
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
    """Read blocked well data from CSV file.

    The CSV file can contain data for multiple wells. This function extracts data
    for a specific well by filtering on the wellname column. If the wellname column
    doesn't exist, treats it as a single-well file.

    Args:
        filepath: Path to CSV file
        wellname: Name of the well to extract from the CSV file. If None, uses the
            first well found in the file (or filename if no wellname column exists).
        xname: Column name for X coordinates (default: "X_UTME")
        yname: Column name for Y coordinates (default: "Y_UTMN")
        zname: Column name for Z coordinates (default: "Z_TVDSS")
        i_indexname: Column name for I-indices (default: "I_INDEX")
        j_indexname: Column name for J-indices (default: "J_INDEX")
        k_indexname: Column name for K-indices (default: "K_INDEX")
        wellname_col: Column name for well name (default: "WELLNAME")

    Returns:
        BlockedWellData object

    Raises:
        ValueError: If required columns are missing or wellname not found

    Example:
        >>> from xtgeo.io._welldata._fformats._csv_table import read_csv_blockedwell
        >>> blocked_well = read_csv_blockedwell("blocked_well.csv", wellname="WELL-1")
        >>> print(f"Well: {blocked_well.name}, Records: {blocked_well.n_records}")
    """
    wrapper = FileWrapper(filepath, mode="r")

    logger.debug("Reading blocked well data from CSV: %s", wrapper.name)

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
    df, actual_wellname = _read_and_filter_csv(
        wrapper, wellname, wellname_col, required_cols
    )

    # Extract survey data
    survey_x = df[xname].to_numpy(dtype=np.float64)
    survey_y = df[yname].to_numpy(dtype=np.float64)
    survey_z = df[zname].to_numpy(dtype=np.float64)

    # Extract grid indices
    i_index = df[i_indexname].to_numpy(dtype=np.float64)
    j_index = df[j_indexname].to_numpy(dtype=np.float64)
    k_index = df[k_indexname].to_numpy(dtype=np.float64)

    # Well header position from first record
    xpos = float(survey_x[0])
    ypos = float(survey_y[0])
    zpos = float(survey_z[0])

    # Extract logs (all columns except coordinates, indices, and wellname)
    excluded_cols = [
        xname,
        yname,
        zname,
        i_indexname,
        j_indexname,
        k_indexname,
        wellname_col,
    ]
    logs = _extract_logs(df, excluded_cols)

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
    """Write blocked well data to CSV file.

    Args:
        blocked_well: BlockedWellData object to write
        filepath: Output CSV file path
        xname: Column name for X coordinates (default: "X_UTME")
        yname: Column name for Y coordinates (default: "Y_UTMN")
        zname: Column name for Z coordinates (default: "Z_TVDSS")
        i_indexname: Column name for I-indices (default: "I_INDEX")
        j_indexname: Column name for J-indices (default: "J_INDEX")
        k_indexname: Column name for K-indices (default: "K_INDEX")
        wellname_col: Column name for well name (default: "WELLNAME")
        include_header: Whether to include column headers (default: True)

    Example:
        >>> from xtgeo.io._welldata._fformats._csv_table import write_csv_blockedwell
        >>> write_csv_blockedwell(blocked_well, "output.csv")
    """
    wrapper = FileWrapper(filepath, mode="w")

    logger.debug("Writing blocked well data to CSV: %s", wrapper.name)

    # Create DataFrame with coordinates, indices, and wellname
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

    with wrapper.get_text_stream_write() as fwell_write:
        df.to_csv(fwell_write, index=False, header=include_header)

    logger.debug(
        "Successfully wrote blocked well '%s' with %d records to %s",
        blocked_well.name,
        blocked_well.n_records,
        wrapper.name,
    )
