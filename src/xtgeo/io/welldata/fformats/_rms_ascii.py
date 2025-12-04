"""RMS ASCII I/O for well data and blocked well data."""

from __future__ import annotations

import io
from pathlib import Path
from typing import TYPE_CHECKING, Union

import numpy as np
import pandas as pd

from xtgeo.common.log import null_logger
from xtgeo.io.welldata._blockedwell_io import BlockedWellData
from xtgeo.io.welldata._well_io import WellData, WellLog

if TYPE_CHECKING:
    from xtgeo.common.types import FileLike

logger = null_logger(__name__)


def read_rms_ascii_well(filepath: FileLike) -> WellData:
    """Read well data from RMS ASCII file or stream.

    Args:
        filepath: Path to RMS ASCII file or a file-like stream object

    Returns:
        WellData object

    Example:
        >>> from xtgeo.io.welldata._io_rms_ascii import read_rms_ascii_well
        >>> well = read_rms_ascii_well("well.txt")
        >>> print(f"Well: {well.name}, Records: {well.n_records}")
    """
    # Handle file paths vs streams
    is_stream = isinstance(filepath, (io.StringIO, io.BytesIO, io.TextIOBase))

    if not is_stream:
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        logger.debug("Reading well data from RMS ASCII: %s", filepath)
    else:
        logger.debug("Reading well data from RMS ASCII stream")

    wlogtype = {}
    wlogrecords = {}
    lognames = []

    lnum = 1

    # Use context manager for files, direct access for streams
    if is_stream:
        fwell = filepath
        should_close = False
        # Save current position for later pandas reading
        initial_pos = fwell.tell() if hasattr(fwell, "tell") else 0
    else:
        fwell = open(filepath, "r", encoding="UTF-8")  # noqa: SIM115
        should_close = True
        initial_pos = 0

    try:
        for line in fwell:
            if lnum <= 2:
                pass  # Skip first two lines (version and description)
            elif lnum == 3:
                # Well header: name xpos ypos [rkb]
                # rkb is optional, typically 0.0 or small value (Kelly Bushing)
                row = line.strip().split()

                if len(row) == 4:
                    # Format: name xpos ypos rkb
                    wname = row[0]
                    xpos = float(row[1])
                    ypos = float(row[2])
                    rkb = float(row[3])
                elif len(row) == 3:
                    # Format: name xpos ypos (no rkb)
                    wname = row[0]
                    xpos = float(row[1])
                    ypos = float(row[2])
                    rkb = 0.0
                else:
                    raise ValueError(f"Invalid well header format: {line}")

            elif lnum == 4:
                nlogs = int(line)
                nlogread = 0
                logger.debug("Number of logs: %s", nlogs)
                if nlogs == 0:
                    break

            else:
                # Reading log definitions
                row = line.strip().split()
                lname = row[0]
                ltype = row[1].upper()

                # Make index names uppercase
                if "_index" in lname.lower():
                    lname = lname.upper()

                lognames.append(lname)
                wlogtype[lname] = ltype

                logger.debug("Reading log name %s of type %s", lname, ltype)

                if ltype == "DISC":
                    # Discrete log: pairs of code and name
                    rxv = row[2:]
                    xdict = {int(rxv[i]): rxv[i + 1] for i in range(0, len(rxv), 2)}
                    wlogrecords[lname] = xdict
                else:
                    # Continuous log: store tuple of metadata (CONT, UNK, lin, etc.)
                    wlogrecords[lname] = tuple(row[1:])

                nlogread += 1

                if nlogread >= nlogs:
                    break

            lnum += 1
    finally:
        if should_close:
            fwell.close()

    # Read the data section
    # RMS ASCII format: X Y Z log1 log2 ... (no headers in data)
    column_names = ["X_UTME", "Y_UTMN", "Z_TVDSS"] + lognames

    # For streams, we need to reset and skip to the data section
    if is_stream:
        if hasattr(filepath, "seek"):
            filepath.seek(initial_pos)
        dfr = pd.read_csv(
            filepath,
            sep=r"\s+",
            skiprows=lnum,
            header=None,
            names=column_names,
            dtype=np.float64,
            na_values=-999,
        )
    else:
        dfr = pd.read_csv(
            filepath,
            sep=r"\s+",
            skiprows=lnum,
            header=None,
            names=column_names,
            dtype=np.float64,
            na_values=-999,
        )

    # Extract coordinates
    survey_x = dfr["X_UTME"].to_numpy(dtype=np.float64)
    survey_y = dfr["Y_UTMN"].to_numpy(dtype=np.float64)
    survey_z = dfr["Z_TVDSS"].to_numpy(dtype=np.float64)

    # Create WellLog objects for logs
    logs = []
    for log_name in lognames:
        values = dfr[log_name].to_numpy(dtype=np.float32)
        is_discrete = wlogtype[log_name] == "DISC"
        # Store metadata for both discrete and continuous logs
        code_names = wlogrecords.get(log_name)

        log = WellLog(
            name=log_name, values=values, is_discrete=is_discrete, code_names=code_names
        )
        logs.append(log)

    # Create WellData object
    well = WellData(
        name=wname,
        xpos=xpos,
        ypos=ypos,
        zpos=rkb,
        survey_x=survey_x,
        survey_y=survey_y,
        survey_z=survey_z,
        logs=tuple(logs),
    )

    logger.debug(
        "Successfully read well '%s' with %d records and %d logs",
        wname,
        well.n_records,
        len(logs),
    )

    return well


def write_rms_ascii_well(
    well: WellData,
    filepath: Union[str, Path, io.StringIO, io.BytesIO],
    precision: int = 4,
) -> None:
    """Write well data to RMS ASCII file or stream.

    Args:
        well: WellData object to write
        filepath: Output RMS ASCII file path or a file-like stream object
        precision: Number of decimal places for floats (default: 4)

    Example:
        >>> from xtgeo.io.welldata._io_rms_ascii import write_rms_ascii_well
        >>> write_rms_ascii_well(well, "output.txt")
    """
    # Handle file paths vs streams
    is_stream = isinstance(filepath, (io.StringIO, io.BytesIO, io.TextIOBase))

    if not is_stream:
        filepath_obj = Path(filepath)
        logger.debug("Writing well data to RMS ASCII: %s", filepath_obj)
        fwell = open(filepath_obj, "w", encoding="utf-8")  # noqa: SIM115
        should_close = True
    else:
        logger.debug("Writing well data to RMS ASCII stream")
        fwell = filepath
        should_close = False

    try:
        # Line 1: Version
        print("1.0", file=fwell)
        # Line 2: Description
        print("Unknown", file=fwell)
        # Line 3: Well header
        if well.zpos is None or well.zpos == 0.0:
            print(f"{well.name} {well.xpos} {well.ypos}", file=fwell)
        else:
            print(f"{well.name} {well.xpos} {well.ypos} {well.zpos}", file=fwell)
        # Line 4: Number of logs (not including X, Y, Z coordinates)
        print(f"{len(well.logs)}", file=fwell)

        # Log definitions (only for actual logs, not coordinates)
        for log in well.logs:
            log_type = "DISC" if log.is_discrete else "UNK"
            wrec = "lin"
            if log.is_discrete and log.code_names:
                # Write code names
                code_parts = []
                for code, name in log.code_names.items():
                    code_parts.extend([str(code), name])
                wrec = " ".join(code_parts)

            print(f"{log.name} {log_type} {wrec}", file=fwell)
    finally:
        if should_close:
            fwell.close()

    # Write data section
    data = {
        "X_UTME": well.survey_x,
        "Y_UTMN": well.survey_y,
        "Z_TVDSS": well.survey_z,
    }

    for log in well.logs:
        data[log.name] = log.values

    tmpdf = pd.DataFrame(data).fillna(value=-999)

    # Convert discrete logs to integers
    for log in well.logs:
        if log.is_discrete:
            tmpdf[[log.name]] = tmpdf[[log.name]].fillna(-999).astype(int)

    cformat = f"%.{precision}f"

    # For streams, append directly; for files, use append mode
    if is_stream:
        tmpdf.to_csv(
            filepath,
            sep=" ",
            header=False,
            index=False,
            float_format=cformat,
            escapechar="\\",
        )
    else:
        tmpdf.to_csv(
            filepath_obj,
            sep=" ",
            header=False,
            index=False,
            float_format=cformat,
            escapechar="\\",
            mode="a",
        )

    logger.debug(
        "Successfully wrote well '%s' with %d records to %s",
        well.name,
        well.n_records,
        filepath,
    )


def read_rms_ascii_blockedwell(
    filepath: Union[str, Path],
) -> BlockedWellData:
    """Read blocked well data from RMS ASCII file.

    This assumes the file contains I_INDEX, J_INDEX, K_INDEX columns.

    Args:
        filepath: Path to RMS ASCII file

    Returns:
        BlockedWellData object

    Example:
        >>> from xtgeo.io.welldata._io_rms_ascii import read_rms_ascii_blockedwell
        >>> well = read_rms_ascii_blockedwell("blocked_well.txt")
        >>> print(f"Well: {well.name}, Records: {well.n_records}")
    """
    # First read as regular well
    well = read_rms_ascii_well(filepath)

    # Extract grid indices from logs
    i_index_log = well.get_log("I_INDEX")
    j_index_log = well.get_log("J_INDEX")
    k_index_log = well.get_log("K_INDEX")

    if not i_index_log or not j_index_log or not k_index_log:
        raise ValueError(
            "File does not contain I_INDEX, J_INDEX, and K_INDEX logs "
            "required for blocked well"
        )

    # Remove index logs from the logs tuple
    remaining_logs = tuple(
        log for log in well.logs if log.name not in ["I_INDEX", "J_INDEX", "K_INDEX"]
    )

    # Create BlockedWellData object
    blocked_well = BlockedWellData(
        name=well.name,
        xpos=well.xpos,
        ypos=well.ypos,
        zpos=well.zpos,
        survey_x=well.survey_x,
        survey_y=well.survey_y,
        survey_z=well.survey_z,
        logs=remaining_logs,
        i_index=i_index_log.values,
        j_index=j_index_log.values,
        k_index=k_index_log.values,
    )

    logger.debug(
        "Successfully read blocked well '%s' with %d records and %d blocked cells",
        blocked_well.name,
        blocked_well.n_records,
        blocked_well.n_blocked_cells,
    )

    return blocked_well


def write_rms_ascii_blockedwell(
    blocked_well: BlockedWellData,
    filepath: Union[str, Path],
    precision: int = 4,
) -> None:
    """Write blocked well data to RMS ASCII file.

    Args:
        blocked_well: BlockedWellData object to write
        filepath: Output RMS ASCII file path
        precision: Number of decimal places for floats (default: 4)

    Example:
        >>> from xtgeo.io.welldata._io_rms_ascii import write_rms_ascii_blockedwell
        >>> write_rms_ascii_blockedwell(blocked_well, "output.txt")
    """
    # Create temporary index logs
    i_log = WellLog(name="I_INDEX", values=blocked_well.i_index, is_discrete=False)
    j_log = WellLog(name="J_INDEX", values=blocked_well.j_index, is_discrete=False)
    k_log = WellLog(name="K_INDEX", values=blocked_well.k_index, is_discrete=False)

    # Create a temporary WellData object with index logs added
    temp_logs = blocked_well.logs + (i_log, j_log, k_log)

    temp_well = WellData(
        name=blocked_well.name,
        xpos=blocked_well.xpos,
        ypos=blocked_well.ypos,
        zpos=blocked_well.zpos,
        survey_x=blocked_well.survey_x,
        survey_y=blocked_well.survey_y,
        survey_z=blocked_well.survey_z,
        logs=temp_logs,
    )

    # Use the regular well writer
    write_rms_ascii_well(temp_well, filepath, precision)

    logger.debug(
        "Successfully wrote blocked well '%s' with %d blocked cells to %s",
        blocked_well.name,
        blocked_well.n_blocked_cells,
        filepath,
    )
