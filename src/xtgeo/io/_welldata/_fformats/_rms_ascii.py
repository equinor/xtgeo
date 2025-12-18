"""RMS ASCII I/O for well data and blocked well data."""

from __future__ import annotations

import io
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, TextIO

import numpy as np
import pandas as pd

from xtgeo.common.log import null_logger
from xtgeo.io._welldata._blockedwell_io import BlockedWellData
from xtgeo.io._welldata._well_io import WellData, WellLog

if TYPE_CHECKING:  # pragma: no cover
    from xtgeo.common.types import FileLike

logger = null_logger(__name__)

RMS_ASCII_UNDEF: Final[float] = -999.0


def read_rms_ascii_well(filepath: FileLike) -> WellData:
    """Read well data from RMS ASCII file or stream.

    Args:
        filepath: Path to RMS ASCII file or a file-like stream object

    Returns:
        WellData object

    """
    is_stream = isinstance(filepath, (io.StringIO, io.BytesIO, io.TextIOBase))

    path_obj: Path | None = None
    if not is_stream:
        assert isinstance(filepath, (str, Path))
        path_obj = Path(filepath)

        if not path_obj.exists():
            raise FileNotFoundError(f"File not found: {path_obj}")
        logger.debug("Reading well data from RMS ASCII: %s", path_obj)
    else:
        logger.debug("Reading well data from RMS ASCII stream")

    wlogtype: dict[str, str] = {}
    wlogrecords: dict[str, dict[int, str] | tuple[Any, ...]] = {}
    lognames: list[str] = []

    lnum = 1

    fwell: TextIO
    if is_stream:
        fwell = filepath  # type: ignore[assignment]
        should_close = False
    else:
        assert path_obj is not None
        fwell = open(path_obj, "r", encoding="UTF-8")  # noqa: SIM115
        should_close = True

    try:
        for line in fwell:
            if lnum <= 2:
                pass  # Skip first two lines (version and description)
            elif lnum == 3:
                # Well header: name xpos ypos [rkb]
                # Well name can contain spaces, so parse from right to left
                # rkb is optional, typically 0.0 or small value (Kelly Bushing height)
                row = line.strip().split()

                if len(row) < 3:
                    raise ValueError(f"Invalid well header format: {line}")

                # Try parsing with RKB (last 3 values are xpos, ypos, rkb)
                try:
                    rkb = float(row[-1])
                    ypos = float(row[-2])
                    xpos = float(row[-3])
                    # Everything before the last 3 values is the well name
                    wname: str = " ".join(row[:-3])
                except (ValueError, IndexError):
                    # No RKB, parse as name xpos ypos (last 2 values are xpos, ypos)
                    try:
                        ypos = float(row[-1])
                        xpos = float(row[-2])
                        # Everything before the last 2 values is the well name
                        wname = " ".join(row[:-2])
                        rkb = 0.0
                    except (ValueError, IndexError):
                        raise ValueError(f"Invalid well header format: {line}")

            elif lnum == 4:
                nlogs = int(line)
                nlogread = 0
                logger.debug("Number of logs: %s", nlogs)
                if nlogs == 0:
                    break

            else:
                row = line.strip().split()
                lname_raw = str(row[0])
                ltype = row[1].upper()

                if "_index" in lname_raw.lower():
                    lname: str = lname_raw.upper()
                else:
                    lname = lname_raw

                lognames.append(lname)
                wlogtype[lname] = ltype

                logger.debug("Reading log name %s of type %s", lname, ltype)

                if ltype == "DISC":
                    # Discrete log: pairs of code and name
                    rxv = row[2:]
                    xdict: dict[int, str] = {
                        int(rxv[i]): str(rxv[i + 1]) for i in range(0, len(rxv), 2)
                    }
                    wlogrecords[lname] = xdict
                else:
                    # Continuous log: store tuple of metadata (CONT, UNK, lin, etc.)
                    wlogrecords[lname] = tuple(row[1:])

                nlogread += 1

                if nlogread >= nlogs:
                    break

            lnum += 1

        # Read the data section
        # The stream is already positioned right after the header, so read directly
        column_names = ["X_UTME", "Y_UTMN", "Z_TVDSS"] + lognames

        dfr = pd.read_csv(  # type: ignore[call-overload]
            fwell,
            sep=r"\s+",
            header=None,
            names=column_names,
            dtype=np.float64,
            na_values=RMS_ASCII_UNDEF,
        )
    finally:
        if should_close:
            fwell.close()

    # Extract coordinates
    survey_x = dfr["X_UTME"].to_numpy(dtype=np.float64)
    survey_y = dfr["Y_UTMN"].to_numpy(dtype=np.float64)
    survey_z = dfr["Z_TVDSS"].to_numpy(dtype=np.float64)

    logs = []
    for log_name in lognames:
        values = dfr[log_name].to_numpy(dtype=np.float64)
        is_discrete = wlogtype[log_name] == "DISC"
        # Store metadata for both discrete and continuous logs
        code_names = wlogrecords.get(log_name)

        log = WellLog(
            name=log_name, values=values, is_discrete=is_discrete, code_names=code_names
        )
        logs.append(log)

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
    filepath: FileLike,
    precision: int = 4,
) -> None:
    """Write well data to RMS ASCII file or stream.

    Args:
        well: WellData object to write
        filepath: Output RMS ASCII file path or a file-like stream object
        precision: Number of decimal places for floats (default: 4)

    """
    # Handle file paths vs streams
    is_stream = isinstance(filepath, (io.StringIO, io.BytesIO, io.TextIOBase))

    if not is_stream:
        assert isinstance(filepath, (str, Path))
        path_obj = Path(filepath)
        logger.debug("Writing well data to RMS ASCII: %s", path_obj)
        fwell_write: TextIO = open(path_obj, "w", encoding="utf-8")  # noqa: SIM115
        should_close = True
    else:
        logger.debug("Writing well data to RMS ASCII stream")
        fwell_write = filepath  # type: ignore[assignment]
        should_close = False

    try:
        # Line 1: Version
        print("1.0", file=fwell_write)
        # Line 2: Description
        print("Unknown", file=fwell_write)
        # Line 3: Well header
        if well.zpos is None or well.zpos == 0.0:
            print(f"{well.name} {well.xpos} {well.ypos}", file=fwell_write)
        else:
            print(f"{well.name} {well.xpos} {well.ypos} {well.zpos}", file=fwell_write)
        # Line 4: Number of logs (not including X, Y, Z coordinates)
        print(f"{len(well.logs)}", file=fwell_write)

        # Log definitions (only for actual logs, not coordinates)
        for log in well.logs:
            if log.is_discrete:
                log_type = "DISC"
                wrec = "lin"
                if log.code_names and isinstance(log.code_names, dict):
                    code_parts: list[str] = []
                    for code, name in sorted(log.code_names.items()):
                        code_parts.extend([str(code), name])
                    wrec = " ".join(code_parts)
            else:
                # Continuous log: try to restore metadata from code_names tuple
                if isinstance(log.code_names, tuple) and len(log.code_names) >= 2:
                    log_type = str(log.code_names[0])
                    wrec = " ".join(str(x) for x in log.code_names[1:])
                else:
                    log_type = "UNK"
                    wrec = "lin"

            print(f"{log.name} {log_type} {wrec}", file=fwell_write)
    finally:
        if should_close:
            fwell_write.close()

    data = {
        "X_UTME": well.survey_x,
        "Y_UTMN": well.survey_y,
        "Z_TVDSS": well.survey_z,
    }

    for log in well.logs:
        data[log.name] = log.values

    tmpdf = pd.DataFrame(data).fillna(value=RMS_ASCII_UNDEF)

    # Convert discrete logs to integers
    for log in well.logs:
        if log.is_discrete:
            tmpdf[[log.name]] = tmpdf[[log.name]].fillna(RMS_ASCII_UNDEF).astype(int)

    cformat = f"%.{precision}f"

    # For streams, append directly; for files, use append mode
    # (append: preparing for writing multiple wells to one file)
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
            path_obj,
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
    filepath: FileLike,
) -> BlockedWellData:
    """Read blocked well data from RMS ASCII file.

    This assumes the file contains I_INDEX, J_INDEX, K_INDEX columns.

    Args:
        filepath: Path to RMS ASCII file

    Returns:
        BlockedWellData object

    """
    well = read_rms_ascii_well(filepath)

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
    filepath: FileLike,
    precision: int = 4,
) -> None:
    """Write blocked well data to RMS ASCII file.

    Args:
        blocked_well: BlockedWellData object to write
        filepath: Output RMS ASCII file path
        precision: Number of decimal places for floats (default: 4)

    """
    i_log = WellLog(name="I_INDEX", values=blocked_well.i_index, is_discrete=False)
    j_log = WellLog(name="J_INDEX", values=blocked_well.j_index, is_discrete=False)
    k_log = WellLog(name="K_INDEX", values=blocked_well.k_index, is_discrete=False)

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

    write_rms_ascii_well(temp_well, filepath, precision)

    logger.debug(
        "Successfully wrote blocked well '%s' with %d blocked cells to %s",
        blocked_well.name,
        blocked_well.n_blocked_cells,
        filepath,
    )
