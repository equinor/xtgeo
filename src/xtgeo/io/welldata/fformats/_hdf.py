"""HDF5 I/O for well data."""

# Note: This experimental HDF implementation does not currently support memory streams

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from xtgeo.common.log import null_logger
from xtgeo.io.welldata._well_io import WellData, WellLog

logger = null_logger(__name__)


def read_hdf5_well(
    filepath: str | Path,
) -> WellData:
    """Read well data from HDF5 file.

    Args:
        filepath: Path to HDF5 file

    Returns:
        WellData object

    Example:
        >>> from xtgeo.io.welldata._io_hdf import read_hdf5_well
        >>> well = read_hdf5_well("well.h5")
        >>> print(f"Well: {well.name}, Records: {well.n_records}")
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    logger.info("Reading well data from HDF5: %s", filepath)

    with pd.HDFStore(filepath, "r") as store:
        data: pd.DataFrame = store.get("Well")  # type: ignore[assignment]
        wstore = store.get_storer("Well")  # type: ignore[operator]
        jmeta = wstore.attrs["metadata"]  # type: ignore[index,operator]

    if isinstance(jmeta, bytes):
        jmeta = jmeta.decode()

    meta = json.loads(jmeta, object_pairs_hook=dict)
    req = meta["_required_"]

    # Extract basic well information
    wname = req["name"]
    xpos = req["xpos"]
    ypos = req["ypos"]
    rkb = req.get("rkb", 0.0)
    zpos = rkb if rkb is not None else 0.0

    # Extract coordinates from dataframe
    survey_x = data["X_UTME"].to_numpy()
    survey_y = data["Y_UTMN"].to_numpy()
    survey_z = data["Z_TVDSS"].to_numpy()

    # Process well logs from metadata
    wlogs_meta = req.get("wlogs", {})
    logs = []

    for log_name in data.columns:
        if log_name in ["X_UTME", "Y_UTMN", "Z_TVDSS"]:
            continue

        values = data[log_name].to_numpy()

        # Get log type and records from metadata
        if log_name in wlogs_meta:
            log_type, log_records = wlogs_meta[log_name]
            is_discrete = log_type == "DISC"
            if is_discrete and log_records:
                # Convert string keys back to integers (JSON converts int keys to str)
                code_names: dict[int, Any] | None = {
                    int(k): v for k, v in log_records.items()
                }
            else:
                code_names = deepcopy(log_records) if log_records else None
        else:
            # Default to continuous log if not in metadata
            is_discrete = False
            code_names = None

        log = WellLog(
            name=log_name, values=values, is_discrete=is_discrete, code_names=code_names
        )
        logs.append(log)

    # Create WellData object
    well = WellData(
        name=wname,
        xpos=xpos,
        ypos=ypos,
        zpos=zpos,
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


def write_hdf5_well(
    well: WellData,
    filepath: str | Path,
    compression: Literal["lzf", "blosc"] | None = "lzf",
) -> None:
    """Write well data to HDF5 file.

    Args:
        well: WellData object to write
        filepath: Output HDF5 file path
        compression: Compression method. Options: "lzf" (default), "blosc", or None

    Example:
        >>> from xtgeo.io.welldata._io_hdf import write_hdf5_well
        >>> write_hdf5_well(well, "output.h5")
    """
    filepath = Path(filepath)

    logger.info("Writing well data to HDF5: %s", filepath)

    # Prepare metadata
    wlogs_meta = {}
    for log in well.logs:
        log_type = "DISC" if log.is_discrete else "CONT"
        log_records = deepcopy(log.code_names) if log.is_discrete else None
        wlogs_meta[log.name] = (log_type, log_records)

    metadata = {
        "_required_": {
            "name": well.name,
            "xpos": well.xpos,
            "ypos": well.ypos,
            "rkb": well.zpos,
            "wlogs": wlogs_meta,
        }
    }

    jmeta = json.dumps(metadata)

    # Prepare compression settings
    complib = "zlib"  # same as default lzf
    complevel = 5
    if compression and compression == "blosc":
        complib = "blosc"
    else:
        complevel = 0

    # Create DataFrame with all data
    data = {
        "X_UTME": well.survey_x,
        "Y_UTMN": well.survey_y,
        "Z_TVDSS": well.survey_z,
    }

    for log in well.logs:
        data[log.name] = log.values

    df = pd.DataFrame(data)

    # Write to HDF5
    complib_param: Literal["zlib", "lzo", "bzip2", "blosc"] | None = (
        "blosc" if complib == "blosc" else None
    )
    with pd.HDFStore(
        filepath, "w", complevel=complevel, complib=complib_param
    ) as store:
        store.put("Well", df)
        wstore_write = store.get_storer("Well")  # type: ignore[operator]
        wstore_write.attrs["metadata"] = jmeta  # type: ignore[index]
        wstore_write.attrs["provider"] = "xtgeo"  # type: ignore[index]
        wstore_write.attrs["format_idcode"] = 1401  # type: ignore[index]

    logger.debug(
        "Successfully wrote well '%s' with %d records to %s",
        well.name,
        well.n_records,
        filepath,
    )
