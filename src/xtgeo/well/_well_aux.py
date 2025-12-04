"""Auxillary functions for the well class."""

from __future__ import annotations

from typing import TYPE_CHECKING

from xtgeo.common._xyz_enum import _AttrType
from xtgeo.common.log import null_logger

if TYPE_CHECKING:
    from xtgeo.io.welldata._well_io import WellData
    from xtgeo.well.well1 import Well

logger = null_logger(__name__)


def welldata_to_well_dict(
    welldata: WellData,
    mdlogname: str | None = None,
    zonelogname: str | None = None,
    lognames: str | list[str] | None = "all",
    lognames_strict: bool = False,
    strict: bool = False,
) -> dict:
    """Convert a WellData object to a dict suitable for Well class initialization.

    Args:
        welldata: WellData object from the new I/O system
        mdlogname: Name of measured depth log, if any
        zonelogname: Name of zonation log, if any
        lognames: Name or list of lognames to include, default is "all"
        lognames_strict: If True, all lognames must be present
        strict: If True, then import will fail if zonelogname or mdlogname are asked
            for but those names are not present in wells

    Returns:
        Dictionary with keys: rkb, xpos, ypos, wname, df, wlogtypes, wlogrecords,
        mdlogname, zonelogname, suitable for passing to Well.__init__()
    """
    # Convert WellData to pandas DataFrame
    df = welldata.to_dataframe()

    # Filter logs if requested
    if lognames != "all":
        coord_cols = ["X_UTME", "Y_UTMN", "Z_TVDSS"]
        if isinstance(lognames, str):
            use_cols = coord_cols + [lognames]
        elif isinstance(lognames, list):
            use_cols = coord_cols + lognames
        else:
            use_cols = coord_cols

        # Check if all requested logs exist
        missing = set(use_cols) - set(df.columns)
        if missing:
            if lognames_strict:
                raise ValueError(
                    f"Requested logs not found: {missing}. "
                    f"Available logs: {list(welldata.log_names)}"
                )
            # Only use the columns that exist
            use_cols = [col for col in use_cols if col in df.columns]

        df = df[use_cols]

    # Build wlogtypes and wlogrecords dicts
    wlogtypes = {}
    wlogrecords = {}

    for log in welldata.logs:
        # Only include logs that are in the filtered dataframe
        if log.name not in df.columns:
            continue

        if log.is_discrete:
            wlogtypes[log.name] = _AttrType.DISC.value
            if log.code_names:
                wlogrecords[log.name] = log.code_names
        else:
            wlogtypes[log.name] = _AttrType.CONT.value
            # For continuous logs, check if there's metadata stored
            # (this comes from the old format where continuous logs had tuples)
            if log.code_names and isinstance(log.code_names, (tuple, list)):
                wlogrecords[log.name] = tuple(log.code_names)
            # If code_names is a dict with string keys, convert to tuple
            elif log.code_names and isinstance(log.code_names, dict):
                # This is the metadata stored as dict, convert to tuple
                wlogrecords[log.name] = log.code_names

    # Check for mdlogname and zonelogname
    checked_mdlogname = mdlogname
    checked_zonelogname = zonelogname

    if mdlogname is not None and mdlogname not in df.columns:
        msg = (
            f"mdlogname={mdlogname} was requested but no such log found for "
            f"well {welldata.name}"
        )
        checked_mdlogname = None
        if strict:
            raise ValueError(msg)
        logger.warning(msg)

    if zonelogname is not None and zonelogname not in df.columns:
        msg = (
            f"zonelogname={zonelogname} was requested but no such log found "
            f"for well {welldata.name}"
        )
        checked_zonelogname = None
        if strict:
            raise ValueError(msg)
        logger.warning(msg)

    return {
        "rkb": welldata.zpos,
        "xpos": welldata.xpos,
        "ypos": welldata.ypos,
        "wname": welldata.name,
        "df": df,
        "wlogtypes": wlogtypes,
        "wlogrecords": wlogrecords,
        "mdlogname": checked_mdlogname,
        "zonelogname": checked_zonelogname,
    }


def well_to_welldata(well: Well) -> WellData:
    """Convert a Well instance to WellData format.

    Args:
        well: The Well instance to convert

    Returns:
        WellData instance
    """
    from xtgeo.io.welldata._well_io import WellData, WellLog

    # Get the dataframe
    df = well.get_dataframe(copy=False)

    # Extract survey data
    survey_x = df[well.xname].values
    survey_y = df[well.yname].values
    survey_z = df[well.zname].values

    # Create WellLog objects for each log (excluding X, Y, Z)
    logs = []
    for logname in well.lognames:  # This excludes X, Y, Z
        values = df[logname].values
        is_discrete = well.isdiscrete(logname)
        code_names = well.get_logrecord(logname) if is_discrete else None

        logs.append(
            WellLog(
                name=logname,
                values=values,
                is_discrete=is_discrete,
                code_names=code_names,
            )
        )

    return WellData(
        name=well.name,
        xpos=well.xpos,
        ypos=well.ypos,
        zpos=well.rkb,
        survey_x=survey_x,
        survey_y=survey_y,
        survey_z=survey_z,
        logs=tuple(logs),
    )
