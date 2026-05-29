# -*- coding: utf-8 -*-
"""BlockedWellbore Representation converter between xtgeo.BlockedWell and RESQML 2.0.1.

Handles bidirectional conversion:
  - RESQML BlockedWellboreRepresentation -> xtgeo.BlockedWell
  - xtgeo.BlockedWell -> RESQML BlockedWellboreRepresentation

RESQML BlockedWellbore data model:
  - BlockedWellboreRepresentation: Well trajectory blocked to grid cells
  - References a WellboreTrajectoryRepresentation and an IjkGridRepresentation
  - Cell indices (I, J, K) + entry/exit points per cell block
"""

from __future__ import annotations

import logging
import uuid as _uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np

from ._resqml_meta import _get_resqml_meta, _set_resqml_meta

if TYPE_CHECKING:
    from ._provider_base import ResqmlDataProvider

logger = logging.getLogger(__name__)


def blocked_well_to_xtgeo(
    provider: ResqmlDataProvider,
    blocked_well_uuid: str,
) -> Any:
    """Read a BlockedWellboreRepresentation and convert to xtgeo.BlockedWell.

    Parameters
    ----------
    provider : ResqmlDataProvider
        An open data provider.
    blocked_well_uuid : str
        UUID of the BlockedWellboreRepresentation to read.

    Returns
    -------
    xtgeo.BlockedWell
    """
    import pandas as pd

    from xtgeo import BlockedWell

    data = provider.get_blocked_wellbore(blocked_well_uuid)
    md = data["md"]
    xyz = data["xyz"]  # (N, 3)
    cell_indices = data["cell_indices"]  # (N, 3) — I, J, K
    title = data.get("title", "")
    crs_uuid = data.get("crs_uuid", "")
    grid_uuid = data.get("grid_uuid", "")
    trajectory_uuid = data.get("trajectory_uuid", "")

    if xyz.ndim == 1:
        xyz = xyz.reshape(-1, 3)
    if cell_indices.ndim == 1:
        cell_indices = cell_indices.reshape(-1, 3)

    # Build dataframe
    df = pd.DataFrame(
        {
            "X_UTME": xyz[:, 0],
            "Y_UTMN": xyz[:, 1],
            "Z_TVDSS": xyz[:, 2],
            "I_INDEX": cell_indices[:, 0].astype(np.float64),
            "J_INDEX": cell_indices[:, 1].astype(np.float64),
            "K_INDEX": cell_indices[:, 2].astype(np.float64),
        }
    )

    # Add MD if available
    mdlogname = None
    if md is not None and len(md) == len(xyz):
        df["M_MDEPTH"] = md
        mdlogname = "M_MDEPTH"

    wlogtypes: Dict[str, str] = {
        "I_INDEX": "CONT",
        "J_INDEX": "CONT",
        "K_INDEX": "CONT",
    }
    wlogrecords: Dict[str, Any] = {}

    # Load associated properties
    for prop in data.get("properties", []):
        logname = prop["title"]
        values = prop["values"]
        is_discrete = prop.get("is_discrete", False)

        if len(values) != len(df):
            padded = np.full(len(df), np.nan)
            n = min(len(values), len(df))
            padded[:n] = values[:n]
            values = padded

        df[logname] = values.astype(np.float64)
        wlogtypes[logname] = "DISC" if is_discrete else "CONT"
        if is_discrete:
            wlogrecords[logname] = {}

    # Determine well head position
    xpos = float(xyz[0, 0]) if len(xyz) > 0 else 0.0
    ypos = float(xyz[0, 1]) if len(xyz) > 0 else 0.0

    bwell = BlockedWell(
        rkb=0.0,
        xpos=xpos,
        ypos=ypos,
        wname=title or "Unknown",
        df=df,
        mdlogname=mdlogname,
        wlogtypes=wlogtypes,
        wlogrecords=wlogrecords,
    )
    bwell._gridname = grid_uuid  # Store grid UUID as gridname for reference

    # Attach RESQML provenance metadata
    _set_resqml_meta(
        bwell,
        {
            "uuid": blocked_well_uuid,
            "schema_version": "2.0.1",
            "object_type": "BlockedWellboreRepresentation",
            "crs_uuid": crs_uuid,
            "title": title,
            "grid_uuid": grid_uuid,
            "trajectory_uuid": trajectory_uuid,
        },
    )

    return bwell


def xtgeo_blocked_well_to_resqml(
    provider: ResqmlDataProvider,
    blocked_well: Any,
    title: Optional[str] = None,
    blocked_well_uuid: Optional[str] = None,
    trajectory_uuid: Optional[str] = None,
    grid_uuid: Optional[str] = None,
    crs_uuid: Optional[str] = None,
    crs_epsg: Optional[int] = None,
) -> Dict[str, str]:
    """Write xtgeo.BlockedWell to RESQML as BlockedWellboreRepresentation.

    Parameters
    ----------
    provider : ResqmlDataProvider
        An open data provider in write mode.
    blocked_well : xtgeo.BlockedWell
        The blocked well to export.
    title : str, optional
        Title for the RESQML object. Uses well name if not given.
    blocked_well_uuid : str, optional
        UUID for the blocked well. Auto-generated if not provided.
    trajectory_uuid : str, optional
        UUID of the trajectory this blocked well references.
        If not given, a trajectory will be created.
    grid_uuid : str, optional
        UUID of the grid this blocked well is associated with.
    crs_uuid : str, optional
        UUID of existing CRS.
    crs_epsg : int, optional
        EPSG code for projected CRS if creating default.

    Returns
    -------
    dict mapping object titles to their UUIDs.
    """
    saved = _get_resqml_meta(blocked_well)

    if title is None:
        title = blocked_well.wellname or "Exported BlockedWell"
    if blocked_well_uuid is None:
        blocked_well_uuid = saved.get("uuid") or str(_uuid.uuid4())
    if trajectory_uuid is None:
        trajectory_uuid = saved.get("trajectory_uuid") or str(_uuid.uuid4())
    if grid_uuid is None:
        grid_uuid = saved.get("grid_uuid") or ""
    if crs_uuid is None:
        crs_uuid = saved.get("crs_uuid") or None

    result_uuids: Dict[str, str] = {}

    # Create CRS if needed
    if crs_uuid is None:
        crs_uuid = str(_uuid.uuid4())
        provider.put_crs(
            uuid=crs_uuid,
            title="Default CRS",
            origin_x=0.0,
            origin_y=0.0,
            origin_z=0.0,
            areal_rotation=0.0,
            z_increasing_downward=True,
            projected_crs_epsg=crs_epsg,
        )
        result_uuids["CRS"] = crs_uuid

    # Extract data
    df = blocked_well.get_dataframe()
    xyz = df[["X_UTME", "Y_UTMN", "Z_TVDSS"]].values.astype(np.float64)

    # Cell indices
    cell_indices = np.column_stack(
        [
            df["I_INDEX"].values.astype(np.float64),
            df["J_INDEX"].values.astype(np.float64),
            df["K_INDEX"].values.astype(np.float64),
        ]
    )

    # Get MD
    md = None
    if blocked_well.mdlogname and blocked_well.mdlogname in df.columns:
        md = df[blocked_well.mdlogname].values.astype(np.float64)
    else:
        diffs = np.diff(xyz, axis=0)
        segment_lengths = np.sqrt(np.sum(diffs**2, axis=1))
        md = np.zeros(len(xyz))
        md[1:] = np.cumsum(segment_lengths)

    # First create the trajectory that the blocked well references
    provider.put_wellbore_trajectory(
        uuid=trajectory_uuid,
        title=f"{title} Trajectory",
        md=md,
        xyz=xyz,
        crs_uuid=crs_uuid,
    )
    result_uuids[f"{title} Trajectory"] = trajectory_uuid

    # Collect properties (logs beyond XYZ and indices)
    skip_cols = {"X_UTME", "Y_UTMN", "Z_TVDSS", "I_INDEX", "J_INDEX", "K_INDEX"}
    if blocked_well.mdlogname:
        skip_cols.add(blocked_well.mdlogname)

    properties: List[Dict[str, Any]] = []
    for logname in df.columns:
        if logname in skip_cols:
            continue
        values = df[logname].values.astype(np.float64)
        is_discrete = blocked_well.wlogtypes.get(logname, "CONT") == "DISC"
        prop_uuid = str(_uuid.uuid4())
        properties.append(
            {
                "uuid": prop_uuid,
                "title": logname,
                "values": values,
                "is_discrete": is_discrete,
                "property_kind": "discrete" if is_discrete else "continuous",
            }
        )
        result_uuids[logname] = prop_uuid

    # Write blocked wellbore
    provider.put_blocked_wellbore(
        uuid=blocked_well_uuid,
        title=title,
        trajectory_uuid=trajectory_uuid,
        grid_uuid=grid_uuid,
        md=md,
        xyz=xyz,
        cell_indices=cell_indices.astype(np.int32),
        properties=properties,
        crs_uuid=crs_uuid,
    )
    result_uuids[title] = blocked_well_uuid

    return result_uuids
