# -*- coding: utf-8 -*-
"""Well Representation converter between xtgeo.Well and RESQML 2.0.1.

Handles bidirectional conversion:
  - RESQML WellboreTrajectoryRepresentation + WellboreFrameRepresentation -> xtgeo.Well
  - xtgeo.Well -> RESQML WellboreTrajectoryRepresentation + WellboreFrameRepresentation

RESQML Well data model:
  - WellboreTrajectoryRepresentation: MD, X, Y, Z arrays (deviation survey)
  - WellboreFrameRepresentation: MD stations where logs are sampled
  - ContinuousProperty/DiscreteProperty: log values attached to frame
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


def well_to_xtgeo(
    provider: ResqmlDataProvider,
    trajectory_uuid: str,
    *,
    load_logs: bool = True,
) -> Any:
    """Read a WellboreTrajectory and associated logs, convert to xtgeo.Well.

    Parameters
    ----------
    provider : ResqmlDataProvider
        An open data provider.
    trajectory_uuid : str
        UUID of the WellboreTrajectoryRepresentation to read.
    load_logs : bool
        If True, also load WellboreFrame logs.

    Returns
    -------
    xtgeo.Well
    """
    import pandas as pd

    from xtgeo import Well

    data = provider.get_wellbore_trajectory(trajectory_uuid)
    md = data["md"]
    xyz = data["xyz"]  # (N, 3)
    title = data.get("title", "")
    crs_uuid = data.get("crs_uuid", "")

    if xyz.ndim == 1:
        xyz = xyz.reshape(-1, 3)

    # Build base dataframe with trajectory
    df = pd.DataFrame(
        {
            "X_UTME": xyz[:, 0],
            "Y_UTMN": xyz[:, 1],
            "Z_TVDSS": xyz[:, 2],
        }
    )

    # Add MD column if available
    mdlogname = None
    if md is not None and len(md) == len(xyz):
        df["M_MDEPTH"] = md
        mdlogname = "M_MDEPTH"

    wlogtypes: Dict[str, str] = {}
    wlogrecords: Dict[str, Any] = {}

    # Load logs from associated WellboreFrame(s)
    if load_logs:
        frame_data = data.get("frames", [])
        for frame in frame_data:
            frame_md = frame.get("md")
            props = frame.get("properties", [])
            for prop in props:
                logname = prop["title"]
                values = prop["values"]
                is_discrete = prop.get("is_discrete", False)

                # Interpolate log values to trajectory MD if frames differ
                if frame_md is not None and md is not None and len(frame_md) != len(md):
                    values = np.interp(md, frame_md, values, left=np.nan, right=np.nan)
                elif len(values) != len(df):
                    # Pad or truncate
                    padded = np.full(len(df), np.nan)
                    n = min(len(values), len(df))
                    padded[:n] = values[:n]
                    values = padded

                df[logname] = values.astype(np.float64)
                wlogtypes[logname] = "DISC" if is_discrete else "CONT"
                if is_discrete:
                    wlogrecords[logname] = {}

    # Determine well head position (first point)
    xpos = float(xyz[0, 0]) if len(xyz) > 0 else 0.0
    ypos = float(xyz[0, 1]) if len(xyz) > 0 else 0.0

    well = Well(
        rkb=0.0,
        xpos=xpos,
        ypos=ypos,
        wname=title or "Unknown",
        df=df,
        mdlogname=mdlogname,
        wlogtypes=wlogtypes,
        wlogrecords=wlogrecords,
    )

    # Attach RESQML provenance metadata
    _set_resqml_meta(
        well,
        {
            "uuid": trajectory_uuid,
            "schema_version": "2.0.1",
            "object_type": "WellboreTrajectoryRepresentation",
            "crs_uuid": crs_uuid,
            "title": title,
        },
    )

    return well


def xtgeo_well_to_resqml(
    provider: ResqmlDataProvider,
    well: Any,
    title: Optional[str] = None,
    trajectory_uuid: Optional[str] = None,
    crs_uuid: Optional[str] = None,
    crs_epsg: Optional[int] = None,
    export_logs: bool = True,
) -> Dict[str, str]:
    """Write xtgeo.Well to RESQML as WellboreTrajectory + Frame + Properties.

    Parameters
    ----------
    provider : ResqmlDataProvider
        An open data provider in write mode.
    well : xtgeo.Well
        The well to export.
    title : str, optional
        Title for the RESQML object. Uses well.wellname if not given.
    trajectory_uuid : str, optional
        UUID for the trajectory. Auto-generated if not provided.
    crs_uuid : str, optional
        UUID of existing CRS.
    crs_epsg : int, optional
        EPSG code for projected CRS if creating default.
    export_logs : bool
        If True, also export well logs as properties on a WellboreFrame.

    Returns
    -------
    dict mapping object titles to their UUIDs.
    """
    saved = _get_resqml_meta(well)

    if title is None:
        title = well.wellname or "Exported Well"
    if trajectory_uuid is None:
        trajectory_uuid = saved.get("uuid") or str(_uuid.uuid4())
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

    # Extract trajectory data
    df = well.get_dataframe()
    xyz = df[["X_UTME", "Y_UTMN", "Z_TVDSS"]].values.astype(np.float64)

    # Get MD
    md = None
    if well.mdlogname and well.mdlogname in df.columns:
        md = df[well.mdlogname].values.astype(np.float64)
    else:
        # Compute MD from cumulative distance
        diffs = np.diff(xyz, axis=0)
        segment_lengths = np.sqrt(np.sum(diffs**2, axis=1))
        md = np.zeros(len(xyz))
        md[1:] = np.cumsum(segment_lengths)

    # Write trajectory
    provider.put_wellbore_trajectory(
        uuid=trajectory_uuid,
        title=title,
        md=md,
        xyz=xyz,
        crs_uuid=crs_uuid,
    )
    result_uuids[title] = trajectory_uuid

    # Write logs as WellboreFrame + properties
    if export_logs:
        log_cols = [
            c
            for c in df.columns
            if c not in ["X_UTME", "Y_UTMN", "Z_TVDSS"]
            and c != well.mdlogname
        ]
        if log_cols:
            frame_uuid = str(_uuid.uuid4())
            frame_title = f"{title} Frame"

            properties: List[Dict[str, Any]] = []
            for logname in log_cols:
                values = df[logname].values.astype(np.float64)
                is_discrete = well.wlogtypes.get(logname, "CONT") == "DISC"
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

            provider.put_wellbore_frame(
                uuid=frame_uuid,
                title=frame_title,
                trajectory_uuid=trajectory_uuid,
                md=md,
                properties=properties,
                crs_uuid=crs_uuid,
            )
            result_uuids[frame_title] = frame_uuid

    return result_uuids
