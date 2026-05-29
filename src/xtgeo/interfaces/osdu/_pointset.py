# -*- coding: utf-8 -*-
"""PointSet Representation converter between xtgeo.Points and RESQML 2.0.1.

Handles bidirectional conversion:
  - RESQML PointSetRepresentation -> xtgeo.Points
  - xtgeo.Points -> RESQML PointSetRepresentation
"""

from __future__ import annotations

import logging
import uuid as _uuid
from typing import TYPE_CHECKING, Any, Dict, Optional

import numpy as np

from ._resqml_meta import _get_resqml_meta, _set_resqml_meta

if TYPE_CHECKING:
    from ._provider_base import ResqmlDataProvider

logger = logging.getLogger(__name__)


def pointset_to_xtgeo(
    provider: ResqmlDataProvider,
    pointset_uuid: str,
) -> Any:
    """Read a PointSet representation and convert to xtgeo.Points.

    Parameters
    ----------
    provider : ResqmlDataProvider
        An open data provider.
    pointset_uuid : str
        UUID of the PointSetRepresentation to read.

    Returns
    -------
    xtgeo.Points
    """
    import pandas as pd

    from xtgeo import Points

    data = provider.get_pointset(pointset_uuid)
    points_arr = data["points"]

    if points_arr.size == 0:
        df = pd.DataFrame(columns=["X_UTME", "Y_UTMN", "Z_TVDSS"])
    else:
        if points_arr.ndim == 1:
            points_arr = points_arr.reshape(-1, 3)
        df = pd.DataFrame(points_arr, columns=["X_UTME", "Y_UTMN", "Z_TVDSS"])

    pts = Points(values=df)

    # Attach RESQML provenance metadata
    _set_resqml_meta(
        pts,
        {
            "uuid": pointset_uuid,
            "schema_version": "2.0.1",
            "object_type": "PointSetRepresentation",
            "crs_uuid": data.get("crs_uuid", ""),
            "title": data.get("title", ""),
        },
    )

    return pts


def xtgeo_points_to_resqml(
    provider: ResqmlDataProvider,
    points: Any,
    title: str = "Exported Points",
    pointset_uuid: Optional[str] = None,
    crs_uuid: Optional[str] = None,
    crs_epsg: Optional[int] = None,
) -> Dict[str, str]:
    """Write xtgeo.Points to a RESQML provider as PointSetRepresentation.

    Parameters
    ----------
    provider : ResqmlDataProvider
        An open data provider in write mode.
    points : xtgeo.Points
        The points to export.
    title : str
        Title for the RESQML object.
    pointset_uuid : str, optional
        UUID for the pointset. Auto-generated if not provided.
    crs_uuid : str, optional
        UUID of existing CRS.
    crs_epsg : int, optional
        EPSG code for projected CRS if creating default.

    Returns
    -------
    dict mapping object titles to their UUIDs.
    """
    # Try to recover UUIDs from metadata if available
    saved = _get_resqml_meta(points)

    if pointset_uuid is None:
        pointset_uuid = saved.get("uuid") or str(_uuid.uuid4())
    if crs_uuid is None:
        crs_uuid = saved.get("crs_uuid") or None

    result_uuids = {}

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

    # Extract XYZ from points dataframe
    df = points.get_dataframe()
    xyz = df[["X_UTME", "Y_UTMN", "Z_TVDSS"]].values.astype(np.float64)

    provider.put_pointset(
        uuid=pointset_uuid,
        title=title,
        points=xyz,
        crs_uuid=crs_uuid,
    )
    result_uuids[title] = pointset_uuid

    return result_uuids
