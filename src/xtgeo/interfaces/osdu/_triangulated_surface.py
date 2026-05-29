# -*- coding: utf-8 -*-
"""TriangulatedSetRepresentation converter between xtgeo and RESQML 2.0.1.

Handles bidirectional conversion:
  - RESQML TriangulatedSetRepresentation -> xtgeo.TriangulatedSurface
  - xtgeo.TriangulatedSurface -> RESQML TriangulatedSetRepresentation

RESQML TriangulatedSet data model:
  - NodeCount: number of vertices
  - TriangleCount: number of triangles
  - Points: (N, 3) vertex coordinates
  - Triangles: (M, 3) 0-based vertex indices per face
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


def triangulated_surface_to_xtgeo(
    provider: ResqmlDataProvider,
    trisurface_uuid: str,
) -> Any:
    """Read a TriangulatedSetRepresentation and convert to xtgeo.TriangulatedSurface.

    Parameters
    ----------
    provider : ResqmlDataProvider
        An open data provider.
    trisurface_uuid : str
        UUID of the TriangulatedSetRepresentation to read.

    Returns
    -------
    xtgeo.TriangulatedSurface
    """
    from xtgeo.surface.triangulated_surface import TriangulatedSurface

    data = provider.get_triangulated_set(trisurface_uuid)
    vertices = data["vertices"]
    triangles = data["triangles"]
    title = data.get("title", "")
    crs_uuid = data.get("crs_uuid", "")

    if vertices.ndim == 1:
        vertices = vertices.reshape(-1, 3)
    if triangles.ndim == 1:
        triangles = triangles.reshape(-1, 3)

    trisurf = TriangulatedSurface(
        vertices=vertices.astype(np.float64),
        triangles=triangles.astype(np.int_),
        name=title or "Unknown",
    )

    # Attach RESQML provenance metadata
    _set_resqml_meta(
        trisurf,
        {
            "uuid": trisurface_uuid,
            "schema_version": "2.0.1",
            "object_type": "TriangulatedSetRepresentation",
            "crs_uuid": crs_uuid,
            "title": title,
        },
    )

    return trisurf


def xtgeo_triangulated_surface_to_resqml(
    provider: ResqmlDataProvider,
    trisurf: Any,
    title: str = "Exported TriangulatedSurface",
    trisurface_uuid: Optional[str] = None,
    crs_uuid: Optional[str] = None,
    crs_epsg: Optional[int] = None,
) -> Dict[str, str]:
    """Write xtgeo.TriangulatedSurface to RESQML as TriangulatedSetRepresentation.

    Parameters
    ----------
    provider : ResqmlDataProvider
        An open data provider in write mode.
    trisurf : xtgeo.TriangulatedSurface
        The triangulated surface to export.
    title : str
        Title for the RESQML object.
    trisurface_uuid : str, optional
        UUID for the object. Auto-generated if not provided.
    crs_uuid : str, optional
        UUID of existing CRS.
    crs_epsg : int, optional
        EPSG code for projected CRS if creating default.

    Returns
    -------
    dict mapping object titles to their UUIDs.
    """
    saved = _get_resqml_meta(trisurf)

    if trisurface_uuid is None:
        trisurface_uuid = saved.get("uuid") or str(_uuid.uuid4())
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

    vertices = np.asarray(trisurf.vertices, dtype=np.float64)
    triangles = np.asarray(trisurf.triangles, dtype=np.int32)

    provider.put_triangulated_set(
        uuid=trisurface_uuid,
        title=title,
        vertices=vertices,
        triangles=triangles,
        crs_uuid=crs_uuid,
    )
    result_uuids[title] = trisurface_uuid

    return result_uuids
