# -*- coding: utf-8 -*-
"""PolylineSet Representation converter between xtgeo.Polygons and RESQML 2.0.1.

Handles bidirectional conversion:
  - RESQML PolylineSetRepresentation -> xtgeo.Polygons
  - xtgeo.Polygons -> RESQML PolylineSetRepresentation

Supports the full RESQML fault chain:
  BoundaryFeature -> FaultInterpretation -> PolylineSetRepresentation
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


def polylineset_to_xtgeo(
    provider: ResqmlDataProvider,
    polyline_uuid: str,
) -> Any:
    """Read a PolylineSet representation and convert to xtgeo.Polygons.

    If the PolylineSetRepresentation references a FaultInterpretation, the fault
    name (from BoundaryFeature) is stored on the Polygons object's name attribute
    and in RESQML metadata.

    Parameters
    ----------
    provider : ResqmlDataProvider
        An open data provider.
    polyline_uuid : str
        UUID of the PolylineSetRepresentation to read.

    Returns
    -------
    xtgeo.Polygons
    """
    import pandas as pd

    from xtgeo import Polygons

    data = provider.get_polylineset(polyline_uuid)
    polylines = data["polylines"]
    closed = data["closed"]

    # Build dataframe with polygon ID column (vectorized)
    chunks = []
    for poly_id, (poly, is_closed) in enumerate(zip(polylines, closed)):
        if poly.ndim == 1:
            poly = poly.reshape(-1, 3)
        if len(poly) == 0:
            continue
        ids = np.full(len(poly), poly_id, dtype=np.float64)
        chunk = np.column_stack([poly, ids])
        chunks.append(chunk)
        if is_closed:
            chunks.append(np.array([[poly[0, 0], poly[0, 1], poly[0, 2], poly_id]]))

    if chunks:
        df = pd.DataFrame(
            np.vstack(chunks), columns=["X_UTME", "Y_UTMN", "Z_TVDSS", "POLY_ID"]
        )
    else:
        df = pd.DataFrame(columns=["X_UTME", "Y_UTMN", "Z_TVDSS", "POLY_ID"])

    # Resolve fault name from interpretation chain
    fault_name = ""
    interpretation_uuid = data.get("interpretation_uuid", "")
    if interpretation_uuid:
        try:
            interp_data = provider.get_fault_interpretation(interpretation_uuid)
            fault_name = interp_data.get("feature_title", "") or interp_data.get(
                "title", ""
            )
        except Exception:
            logger.debug(
                "Could not resolve FaultInterpretation %s", interpretation_uuid
            )

    obj_title = data.get("title", "")
    polys = Polygons(values=df, name=fault_name or obj_title or "poly")

    # Attach RESQML provenance metadata
    _set_resqml_meta(
        polys,
        {
            "uuid": polyline_uuid,
            "schema_version": "2.0.1",
            "object_type": "PolylineSetRepresentation",
            "crs_uuid": data.get("crs_uuid", ""),
            "title": obj_title,
            "interpretation_uuid": interpretation_uuid,
            "fault_name": fault_name,
        },
    )

    return polys


def xtgeo_polygons_to_resqml(
    provider: ResqmlDataProvider,
    polygons: Any,
    title: str = "Exported Polygons",
    polyline_uuid: Optional[str] = None,
    crs_uuid: Optional[str] = None,
    crs_epsg: Optional[int] = None,
    fault_name: Optional[str] = None,
    line_role: Optional[str] = None,
) -> Dict[str, str]:
    """Write xtgeo.Polygons to a RESQML provider as PolylineSetRepresentation.

    If ``fault_name`` is provided, creates the full RESQML fault chain:
      BoundaryFeature(fault_name) -> FaultInterpretation -> PolylineSetRepresentation

    This matches what AspenTech and Landmark software produce when pushing fault
    polylines to RDDMS.

    Parameters
    ----------
    provider : ResqmlDataProvider
        An open data provider in write mode.
    polygons : xtgeo.Polygons
        The polygons to export.
    title : str
        Title for the RESQML PolylineSetRepresentation object.
    polyline_uuid : str, optional
        UUID for the polylineset. Auto-generated if not provided.
    crs_uuid : str, optional
        UUID of existing CRS.
    crs_epsg : int, optional
        EPSG code for projected CRS if creating default.
    fault_name : str, optional
        Name of the fault. If provided, creates BoundaryFeature +
        FaultInterpretation objects and links them to the representation.
        If not provided but the Polygons has a non-default name or was
        previously read from a fault interpretation, that name is used.
    line_role : str, optional
        RESQML LineRole value (e.g. "fault center line"). Defaults to
        "fault center line" when fault_name is set.

    Returns
    -------
    dict mapping object titles to their UUIDs.
    """
    # Try to recover UUIDs from metadata if available
    saved = _get_resqml_meta(polygons)

    if polyline_uuid is None:
        polyline_uuid = saved.get("uuid") or str(_uuid.uuid4())
    if crs_uuid is None:
        crs_uuid = saved.get("crs_uuid") or None

    # Resolve fault_name from Polygons metadata/name if not explicitly given
    if fault_name is None:
        fault_name = saved.get("fault_name", "")
        if not fault_name and polygons.name not in ("poly", ""):
            fault_name = polygons.name

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

    # Create fault interpretation chain if fault_name is set
    interpretation_uuid: Optional[str] = None
    if fault_name:
        feature_uuid = str(_uuid.uuid4())
        interpretation_uuid = str(_uuid.uuid4())

        provider._put_boundary_feature(feature_uuid, fault_name)
        provider._put_fault_interpretation(
            interpretation_uuid, fault_name, feature_uuid
        )

        result_uuids[f"BoundaryFeature:{fault_name}"] = feature_uuid
        result_uuids[f"FaultInterpretation:{fault_name}"] = interpretation_uuid

        # Default line_role for faults
        if line_role is None:
            line_role = "fault center line"

    # Extract polylines from xtgeo Polygons
    df = polygons.get_dataframe()
    polylines: List[np.ndarray] = []
    closed_list: List[bool] = []

    if "POLY_ID" in df.columns:
        for _, group in df.groupby("POLY_ID", sort=False):
            pts = group[["X_UTME", "Y_UTMN", "Z_TVDSS"]].values.astype(np.float64)

            # Detect if closed (first point == last point)
            is_closed = False
            if len(pts) > 2 and np.allclose(pts[0], pts[-1], atol=1e-10):
                pts = pts[:-1]  # Remove repeated closing point
                is_closed = True

            polylines.append(pts)
            closed_list.append(is_closed)
    else:
        # Single polyline
        pts = df[["X_UTME", "Y_UTMN", "Z_TVDSS"]].values.astype(np.float64)
        is_closed = False
        if len(pts) > 2 and np.allclose(pts[0], pts[-1], atol=1e-10):
            pts = pts[:-1]
            is_closed = True
        polylines.append(pts)
        closed_list.append(is_closed)

    provider.put_polylineset(
        uuid=polyline_uuid,
        title=title,
        polylines=polylines,
        closed=closed_list,
        crs_uuid=crs_uuid,
        interpretation_uuid=interpretation_uuid,
        line_role=line_role,
    )
    result_uuids[title] = polyline_uuid

    return result_uuids
