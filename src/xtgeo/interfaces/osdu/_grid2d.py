# -*- coding: utf-8 -*-
"""Grid2D Representation converter between xtgeo.RegularSurface and RESQML 2.0.1.

Handles bidirectional conversion:
  - RESQML Grid2dRepresentation -> xtgeo.RegularSurface
  - xtgeo.RegularSurface -> RESQML Grid2dRepresentation

Supports the full RESQML horizon chain:
  GeneticBoundaryFeature -> HorizonInterpretation -> Grid2dRepresentation
"""

from __future__ import annotations

import logging
import math
import uuid as _uuid
from typing import TYPE_CHECKING, Any, Dict, Optional

import numpy as np

from ._resqml_meta import _get_resqml_meta, _set_resqml_meta

if TYPE_CHECKING:
    from ._provider_base import ResqmlDataProvider

logger = logging.getLogger(__name__)


def grid2d_to_xtgeo(
    provider: ResqmlDataProvider,
    surface_uuid: str,
) -> Any:
    """Read a Grid2D representation and convert to xtgeo.RegularSurface.

    If the Grid2dRepresentation references a HorizonInterpretation, the horizon
    name (from GeneticBoundaryFeature) is stored on the surface's name attribute
    and in RESQML metadata.

    Parameters
    ----------
    provider : ResqmlDataProvider
        An open data provider.
    surface_uuid : str
        UUID of the Grid2dRepresentation to read.

    Returns
    -------
    xtgeo.RegularSurface
    """
    from xtgeo import RegularSurface

    geom = provider.get_grid2d_geometry(surface_uuid)

    ncol = geom["ni"]
    nrow = geom["nj"]
    xori = geom["origin_x"]
    yori = geom["origin_y"]
    xinc = geom["di"]
    yinc = geom["dj"]
    rotation = math.degrees(geom.get("rotation", 0.0))
    values = geom["values"]

    if values.shape != (nrow, ncol):
        # Try to reshape
        if values.size == nrow * ncol:
            values = values.reshape((nrow, ncol))
        else:
            logger.warning(
                "Surface values shape %s doesn't match grid dims (%d, %d)",
                values.shape,
                nrow,
                ncol,
            )

    # xtgeo RegularSurface uses masked arrays
    masked_values = np.ma.masked_invalid(values.astype(np.float64))

    # Resolve horizon name from interpretation chain
    horizon_name = ""
    interpretation_uuid = geom.get("interpretation_uuid", "")
    if interpretation_uuid:
        try:
            interp_data = provider.get_horizon_interpretation(interpretation_uuid)
            horizon_name = interp_data.get("feature_title", "") or interp_data.get(
                "title", ""
            )
        except Exception:
            logger.debug(
                "Could not resolve HorizonInterpretation %s", interpretation_uuid
            )

    obj_title = geom.get("title", "")

    surf = RegularSurface(
        ncol=ncol,
        nrow=nrow,
        xori=xori,
        yori=yori,
        xinc=xinc,
        yinc=yinc,
        rotation=rotation,
        values=masked_values,
        name=horizon_name or obj_title or "unknown",
    )

    # Attach RESQML provenance metadata
    _set_resqml_meta(
        surf,
        {
            "uuid": surface_uuid,
            "schema_version": "2.0.1",
            "object_type": "Grid2dRepresentation",
            "crs_uuid": geom.get("crs_uuid", ""),
            "title": obj_title,
            "interpretation_uuid": interpretation_uuid,
            "horizon_name": horizon_name,
        },
    )

    return surf


def xtgeo_surface_to_resqml(
    provider: ResqmlDataProvider,
    surface: Any,
    title: str = "Exported Surface",
    surface_uuid: Optional[str] = None,
    crs_uuid: Optional[str] = None,
    crs_epsg: Optional[int] = None,
    horizon_name: Optional[str] = None,
) -> Dict[str, str]:
    """Write an xtgeo.RegularSurface to a RESQML provider as Grid2dRepresentation.

    If ``horizon_name`` is provided, creates the full RESQML horizon
    chain: GeneticBoundaryFeature -> HorizonInterpretation -> Grid2d.

    Parameters
    ----------
    provider : ResqmlDataProvider
        An open data provider in write mode.
    surface : xtgeo.RegularSurface
        The surface to export.
    title : str
        Title for the RESQML object.
    surface_uuid : str, optional
        UUID for the surface. Auto-generated if not provided.
    crs_uuid : str, optional
        UUID of existing CRS. If None, a default CRS is created.
    crs_epsg : int, optional
        EPSG code for projected CRS if creating default.
    horizon_name : str, optional
        Name of the horizon. If provided, creates GeneticBoundaryFeature +
        HorizonInterpretation objects and links them to the representation.
        If not provided but the surface has a non-default name or was
        previously read from a horizon interpretation, that name is used.

    Returns
    -------
    dict mapping object titles to their UUIDs.
    """
    # Try to recover UUIDs from metadata if available
    saved = _get_resqml_meta(surface)

    if surface_uuid is None:
        surface_uuid = saved.get("uuid") or str(_uuid.uuid4())
    if crs_uuid is None:
        crs_uuid = saved.get("crs_uuid") or None

    # Resolve horizon_name from surface metadata/name if not explicitly given
    if horizon_name is None:
        horizon_name = saved.get("horizon_name", "")
        if not horizon_name and surface.name not in ("unknown", ""):
            horizon_name = surface.name

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

    # Create horizon interpretation chain if horizon_name is set
    interpretation_uuid: Optional[str] = None
    if horizon_name:
        feature_uuid = str(_uuid.uuid4())
        interpretation_uuid = str(_uuid.uuid4())

        provider._put_genetic_boundary_feature(feature_uuid, horizon_name)
        provider._put_horizon_interpretation(
            interpretation_uuid, horizon_name, feature_uuid
        )

        result_uuids[f"GeneticBoundaryFeature:{horizon_name}"] = feature_uuid
        result_uuids[f"HorizonInterpretation:{horizon_name}"] = interpretation_uuid

    # Extract values (fill masked with NaN for HDF5 storage)
    values = surface.values.filled(np.nan).astype(np.float64)
    rotation_rad = math.radians(surface.rotation)

    provider.put_grid2d_geometry(
        uuid=surface_uuid,
        title=title,
        ni=surface.ncol,
        nj=surface.nrow,
        origin_x=surface.xori,
        origin_y=surface.yori,
        di=surface.xinc,
        dj=surface.yinc,
        rotation=rotation_rad,
        values=values,
        crs_uuid=crs_uuid,
        interpretation_uuid=interpretation_uuid,
    )
    result_uuids[title] = surface_uuid

    return result_uuids
