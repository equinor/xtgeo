# -*- coding: utf-8 -*-
"""IJK Grid Representation converter between xtgeo.Grid and RESQML 2.0.1.

Handles bidirectional conversion:
  - RESQML IjkGridRepresentation -> xtgeo.Grid
  - xtgeo.Grid -> RESQML IjkGridRepresentation

Supports both EPC file and ETP protocol backends via the provider abstraction.
"""

from __future__ import annotations

import logging
import uuid as _uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np

from ._metadata import resolve_property_mapping
from ._resqml_meta import _get_resqml_meta, _set_resqml_meta

if TYPE_CHECKING:
    from ._provider_base import ResqmlDataProvider

logger = logging.getLogger(__name__)


def ijk_grid_to_xtgeo(
    provider: ResqmlDataProvider,
    grid_uuid: str,
    load_properties: bool = True,
) -> Tuple[Any, List[Any]]:
    """Read an IJK Grid from a RESQML provider and convert to xtgeo.Grid.

    Parameters
    ----------
    provider : ResqmlDataProvider
        An open data provider (EPC file or ETP connection).
    grid_uuid : str
        UUID of the IjkGridRepresentation to read.
    load_properties : bool
        If True, also discover and load associated grid properties.

    Returns
    -------
    tuple of (xtgeo.Grid, list of xtgeo.GridProperty)
    """
    from xtgeo import Grid, GridProperty

    geom = provider.get_ijk_grid_geometry(grid_uuid)

    ni = geom["ni"]
    nj = geom["nj"]
    nk = geom["nk"]
    coord = geom["coord"]
    zcorn = geom["zcorn"]
    actnum = geom["actnum"]

    if coord is None or zcorn is None:
        raise ValueError(f"Grid {grid_uuid}: missing geometry arrays (coord/zcorn)")

    # Reshape arrays to xtgeo expected formats:
    #   coordsv: (ni+1, nj+1, 6)  — pillar top/bottom XYZ
    #   zcornsv: (ni+1, nj+1, nk+1, 4) — corner Z at each node
    #   actnumsv: (ni, nj, nk)
    try:
        coordsv = coord.reshape((ni + 1, nj + 1, 6)).astype(np.float64)
    except ValueError:
        raise ValueError(
            f"Grid {grid_uuid}: coord array size {coord.size} incompatible "
            f"with expected (ni+1)*(nj+1)*6 = {(ni + 1) * (nj + 1) * 6}"
        )

    try:
        zcornsv = zcorn.reshape((ni + 1, nj + 1, nk + 1, 4)).astype(np.float32)
    except ValueError:
        raise ValueError(
            f"Grid {grid_uuid}: zcorn array size {zcorn.size} incompatible "
            f"with expected (ni+1)*(nj+1)*(nk+1)*4 = "
            f"{(ni + 1) * (nj + 1) * (nk + 1) * 4}"
        )

    try:
        actnumsv = actnum.reshape((ni, nj, nk)).astype(np.int32)
    except ValueError:
        # Fallback: ones if size mismatch
        logger.warning(
            "actnum array size %d != expected %d; using all-active",
            actnum.size,
            ni * nj * nk,
        )
        actnumsv = np.ones((ni, nj, nk), dtype=np.int32)

    grid = Grid(coordsv, zcornsv, actnumsv)

    # Attach RESQML provenance metadata
    _set_resqml_meta(
        grid,
        {
            "uuid": grid_uuid,
            "schema_version": "2.0.1",
            "object_type": "IjkGridRepresentation",
            "crs_uuid": geom.get("crs_uuid", ""),
            "k_direction": geom.get("k_direction", "down"),
            "title": geom.get("title", ""),
        },
    )

    # Load properties if requested
    properties = []
    if load_properties:
        all_objects = provider.list_objects("Property")
        for obj in all_objects:
            prop_uuid = obj.get("uuid", "")
            if not prop_uuid:
                continue
            try:
                # Determine the object type from the listing
                obj_type = obj.get("type", "resqml20.ContinuousProperty")
                # Extract local name e.g. "DiscreteProperty"
                # from "resqml20.DiscreteProperty"
                obj_type_short = (
                    obj_type.split(".")[-1] if "." in obj_type else obj_type
                )
                prop_data = provider.get_property_values(
                    prop_uuid, object_type=obj_type_short
                )
                # Check if this property belongs to our grid
                if prop_data["supporting_representation_uuid"] != grid_uuid:
                    continue

                values = prop_data["values"]
                if values.size != ni * nj * nk:
                    continue

                name = prop_data["title"]
                is_discrete = prop_data["is_discrete"]

                if is_discrete:
                    grid_prop = GridProperty(
                        ncol=ni,
                        nrow=nj,
                        nlay=nk,
                        name=name,
                        discrete=True,
                        values=values.reshape((ni, nj, nk)).astype(np.int32),
                    )
                else:
                    grid_prop = GridProperty(
                        ncol=ni,
                        nrow=nj,
                        nlay=nk,
                        name=name,
                        discrete=False,
                        values=values.reshape((ni, nj, nk)).astype(np.float64),
                    )

                # Attach RESQML provenance to the property
                mapping = resolve_property_mapping(title=name)
                _set_resqml_meta(
                    grid_prop,
                    {
                        "uuid": prop_uuid,
                        "schema_version": "2.0.1",
                        "object_type": obj_type_short,
                        "supporting_representation_uuid": grid_uuid,
                        "property_kind": mapping.osdu_name if mapping else name,
                        "osdu_reference": mapping.osdu_reference if mapping else None,
                        "uom": mapping.uom_family if mapping else "",
                        "is_discrete": is_discrete,
                        "title": name,
                    },
                )
                properties.append(grid_prop)
            except Exception as e:
                logger.debug("Skipping property %s: %s", prop_uuid, e)

    return grid, properties


def xtgeo_grid_to_resqml(
    provider: ResqmlDataProvider,
    grid: Any,
    title: str = "Exported Grid",
    grid_uuid: Optional[str] = None,
    crs_uuid: Optional[str] = None,
    crs_epsg: Optional[int] = None,
    properties: Optional[List[Any]] = None,
) -> Dict[str, str]:
    """Write an xtgeo.Grid to a RESQML provider.

    Parameters
    ----------
    provider : ResqmlDataProvider
        An open data provider (EPC file or ETP connection) in write mode.
    grid : xtgeo.Grid
        The grid to export.
    title : str
        Title for the RESQML object.
    grid_uuid : str, optional
        UUID for the grid. Auto-generated if not provided.
    crs_uuid : str, optional
        UUID of existing CRS. If None, a default CRS is created.
    crs_epsg : int, optional
        EPSG code for the projected CRS (used when creating default CRS).
    properties : list of xtgeo.GridProperty, optional
        Properties to export alongside the grid.

    Returns
    -------
    dict mapping object titles to their UUIDs.
    """
    # Try to recover UUIDs from metadata if available
    saved = _get_resqml_meta(grid)

    if grid_uuid is None:
        grid_uuid = saved.get("uuid") or str(_uuid.uuid4())
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

    # Extract geometry from xtgeo grid
    ni = grid.ncol
    nj = grid.nrow
    nk = grid.nlay

    # Get raw arrays from grid
    coordsv = grid._coordsv.copy()
    zcornsv = grid._zcornsv.copy()
    actnumsv = grid._actnumsv.copy()

    provider.put_ijk_grid_geometry(
        uuid=grid_uuid,
        title=title,
        ni=ni,
        nj=nj,
        nk=nk,
        coord=coordsv,
        zcorn=zcornsv,
        actnum=actnumsv,
        crs_uuid=crs_uuid,
        k_direction="down",
    )
    result_uuids[title] = grid_uuid

    # Export properties
    if properties:
        for prop in properties:
            # Recover UUID from property metadata if available
            prop_saved = _get_resqml_meta(prop)
            prop_uuid = prop_saved.get("uuid") or str(_uuid.uuid4())
            values = prop.values.flatten()

            # Determine property kind and mapping
            mapping = resolve_property_mapping(title=prop.name)
            if mapping:
                property_kind = mapping.osdu_name or prop.name
                uom = mapping.uom_family
            else:
                property_kind = prop.name
                uom = ""

            provider.put_property_values(
                uuid=prop_uuid,
                title=prop.name,
                values=values,
                supporting_representation_uuid=grid_uuid,
                property_kind=property_kind,
                indexable_element="cells",
                is_discrete=bool(prop.isdiscrete),
                uom=uom,
            )
            result_uuids[prop.name] = prop_uuid

    return result_uuids
