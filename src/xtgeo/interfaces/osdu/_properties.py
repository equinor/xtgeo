# -*- coding: utf-8 -*-
"""Grid property converter between xtgeo.GridProperty and RESQML 2.0.1.

Standalone property I/O (when not loaded with the grid).
"""

from __future__ import annotations

import logging
import uuid as _uuid
from typing import TYPE_CHECKING, Any, List, Optional

import numpy as np

from ._metadata import resolve_property_mapping

if TYPE_CHECKING:
    from ._provider_base import ResqmlDataProvider

logger = logging.getLogger(__name__)


def read_grid_properties(
    provider: ResqmlDataProvider,
    grid_uuid: str,
    ni: int,
    nj: int,
    nk: int,
) -> List[Any]:
    """Read all properties associated with a grid.

    Parameters
    ----------
    provider : ResqmlDataProvider
        An open data provider.
    grid_uuid : str
        UUID of the supporting grid representation.
    ni, nj, nk : int
        Grid dimensions for reshape validation.

    Returns
    -------
    list of xtgeo.GridProperty
    """
    from xtgeo import GridProperty

    properties = []
    all_objects = provider.list_objects("Property")

    for obj in all_objects:
        prop_uuid = obj.get("uuid", "")
        if not prop_uuid:
            continue
        try:
            prop_data = provider.get_property_values(prop_uuid)
            if prop_data["supporting_representation_uuid"] != grid_uuid:
                continue

            values = prop_data["values"]
            if values.size != ni * nj * nk:
                logger.debug(
                    "Property %s size %d != grid size %d, skipping",
                    prop_uuid,
                    values.size,
                    ni * nj * nk,
                )
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
            properties.append(grid_prop)
        except Exception as e:
            logger.debug("Skipping property %s: %s", prop_uuid, e)

    return properties


def write_grid_property(
    provider: ResqmlDataProvider,
    grid_property: Any,
    grid_uuid: str,
    prop_uuid: Optional[str] = None,
) -> str:
    """Write a single xtgeo.GridProperty to a RESQML provider.

    Parameters
    ----------
    provider : ResqmlDataProvider
        An open data provider in write mode.
    grid_property : xtgeo.GridProperty
        The property to export.
    grid_uuid : str
        UUID of the supporting grid representation.
    prop_uuid : str, optional
        UUID for the property. Auto-generated if not provided.

    Returns
    -------
    str - UUID of the created property.
    """
    if prop_uuid is None:
        prop_uuid = str(_uuid.uuid4())

    values = grid_property.values.flatten()
    name = grid_property.name

    # Resolve OSDU mapping
    mapping = resolve_property_mapping(title=name)
    if mapping:
        property_kind = mapping.osdu_name or name
        uom = mapping.uom_family
        is_discrete = mapping.is_discrete
    else:
        property_kind = name
        uom = ""
        is_discrete = bool(grid_property.isdiscrete)

    provider.put_property_values(
        uuid=prop_uuid,
        title=name,
        values=values,
        supporting_representation_uuid=grid_uuid,
        property_kind=property_kind,
        indexable_element="cells",
        is_discrete=is_discrete,
        uom=uom,
    )

    return prop_uuid
