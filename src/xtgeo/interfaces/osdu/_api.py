# -*- coding: utf-8 -*-
"""User-friendly high-level API for OSDU/RESQML data access.

This module provides the public API for reading and writing xtgeo objects
to/from OSDU Reservoir DDMS (RDDMS) servers or EPC files in a style consistent
with xtgeo's existing IO patterns (grid_from_file, surface_from_file, etc.).

Key functions:
  - ``grid_from_osdu()`` — read a grid (with optional properties)
  - ``surface_from_osdu()`` — read a surface/map
  - ``points_from_osdu()`` — read a pointset
  - ``polygons_from_osdu()`` — read a polylineset
  - ``grid_to_osdu()`` — write a grid (with optional properties)
  - ``surface_to_osdu()`` — write a surface
  - ``points_to_osdu()`` — write a pointset
  - ``polygons_to_osdu()`` — write a polylineset
  - ``list_osdu_objects()`` — discover available objects
  - ``search_osdu()`` — search objects by name/type/keyword

These functions accept either:
  - An ``OsduSession`` (for ETP connections to RDDMS)
  - A path to an EPC file (for file-based access)

Example usage::

    import xtgeo
    from xtgeo.interfaces.osdu import OsduSession

    session = OsduSession(etp_url="ws://localhost:9002", dataspace="maap/drogon")
    # or: session = OsduSession.load("equinor-dev")

    # List all objects
    objects = xtgeo.list_osdu_objects(session)

    # Search for grids
    grids = xtgeo.search_osdu(session, name="*Drogon*", object_type="grid")

    # Read a grid with properties
    grid, props = xtgeo.grid_from_osdu(session, name="Drogon")
    # or by UUID: grid, props = xtgeo.grid_from_osdu(session, uuid="abc-123")

    # Write to OSDU
    xtgeo.grid_to_osdu(session, grid, title="MyGrid", properties=props, crs_epsg=23031)

    # Read a surface
    surf = xtgeo.surface_from_osdu(session, name="TopVolantis")
"""

from __future__ import annotations

import fnmatch
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from ._epc_provider import EpcFileProvider
from ._etp_provider import EtpProvider
from ._provider_base import ResqmlDataProvider
from ._session import OsduSession

if TYPE_CHECKING:
    import os

logger = logging.getLogger(__name__)

# Type alias for session-or-path
SessionLike = Union[OsduSession, str, "os.PathLike[str]"]


def _open_provider(
    source: SessionLike, mode: str = "r"
) -> Tuple[ResqmlDataProvider, bool]:
    """Open a provider from a session or file path.

    Returns (provider, needs_close) tuple.
    """
    import os

    if isinstance(source, ResqmlDataProvider):
        return source, False
    if isinstance(source, OsduSession):
        config = source.etp_config()
        p = EtpProvider(config)
        p.open()
        return p, True
    if isinstance(source, (str, os.PathLike)):
        path = str(source)
        if path.endswith(".epc"):
            p = EpcFileProvider(path, mode=mode)
            p.open()
            return p, True
        raise ValueError(
            f"Unsupported file format: {path}. Expected .epc file or OsduSession."
        )
    raise TypeError(f"Expected OsduSession or file path, got {type(source).__name__}")


# ---------------------------------------------------------------------------
# Discovery & Search
# ---------------------------------------------------------------------------


def list_osdu_objects(
    source: SessionLike,
    object_type: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """List available RESQML objects in a dataspace or EPC file.

    Parameters
    ----------
    source : OsduSession or str
        An OSDU session (for ETP) or path to an EPC file.
    object_type : str, optional
        Filter by type: "grid", "surface", "points", "polygons",
        "property", "crs", or any RESQML type substring.

    Returns
    -------
    list of dict
        Each dict has keys: 'uuid', 'title', 'type', and optionally 'uri'.

    Examples
    --------
    >>> from xtgeo.interfaces.osdu import OsduSession
    >>> session = OsduSession(etp_url="ws://localhost:9002", dataspace="maap/drogon")
    >>> objects = list_osdu_objects(session)
    >>> grids = list_osdu_objects(session, object_type="grid")
    """
    # Map user-friendly type names to RESQML types
    type_map = {
        "grid": "IjkGrid",
        "surface": "Grid2d",
        "map": "Grid2d",
        "points": "PointSet",
        "pointset": "PointSet",
        "polygons": "PolylineSet",
        "polylines": "PolylineSet",
        "property": "Property",
        "crs": "LocalDepth3dCrs",
    }
    resqml_type = (
        type_map.get(object_type.lower(), object_type) if object_type else None
    )

    provider, needs_close = _open_provider(source)
    try:
        return provider.list_objects(resqml_type)
    finally:
        if needs_close:
            provider.close()


def search_osdu(
    source: SessionLike,
    *,
    name: Optional[str] = None,
    object_type: Optional[str] = None,
    uuid: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Search for RESQML objects by name pattern, type, or UUID.

    Parameters
    ----------
    source : OsduSession or str
        An OSDU session or path to an EPC file.
    name : str, optional
        Name/title pattern (supports wildcards: ``*``, ``?``).
        Case-insensitive.
    object_type : str, optional
        Filter by type (same options as ``list_osdu_objects``).
    uuid : str, optional
        Find a specific object by UUID.

    Returns
    -------
    list of dict
        Matching objects with 'uuid', 'title', 'type' keys.

    Examples
    --------
    >>> results = search_osdu(session, name="*Drogon*", object_type="grid")
    >>> results = search_osdu(session, name="PORO", object_type="property")
    """
    objects = list_osdu_objects(source, object_type=object_type)

    if uuid:
        objects = [o for o in objects if o.get("uuid", "").lower() == uuid.lower()]

    if name:
        pattern = name.lower()
        objects = [
            o for o in objects if fnmatch.fnmatch(o.get("title", "").lower(), pattern)
        ]

    return objects


def list_osdu_dataspaces(source: SessionLike) -> List[Dict[str, Any]]:
    """List all dataspaces available on an RDDMS server.

    Parameters
    ----------
    source : OsduSession
        An OSDU session connected to an RDDMS server.

    Returns
    -------
    list of dict
        Each dict has 'path', 'uri', and 'last_changed' keys.

    Examples
    --------
    >>> for ds in xtgeo.list_osdu_dataspaces(session):
    ...     print(ds['path'])
    maap/drogon
    maap/production
    """
    from ._etp_provider import EtpProvider

    if not isinstance(source, OsduSession):
        raise TypeError("list_osdu_dataspaces requires an OsduSession (ETP connection)")

    config = source.etp_config()
    provider = EtpProvider(config)
    provider.open()
    try:
        return provider.get_dataspaces()
    finally:
        provider.close()


def query_osdu(
    source: SessionLike,
    *,
    name: Optional[str] = None,
    object_type: Optional[str] = None,
    dataspace: Optional[str] = None,
    uuid: Optional[str] = None,
    format: str = "list",
) -> Any:
    """Query RESQML objects with flexible filtering. The main discovery tool.

    Combines searching by name pattern, type, and optionally switches dataspace
    before querying. Returns results as a list of dicts or as a formatted table.

    Parameters
    ----------
    source : OsduSession or str
        An OSDU session or path to an EPC file.
    name : str, optional
        Name/title pattern (supports wildcards: ``*``, ``?``). Case-insensitive.
    object_type : str, optional
        Filter by type: "grid", "surface", "points", "polygons", "property", "crs".
    dataspace : str, optional
        Query a specific dataspace (temporarily switches, then restores).
        Only works with OsduSession sources.
    uuid : str, optional
        Find a specific object by UUID.
    format : str
        Output format: "list" (default, list of dicts), "table" (formatted string),
        "df" (pandas DataFrame).

    Returns
    -------
    list of dict, str, or pandas.DataFrame
        Depending on ``format`` parameter.

    Examples
    --------
    >>> # Search grids across a specific dataspace
    >>> results = xtgeo.query_osdu(session, object_type="grid", dataspace="maap/drogon")

    >>> # Wildcard name search
    >>> results = xtgeo.query_osdu(session, name="*poro*", object_type="property")

    >>> # Get a formatted table for display
    >>> print(xtgeo.query_osdu(session, format="table"))
    TYPE                 UUID         TITLE
    IjkGrid              a1b2c3d4...  Drogon
    Grid2d               e5f6g7h8...  TopVolantis
    ContinuousProperty   i9j0k1l2...  PORO

    >>> # Get a pandas DataFrame
    >>> df = xtgeo.query_osdu(session, format="df")
    >>> df[df["type"].str.contains("Property")]
    """
    original_dataspace = None

    # Temporarily switch dataspace if requested
    if dataspace and isinstance(source, OsduSession):
        original_dataspace = source.dataspace
        source.dataspace = dataspace

    try:
        results = search_osdu(source, name=name, object_type=object_type, uuid=uuid)
        # Add dataspace info to results
        if isinstance(source, OsduSession):
            ds = source.dataspace
            for r in results:
                r.setdefault("dataspace", ds)
    finally:
        if original_dataspace is not None:
            source.dataspace = original_dataspace

    if format == "table":
        return _format_table(results)
    if format == "df":
        import pandas as pd

        return pd.DataFrame(results)
    return results


def query_osdu_all_dataspaces(
    source: SessionLike,
    *,
    name: Optional[str] = None,
    object_type: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Search across ALL dataspaces on the server.

    Iterates over every dataspace and collects matching objects.
    Useful for discovering where data lives.

    Parameters
    ----------
    source : OsduSession
        An OSDU session.
    name : str, optional
        Name pattern (wildcards supported).
    object_type : str, optional
        Filter by type.

    Returns
    -------
    list of dict
        Matching objects, each with a 'dataspace' key indicating source.

    Examples
    --------
    >>> # Find all grids across all dataspaces
    >>> all_grids = xtgeo.query_osdu_all_dataspaces(session, object_type="grid")
    >>> for g in all_grids:
    ...     print(f"{g['dataspace']:20s} {g['title']}")
    maap/drogon          Drogon
    maap/production      ProdGrid
    """
    if not isinstance(source, OsduSession):
        raise TypeError("query_osdu_all_dataspaces requires an OsduSession")

    dataspaces = list_osdu_dataspaces(source)
    all_results = []

    original_dataspace = source.dataspace
    try:
        for ds in dataspaces:
            ds_path = ds.get("path", "")
            if not ds_path:
                continue
            source.dataspace = ds_path
            try:
                results = search_osdu(source, name=name, object_type=object_type)
                for r in results:
                    r["dataspace"] = ds_path
                all_results.extend(results)
            except Exception as e:
                logger.debug("Error querying dataspace %s: %s", ds_path, e)
    finally:
        source.dataspace = original_dataspace

    return all_results


def import_osdu(
    source: SessionLike,
    result: Dict[str, Any],
    *,
    load_properties: bool = True,
) -> Any:
    """Import an xtgeo object directly from a search/query result.

    Takes a single dict from ``query_osdu()`` or ``search_osdu()`` results
    and loads the corresponding xtgeo object.

    Parameters
    ----------
    source : OsduSession or str
        The same source used for the query.
    result : dict
        A single result dict (must have 'uuid' and 'type' keys).
    load_properties : bool
        For grids, whether to also load properties.

    Returns
    -------
    xtgeo object
        The appropriate xtgeo type: Grid (+ props), RegularSurface, Points, or Polygons.

    Raises
    ------
    ValueError
        If object type is not recognized.

    Examples
    --------
    >>> results = xtgeo.search_osdu(session, name="Drogon", object_type="grid")
    >>> grid, props = xtgeo.import_osdu(session, results[0])

    >>> surfaces = xtgeo.search_osdu(session, object_type="surface")
    >>> for r in surfaces:
    ...     surf = xtgeo.import_osdu(session, r)
    ...     print(f"{r['title']}: {surf.ncol}x{surf.nrow}")
    """
    obj_type = result.get("type", "").lower()
    obj_uuid = result.get("uuid", "")

    if not obj_uuid:
        raise ValueError("Result dict must have a 'uuid' key")

    # Temporarily switch dataspace if result has one
    original_dataspace = None
    if "dataspace" in result and isinstance(source, OsduSession):
        original_dataspace = source.dataspace
        source.dataspace = result["dataspace"]

    try:
        if "ijkgrid" in obj_type or "grid" in obj_type and "2d" not in obj_type:
            return grid_from_osdu(
                source, uuid=obj_uuid, load_properties=load_properties
            )
        if "grid2d" in obj_type or "surface" in obj_type or "map" in obj_type:
            return surface_from_osdu(source, uuid=obj_uuid)
        if "pointset" in obj_type or "point" in obj_type:
            return points_from_osdu(source, uuid=obj_uuid)
        if "polylineset" in obj_type or "polyline" in obj_type or "polygon" in obj_type:
            return polygons_from_osdu(source, uuid=obj_uuid)
        raise ValueError(
            f"Cannot import object type '{result.get('type')}'. "
            f"Supported: IjkGrid, Grid2d, PointSet, PolylineSet"
        )
    finally:
        if original_dataspace is not None:
            source.dataspace = original_dataspace


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _format_table(results: List[Dict[str, Any]]) -> str:
    """Format query results as a readable table string."""
    if not results:
        return "(no objects found)"

    # Determine columns
    has_dataspace = any("dataspace" in r for r in results)

    lines = []
    if has_dataspace:
        header = f"{'TYPE':<25s} {'UUID':<12s} {'DATASPACE':<20s} {'TITLE'}"
        lines.append(header)
        lines.append("-" * len(header))
        for r in results:
            short_type = r.get("type", "").split(".")[-1][:24]
            short_uuid = (
                r.get("uuid", "")[:11] + "…"
                if len(r.get("uuid", "")) > 11
                else r.get("uuid", "")
            )
            title = r.get("title", "")
            ds = r.get("dataspace", "")
            lines.append(f"{short_type:<25s} {short_uuid:<12s} {ds:<20s} {title}")
    else:
        header = f"{'TYPE':<25s} {'UUID':<12s} {'TITLE'}"
        lines.append(header)
        lines.append("-" * len(header))
        for r in results:
            short_type = r.get("type", "").split(".")[-1][:24]
            short_uuid = (
                r.get("uuid", "")[:11] + "…"
                if len(r.get("uuid", "")) > 11
                else r.get("uuid", "")
            )
            lines.append(f"{short_type:<25s} {short_uuid:<12s} {r.get('title', '')}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Read functions
# ---------------------------------------------------------------------------


def grid_from_osdu(
    source: SessionLike,
    *,
    uuid: Optional[str] = None,
    name: Optional[str] = None,
    load_properties: bool = True,
) -> Tuple[Any, List[Any]]:
    """Read an IJK Grid from OSDU/RDDMS or an EPC file.

    Parameters
    ----------
    source : OsduSession or str
        An OSDU session or path to an EPC file.
    uuid : str, optional
        UUID of the grid to read. Required if ``name`` is not given.
    name : str, optional
        Title/name of the grid. First match is used.
    load_properties : bool
        If True, also load associated grid properties.

    Returns
    -------
    tuple of (xtgeo.Grid, list of xtgeo.GridProperty)

    Raises
    ------
    ValueError
        If no grid found matching the criteria.

    Examples
    --------
    >>> grid, props = grid_from_osdu(session, name="Drogon")
    >>> grid, props = grid_from_osdu("model.epc", uuid="abc-123-def")
    """
    from ._ijk_grid import ijk_grid_to_xtgeo

    grid_uuid = _resolve_uuid(source, uuid, name, "grid")
    provider, needs_close = _open_provider(source)
    try:
        return ijk_grid_to_xtgeo(provider, grid_uuid, load_properties=load_properties)
    finally:
        if needs_close:
            provider.close()


def surface_from_osdu(
    source: SessionLike,
    *,
    uuid: Optional[str] = None,
    name: Optional[str] = None,
) -> Any:
    """Read a regular surface (Grid2D) from OSDU/RDDMS or an EPC file.

    Parameters
    ----------
    source : OsduSession or str
        An OSDU session or path to an EPC file.
    uuid : str, optional
        UUID of the surface to read.
    name : str, optional
        Title/name of the surface. First match is used.

    Returns
    -------
    xtgeo.RegularSurface

    Examples
    --------
    >>> surf = surface_from_osdu(session, name="TopVolantis")
    """
    from ._grid2d import grid2d_to_xtgeo

    surf_uuid = _resolve_uuid(source, uuid, name, "surface")
    provider, needs_close = _open_provider(source)
    try:
        return grid2d_to_xtgeo(provider, surf_uuid)
    finally:
        if needs_close:
            provider.close()


def points_from_osdu(
    source: SessionLike,
    *,
    uuid: Optional[str] = None,
    name: Optional[str] = None,
) -> Any:
    """Read a PointSet from OSDU/RDDMS or an EPC file.

    Parameters
    ----------
    source : OsduSession or str
        An OSDU session or path to an EPC file.
    uuid : str, optional
        UUID of the pointset to read.
    name : str, optional
        Title/name of the pointset.

    Returns
    -------
    xtgeo.Points

    Examples
    --------
    >>> pts = points_from_osdu(session, name="WellTops")
    """
    from ._pointset import pointset_to_xtgeo

    ps_uuid = _resolve_uuid(source, uuid, name, "points")
    provider, needs_close = _open_provider(source)
    try:
        return pointset_to_xtgeo(provider, ps_uuid)
    finally:
        if needs_close:
            provider.close()


def polygons_from_osdu(
    source: SessionLike,
    *,
    uuid: Optional[str] = None,
    name: Optional[str] = None,
) -> Any:
    """Read a PolylineSet from OSDU/RDDMS or an EPC file.

    Parameters
    ----------
    source : OsduSession or str
        An OSDU session or path to an EPC file.
    uuid : str, optional
        UUID of the polylineset to read.
    name : str, optional
        Title/name of the polylineset.

    Returns
    -------
    xtgeo.Polygons

    Examples
    --------
    >>> polys = polygons_from_osdu(session, name="FaultTraces")
    """
    from ._polyline import polylineset_to_xtgeo

    pl_uuid = _resolve_uuid(source, uuid, name, "polygons")
    provider, needs_close = _open_provider(source)
    try:
        return polylineset_to_xtgeo(provider, pl_uuid)
    finally:
        if needs_close:
            provider.close()


# ---------------------------------------------------------------------------
# Write functions
# ---------------------------------------------------------------------------


def grid_to_osdu(
    source: SessionLike,
    grid: Any,
    *,
    title: str = "Exported Grid",
    properties: Optional[List[Any]] = None,
    crs_epsg: Optional[int] = None,
    crs_uuid: Optional[str] = None,
    grid_uuid: Optional[str] = None,
) -> Dict[str, str]:
    """Write an xtgeo Grid to OSDU/RDDMS or an EPC file.

    Parameters
    ----------
    source : OsduSession or str
        An OSDU session or path to an EPC file (will create if writing).
    grid : xtgeo.Grid
        The grid to export.
    title : str
        Title/name for the RESQML object.
    properties : list of xtgeo.GridProperty, optional
        Grid properties to export alongside the grid.
    crs_epsg : int, optional
        EPSG code for the coordinate reference system.
    crs_uuid : str, optional
        UUID of an existing CRS to reference.
    grid_uuid : str, optional
        Explicit UUID for the grid object.

    Returns
    -------
    dict
        Mapping of object titles to their UUIDs.

    Examples
    --------
    >>> uuids = grid_to_osdu(session, grid, title="MyGrid", crs_epsg=23031,
    ...                      properties=[poro, permx])
    """
    from ._ijk_grid import xtgeo_grid_to_resqml

    provider, needs_close = _open_provider(source, mode="w")
    try:
        return xtgeo_grid_to_resqml(
            provider,
            grid,
            title=title,
            grid_uuid=grid_uuid,
            crs_uuid=crs_uuid,
            crs_epsg=crs_epsg,
            properties=properties,
        )
    finally:
        if needs_close:
            provider.close()


def surface_to_osdu(
    source: SessionLike,
    surface: Any,
    *,
    title: str = "Exported Surface",
    crs_epsg: Optional[int] = None,
    crs_uuid: Optional[str] = None,
    surface_uuid: Optional[str] = None,
) -> Dict[str, str]:
    """Write an xtgeo RegularSurface to OSDU/RDDMS or an EPC file.

    Parameters
    ----------
    source : OsduSession or str
        An OSDU session or path to an EPC file.
    surface : xtgeo.RegularSurface
        The surface to export.
    title : str
        Title/name for the RESQML object.
    crs_epsg : int, optional
        EPSG code for the coordinate reference system.
    crs_uuid : str, optional
        UUID of an existing CRS.
    surface_uuid : str, optional
        Explicit UUID for the surface object.

    Returns
    -------
    dict
        Mapping of object titles to their UUIDs.

    Examples
    --------
    >>> uuids = surface_to_osdu(session, surf, title="TopReek", crs_epsg=23031)
    """
    from ._grid2d import xtgeo_surface_to_resqml

    provider, needs_close = _open_provider(source, mode="w")
    try:
        return xtgeo_surface_to_resqml(
            provider,
            surface,
            title=title,
            surface_uuid=surface_uuid,
            crs_uuid=crs_uuid,
            crs_epsg=crs_epsg,
        )
    finally:
        if needs_close:
            provider.close()


def points_to_osdu(
    source: SessionLike,
    points: Any,
    *,
    title: str = "Exported Points",
    crs_epsg: Optional[int] = None,
    crs_uuid: Optional[str] = None,
) -> Dict[str, str]:
    """Write xtgeo Points to OSDU/RDDMS or an EPC file.

    Parameters
    ----------
    source : OsduSession or str
        An OSDU session or path to an EPC file.
    points : xtgeo.Points
        The points to export.
    title : str
        Title for the RESQML object.
    crs_epsg : int, optional
        EPSG code for the CRS.
    crs_uuid : str, optional
        UUID of an existing CRS.

    Returns
    -------
    dict
        Mapping of object titles to their UUIDs.
    """
    from ._pointset import xtgeo_points_to_resqml

    provider, needs_close = _open_provider(source, mode="w")
    try:
        return xtgeo_points_to_resqml(
            provider,
            points,
            title=title,
            crs_uuid=crs_uuid,
            crs_epsg=crs_epsg,
        )
    finally:
        if needs_close:
            provider.close()


def polygons_to_osdu(
    source: SessionLike,
    polygons: Any,
    *,
    title: str = "Exported Polygons",
    crs_epsg: Optional[int] = None,
    crs_uuid: Optional[str] = None,
) -> Dict[str, str]:
    """Write xtgeo Polygons to OSDU/RDDMS or an EPC file.

    Parameters
    ----------
    source : OsduSession or str
        An OSDU session or path to an EPC file.
    polygons : xtgeo.Polygons
        The polygons to export.
    title : str
        Title for the RESQML object.
    crs_epsg : int, optional
        EPSG code for the CRS.
    crs_uuid : str, optional
        UUID of an existing CRS.

    Returns
    -------
    dict
        Mapping of object titles to their UUIDs.
    """
    from ._polyline import xtgeo_polygons_to_resqml

    provider, needs_close = _open_provider(source, mode="w")
    try:
        return xtgeo_polygons_to_resqml(
            provider,
            polygons,
            title=title,
            crs_uuid=crs_uuid,
            crs_epsg=crs_epsg,
        )
    finally:
        if needs_close:
            provider.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_uuid(
    source: SessionLike,
    uuid: Optional[str],
    name: Optional[str],
    object_type: str,
) -> str:
    """Resolve a UUID from either explicit uuid or name search."""
    if uuid:
        return uuid
    if not name:
        raise ValueError("Either 'uuid' or 'name' must be provided")

    results = search_osdu(source, name=name, object_type=object_type)
    if not results:
        raise ValueError(f"No {object_type} found with name matching '{name}'")
    return results[0]["uuid"]


# ---------------------------------------------------------------------------
# Deep Discovery
# ---------------------------------------------------------------------------


def deep_query_osdu(
    source: SessionLike,
    *,
    uuid: Optional[str] = None,
    depth: int = 0,
    scope: str = "targets",
    object_types: Optional[List[str]] = None,
    include_edges: bool = False,
) -> Dict[str, Any]:
    """Deep discovery query — traverse the object graph in a dataspace.

    Use this to explore relationships between RESQML objects: find all
    properties attached to a grid, discover the full object tree, or
    filter by type across the graph.

    Parameters
    ----------
    source : OsduSession or str
        Session object or EPC file path.
    uuid : str, optional
        Start traversal from this specific object. If None, starts from
        the dataspace root.
    depth : int
        Traversal depth: 1=direct relations, 2=two hops, 0=unlimited.
    scope : str
        "targets", "sources", "self", "targets_or_self", "sources_or_self".
    object_types : list of str, optional
        Filter by RESQML type short names, e.g. ["IjkGrid", "ContinuousProperty"].
    include_edges : bool
        If True, return relationship edges between objects.

    Returns
    -------
    dict
        - 'resources': list of object dicts (uuid, title, type, uri, timestamps)
        - 'edges': list of edge dicts (source_uri, target_uri, relationship_kind)

    Examples
    --------
    >>> import xtgeo
    >>> session = xtgeo.OsduSession(url="ws://localhost:9002", dataspace="demo")

    >>> # Discover everything in the dataspace
    >>> result = xtgeo.deep_query_osdu(session, depth=0)
    >>> for r in result['resources']:
    ...     print(f"{r['type']:30s} {r['title']}")

    >>> # Find all properties referencing a specific grid
    >>> result = xtgeo.deep_query_osdu(session, uuid=grid_uuid, scope="sources")
    """
    provider, needs_close = _open_provider(source, mode="r")
    try:
        # Resolve the starting URI from uuid if given
        uri = None
        if uuid:
            objects = provider.list_objects()
            for obj in objects:
                if obj["uuid"] == uuid:
                    uri = obj["uri"]
                    break
            if uri is None:
                raise ValueError(f"Object {uuid} not found in dataspace")

        # Qualify type names
        qualified_types = None
        if object_types:
            qualified_types = [t if "." in t else f"resqml20.{t}" for t in object_types]

        return provider.discover(
            uri=uri,
            depth=depth,
            scope=scope,
            object_types=qualified_types,
            include_edges=include_edges,
        )
    finally:
        if needs_close:
            provider.close()


def watch_osdu_changes(
    source: SessionLike,
    *,
    object_types: Optional[List[str]] = None,
    uuids: Optional[List[str]] = None,
    callback: Optional[Any] = None,
):
    """Subscribe to change notifications in a dataspace.

    Returns a subscription handle that can be polled for changes.
    Uses timestamp-based change detection (polling) since ETP Protocol 5
    (StoreNotification) message classes are not available in the library.

    Parameters
    ----------
    source : OsduSession or str
        Session object or EPC file path.
    object_types : list of str, optional
        RESQML types to watch (e.g. ["IjkGrid", "ContinuousProperty"]).
    uuids : list of str, optional
        Specific object UUIDs to watch.
    callback : callable, optional
        Called with (event_type, event_dict) on each detected change.

    Returns
    -------
    NotificationSubscription
        Subscription handle with .poll(), .stop(), and context manager support.

    Examples
    --------
    >>> import xtgeo
    >>> session = xtgeo.OsduSession(url="ws://localhost:9002", dataspace="demo")
    >>> sub = xtgeo.watch_osdu_changes(session, object_types=["IjkGrid"])
    >>> # ... modify objects ...
    >>> events = sub.poll()
    >>> for e in events:
    ...     print(f"{e['event']:8s} {e['title']}")
    """
    provider, needs_close = _open_provider(source, mode="r")
    # Note: don't close provider here — the subscription needs it alive
    return provider.subscribe_notifications(
        object_types=object_types,
        uuids=uuids,
        callback=callback,
    )
