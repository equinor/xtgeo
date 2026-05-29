# -*- coding: utf-8 -*-
"""Dataspace-level operations: read, write, and compare entire datasets.

Provides a typed container (DataspaceSnapshot) for holding all objects from a
RESQML dataspace and functions to read/write/compare them generically.

Design:
  - Generic type dispatch via RESQML qualified_type strings
  - Provider-agnostic (works with EPC or ETP backends)
  - Preserves UUIDs, titles, CRS metadata, and all property associations
  - Comparison is bitwise-exact for arrays, tolerance-based for floats

Usage::

    from xtgeo.interfaces.osdu._dataspace import (
        read_dataspace, write_dataspace, compare_snapshots,
    )

    # Read everything from source
    snap = read_dataspace(source_provider)

    # Write to target
    write_dataspace(target_provider, snap)

    # Compare two snapshots for equivalence
    diffs = compare_snapshots(snap_a, snap_b)
    assert not diffs, f"Differences found: {diffs}"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np

if TYPE_CHECKING:
    from ._provider_base import ResqmlDataProvider

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Typed object containers
# ---------------------------------------------------------------------------


@dataclass
class ResqmlObject:
    """Generic container for a single RESQML object with its data."""

    uuid: str
    title: str
    resqml_type: str  # qualified type, e.g. "resqml20.IjkGridRepresentation"
    crs_uuid: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GridSnapshot(ResqmlObject):
    """IJK Grid with geometry and associated properties."""

    ni: int = 0
    nj: int = 0
    nk: int = 0
    coord: Optional[np.ndarray] = None
    zcorn: Optional[np.ndarray] = None
    actnum: Optional[np.ndarray] = None
    k_direction: str = "down"
    properties: List["PropertySnapshot"] = field(default_factory=list)


@dataclass
class SurfaceSnapshot(ResqmlObject):
    """Grid2D (regular surface) with geometry."""

    ni: int = 0
    nj: int = 0
    origin_x: float = 0.0
    origin_y: float = 0.0
    di: float = 1.0
    dj: float = 1.0
    rotation: float = 0.0  # radians
    values: Optional[np.ndarray] = None


@dataclass
class PointSetSnapshot(ResqmlObject):
    """PointSet with coordinate array."""

    points: Optional[np.ndarray] = None  # (N, 3)


@dataclass
class PolylineSetSnapshot(ResqmlObject):
    """PolylineSet with polyline arrays."""

    polylines: List[np.ndarray] = field(default_factory=list)  # each (M, 3)
    closed: List[bool] = field(default_factory=list)


@dataclass
class TriangulatedSurfaceSnapshot(ResqmlObject):
    """TriangulatedSetRepresentation with vertices and triangles."""

    vertices: Optional[np.ndarray] = None  # (N, 3)
    triangles: Optional[np.ndarray] = None  # (M, 3) int


@dataclass
class WellSnapshot(ResqmlObject):
    """WellboreTrajectoryRepresentation with MD + XYZ + logs."""

    md: Optional[np.ndarray] = None
    x: Optional[np.ndarray] = None
    y: Optional[np.ndarray] = None
    z: Optional[np.ndarray] = None
    logs: Dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class BlockedWellSnapshot(ResqmlObject):
    """BlockedWellboreRepresentation with cell indices + properties."""

    md: Optional[np.ndarray] = None
    x: Optional[np.ndarray] = None
    y: Optional[np.ndarray] = None
    z: Optional[np.ndarray] = None
    cell_indices_i: Optional[np.ndarray] = None
    cell_indices_j: Optional[np.ndarray] = None
    cell_indices_k: Optional[np.ndarray] = None
    grid_uuid: str = ""
    trajectory_uuid: str = ""
    properties: Dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class PropertySnapshot:
    """Grid property data."""

    uuid: str = ""
    title: str = ""
    resqml_type: str = ""
    property_kind: str = ""
    indexable_element: str = "cells"
    supporting_representation_uuid: str = ""
    is_discrete: bool = False
    uom: str = ""
    facet: Optional[str] = None
    values: Optional[np.ndarray] = None


@dataclass
class CrsSnapshot:
    """Coordinate reference system."""

    uuid: str = ""
    title: str = ""
    origin_x: float = 0.0
    origin_y: float = 0.0
    origin_z: float = 0.0
    areal_rotation: float = 0.0
    z_increasing_downward: bool = True
    projected_crs_epsg: Optional[int] = None
    vertical_crs_epsg: Optional[int] = None


@dataclass
class DataspaceSnapshot:
    """Complete snapshot of all objects in a dataspace."""

    grids: List[GridSnapshot] = field(default_factory=list)
    surfaces: List[SurfaceSnapshot] = field(default_factory=list)
    pointsets: List[PointSetSnapshot] = field(default_factory=list)
    polylinesets: List[PolylineSetSnapshot] = field(default_factory=list)
    triangulated_surfaces: List[TriangulatedSurfaceSnapshot] = field(
        default_factory=list
    )
    wells: List[WellSnapshot] = field(default_factory=list)
    blocked_wells: List[BlockedWellSnapshot] = field(default_factory=list)
    crs_list: List[CrsSnapshot] = field(default_factory=list)

    @property
    def object_count(self) -> int:
        """Return total number of objects in the snapshot."""
        return (
            len(self.grids)
            + len(self.surfaces)
            + len(self.pointsets)
            + len(self.polylinesets)
            + len(self.triangulated_surfaces)
            + len(self.wells)
            + len(self.blocked_wells)
            + len(self.crs_list)
            + sum(len(g.properties) for g in self.grids)
        )

    def summary(self) -> str:
        """Return a human-readable summary of snapshot contents."""
        parts = []
        if self.grids:
            parts.append(f"{len(self.grids)} grids")
            nprops = sum(len(g.properties) for g in self.grids)
            if nprops:
                parts.append(f"{nprops} properties")
        if self.surfaces:
            parts.append(f"{len(self.surfaces)} surfaces")
        if self.pointsets:
            parts.append(f"{len(self.pointsets)} pointsets")
        if self.polylinesets:
            parts.append(f"{len(self.polylinesets)} polylinesets")
        if self.triangulated_surfaces:
            parts.append(
                f"{len(self.triangulated_surfaces)} triangulated surfaces"
            )
        if self.wells:
            parts.append(f"{len(self.wells)} wells")
        if self.blocked_wells:
            parts.append(f"{len(self.blocked_wells)} blocked wells")
        if self.crs_list:
            parts.append(f"{len(self.crs_list)} CRS")
        return ", ".join(parts) if parts else "empty"


# ---------------------------------------------------------------------------
# Type classification
# ---------------------------------------------------------------------------

_TYPE_MAP = {
    "IjkGridRepresentation": "grid",
    "Grid2dRepresentation": "surface",
    "PointSetRepresentation": "pointset",
    "PolylineSetRepresentation": "polylineset",
    "TriangulatedSetRepresentation": "triangulated_surface",
    "WellboreTrajectoryRepresentation": "well",
    "BlockedWellboreRepresentation": "blocked_well",
    "LocalDepth3dCrs": "crs",
    "ContinuousProperty": "property",
    "DiscreteProperty": "property",
    "CategoricalProperty": "property",
}


def _classify_type(qualified_type: str) -> str:
    """Classify a RESQML qualified type into a category."""
    for key, category in _TYPE_MAP.items():
        if key in qualified_type:
            return category
    return "unknown"


# ---------------------------------------------------------------------------
# Read entire dataspace
# ---------------------------------------------------------------------------


def read_dataspace(provider: ResqmlDataProvider) -> DataspaceSnapshot:
    """Read all supported objects from a provider into a DataspaceSnapshot.

    Parameters
    ----------
    provider : ResqmlDataProvider
        An open provider (EPC or ETP).

    Returns
    -------
    DataspaceSnapshot
        Complete typed snapshot of the dataspace contents.
    """
    all_objects = provider.list_objects()
    snapshot = DataspaceSnapshot()

    # Classify objects
    grids_info = []
    surfaces_info = []
    pointsets_info = []
    polylinesets_info = []
    trisurfaces_info = []
    wells_info = []
    blocked_wells_info = []
    crs_info = []
    properties_info = []

    for obj in all_objects:
        category = _classify_type(obj.get("type", ""))
        if category == "grid":
            grids_info.append(obj)
        elif category == "surface":
            surfaces_info.append(obj)
        elif category == "pointset":
            pointsets_info.append(obj)
        elif category == "polylineset":
            polylinesets_info.append(obj)
        elif category == "triangulated_surface":
            trisurfaces_info.append(obj)
        elif category == "well":
            wells_info.append(obj)
        elif category == "blocked_well":
            blocked_wells_info.append(obj)
        elif category == "crs":
            crs_info.append(obj)
        elif category == "property":
            properties_info.append(obj)
        else:
            logger.debug(
                "Skipping unsupported type: %s (%s)", obj.get("type"), obj.get("title")
            )

    # Read CRS objects first (needed for reference)
    for obj in crs_info:
        try:
            crs_data = provider.get_crs(obj["uuid"])
            snapshot.crs_list.append(
                CrsSnapshot(
                    uuid=obj["uuid"],
                    title=crs_data.get("title", obj.get("title", "")),
                    origin_x=crs_data.get("origin_x", 0.0),
                    origin_y=crs_data.get("origin_y", 0.0),
                    origin_z=crs_data.get("origin_z", 0.0),
                    areal_rotation=crs_data.get("areal_rotation", 0.0),
                    z_increasing_downward=crs_data.get("z_increasing_downward", True),
                    projected_crs_epsg=crs_data.get("projected_crs_epsg"),
                    vertical_crs_epsg=crs_data.get("vertical_crs_epsg"),
                )
            )
        except Exception as e:
            logger.warning("Failed to read CRS %s: %s", obj["uuid"], e)

    # Read grids
    for obj in grids_info:
        try:
            geom = provider.get_ijk_grid_geometry(obj["uuid"])
            grid_snap = GridSnapshot(
                uuid=obj["uuid"],
                title=obj.get("title", ""),
                resqml_type=obj.get("type", "resqml20.IjkGridRepresentation"),
                crs_uuid=geom.get("crs_uuid", ""),
                ni=geom["ni"],
                nj=geom["nj"],
                nk=geom["nk"],
                coord=geom["coord"],
                zcorn=geom["zcorn"],
                actnum=geom["actnum"],
                k_direction=geom.get("k_direction", "down"),
            )
            snapshot.grids.append(grid_snap)
        except Exception as e:
            logger.warning("Failed to read grid %s: %s", obj["uuid"], e)

    # Read properties and associate with grids
    for obj in properties_info:
        try:
            obj_type = obj.get("type", "resqml20.ContinuousProperty")
            obj_type_short = obj_type.split(".")[-1] if "." in obj_type else obj_type
            prop_data = provider.get_property_values(
                obj["uuid"], object_type=obj_type_short
            )

            prop_snap = PropertySnapshot(
                uuid=obj["uuid"],
                title=prop_data.get("title", obj.get("title", "")),
                resqml_type=obj_type,
                property_kind=prop_data.get("property_kind", ""),
                indexable_element=prop_data.get("indexable_element", "cells"),
                supporting_representation_uuid=prop_data.get(
                    "supporting_representation_uuid", ""
                ),
                is_discrete=prop_data.get("is_discrete", False),
                uom=prop_data.get("uom", ""),
                facet=prop_data.get("facet"),
                values=prop_data.get("values"),
            )

            # Associate with parent grid
            attached = False
            for grid_snap in snapshot.grids:
                if grid_snap.uuid == prop_snap.supporting_representation_uuid:
                    grid_snap.properties.append(prop_snap)
                    attached = True
                    break
            if not attached:
                logger.debug(
                    "Property %s (%s) orphaned - no matching grid %s",
                    prop_snap.title,
                    prop_snap.uuid,
                    prop_snap.supporting_representation_uuid,
                )
        except Exception as e:
            logger.warning("Failed to read property %s: %s", obj["uuid"], e)

    # Read surfaces
    for obj in surfaces_info:
        try:
            geom = provider.get_grid2d_geometry(obj["uuid"])
            snapshot.surfaces.append(
                SurfaceSnapshot(
                    uuid=obj["uuid"],
                    title=obj.get("title", ""),
                    resqml_type=obj.get("type", "resqml20.Grid2dRepresentation"),
                    crs_uuid=geom.get("crs_uuid", ""),
                    ni=geom["ni"],
                    nj=geom["nj"],
                    origin_x=geom["origin_x"],
                    origin_y=geom["origin_y"],
                    di=geom["di"],
                    dj=geom["dj"],
                    rotation=geom.get("rotation", 0.0),
                    values=geom["values"],
                )
            )
        except Exception as e:
            logger.warning("Failed to read surface %s: %s", obj["uuid"], e)

    # Read pointsets
    for obj in pointsets_info:
        try:
            data = provider.get_pointset(obj["uuid"])
            snapshot.pointsets.append(
                PointSetSnapshot(
                    uuid=obj["uuid"],
                    title=obj.get("title", ""),
                    resqml_type=obj.get("type", "resqml20.PointSetRepresentation"),
                    crs_uuid=data.get("crs_uuid", ""),
                    points=data["points"],
                )
            )
        except Exception as e:
            logger.warning("Failed to read pointset %s: %s", obj["uuid"], e)

    # Read polylinesets
    for obj in polylinesets_info:
        try:
            data = provider.get_polylineset(obj["uuid"])
            snapshot.polylinesets.append(
                PolylineSetSnapshot(
                    uuid=obj["uuid"],
                    title=obj.get("title", ""),
                    resqml_type=obj.get("type", "resqml20.PolylineSetRepresentation"),
                    crs_uuid=data.get("crs_uuid", ""),
                    polylines=data["polylines"],
                    closed=data["closed"],
                )
            )
        except Exception as e:
            logger.warning("Failed to read polylineset %s: %s", obj["uuid"], e)

    # Read triangulated surfaces
    for obj in trisurfaces_info:
        try:
            data = provider.get_triangulated_set(obj["uuid"])
            snapshot.triangulated_surfaces.append(
                TriangulatedSurfaceSnapshot(
                    uuid=obj["uuid"],
                    title=obj.get("title", ""),
                    resqml_type=obj.get(
                        "type", "resqml20.TriangulatedSetRepresentation"
                    ),
                    crs_uuid=data.get("crs_uuid", ""),
                    vertices=data["vertices"],
                    triangles=data["triangles"],
                )
            )
        except Exception as e:
            logger.warning("Failed to read triangulated surface %s: %s", obj["uuid"], e)

    # Read wells (trajectories)
    for obj in wells_info:
        try:
            data = provider.get_wellbore_trajectory(obj["uuid"])
            snapshot.wells.append(
                WellSnapshot(
                    uuid=obj["uuid"],
                    title=obj.get("title", ""),
                    resqml_type=obj.get(
                        "type", "resqml20.WellboreTrajectoryRepresentation"
                    ),
                    crs_uuid=data.get("crs_uuid", ""),
                    md=data.get("md"),
                    x=data.get("x"),
                    y=data.get("y"),
                    z=data.get("z"),
                )
            )
        except Exception as e:
            logger.warning("Failed to read well trajectory %s: %s", obj["uuid"], e)

    # Read blocked wells
    for obj in blocked_wells_info:
        try:
            data = provider.get_blocked_wellbore(obj["uuid"])
            snapshot.blocked_wells.append(
                BlockedWellSnapshot(
                    uuid=obj["uuid"],
                    title=obj.get("title", ""),
                    resqml_type=obj.get(
                        "type", "resqml20.BlockedWellboreRepresentation"
                    ),
                    crs_uuid=data.get("crs_uuid", ""),
                    md=data.get("md"),
                    x=data.get("x"),
                    y=data.get("y"),
                    z=data.get("z"),
                    cell_indices_i=data.get("cell_indices_i"),
                    cell_indices_j=data.get("cell_indices_j"),
                    cell_indices_k=data.get("cell_indices_k"),
                    grid_uuid=data.get("grid_uuid", ""),
                    trajectory_uuid=data.get("trajectory_uuid", ""),
                )
            )
        except Exception as e:
            logger.warning("Failed to read blocked well %s: %s", obj["uuid"], e)

    logger.info("Read dataspace: %s", snapshot.summary())
    return snapshot


# ---------------------------------------------------------------------------
# Write entire dataspace
# ---------------------------------------------------------------------------


def write_dataspace(
    provider: ResqmlDataProvider,
    snapshot: DataspaceSnapshot,
    *,
    preserve_uuids: bool = True,
) -> Dict[str, str]:
    """Write a DataspaceSnapshot to a provider.

    Parameters
    ----------
    provider : ResqmlDataProvider
        An open provider in write mode.
    snapshot : DataspaceSnapshot
        The dataset to write.
    preserve_uuids : bool
        If True, reuse the original UUIDs. If False, generate new ones.

    Returns
    -------
    dict
        Mapping of original UUIDs to written UUIDs.
    """
    import uuid as _uuid

    uuid_map: Dict[str, str] = {}

    def _get_uuid(original: str) -> str:
        if preserve_uuids:
            uuid_map[original] = original
            return original
        new = str(_uuid.uuid4())
        uuid_map[original] = new
        return new

    # Write CRS objects first (other objects reference them)
    for crs in snapshot.crs_list:
        new_uuid = _get_uuid(crs.uuid)
        provider.put_crs(
            uuid=new_uuid,
            title=crs.title,
            origin_x=crs.origin_x,
            origin_y=crs.origin_y,
            origin_z=crs.origin_z,
            areal_rotation=crs.areal_rotation,
            z_increasing_downward=crs.z_increasing_downward,
            projected_crs_epsg=crs.projected_crs_epsg,
            vertical_crs_epsg=crs.vertical_crs_epsg,
        )
        logger.debug("Wrote CRS: %s -> %s", crs.title, new_uuid)

    # Write grids
    for grid in snapshot.grids:
        new_uuid = _get_uuid(grid.uuid)
        crs_uuid = uuid_map.get(grid.crs_uuid, grid.crs_uuid)

        provider.put_ijk_grid_geometry(
            uuid=new_uuid,
            title=grid.title,
            ni=grid.ni,
            nj=grid.nj,
            nk=grid.nk,
            coord=grid.coord,
            zcorn=grid.zcorn,
            actnum=grid.actnum,
            crs_uuid=crs_uuid,
            k_direction=grid.k_direction,
        )
        logger.debug("Wrote grid: %s -> %s", grid.title, new_uuid)

        # Write associated properties
        for prop in grid.properties:
            prop_uuid = _get_uuid(prop.uuid)
            provider.put_property_values(
                uuid=prop_uuid,
                title=prop.title,
                values=prop.values,
                supporting_representation_uuid=new_uuid,
                property_kind=prop.property_kind,
                indexable_element=prop.indexable_element,
                is_discrete=prop.is_discrete,
                uom=prop.uom,
                facet=prop.facet,
            )
            logger.debug("Wrote property: %s -> %s", prop.title, prop_uuid)

    # Write surfaces
    for surf in snapshot.surfaces:
        new_uuid = _get_uuid(surf.uuid)
        crs_uuid = uuid_map.get(surf.crs_uuid, surf.crs_uuid)

        provider.put_grid2d_geometry(
            uuid=new_uuid,
            title=surf.title,
            ni=surf.ni,
            nj=surf.nj,
            origin_x=surf.origin_x,
            origin_y=surf.origin_y,
            di=surf.di,
            dj=surf.dj,
            rotation=surf.rotation,
            values=surf.values,
            crs_uuid=crs_uuid,
        )
        logger.debug("Wrote surface: %s -> %s", surf.title, new_uuid)

    # Write pointsets
    for ps in snapshot.pointsets:
        new_uuid = _get_uuid(ps.uuid)
        crs_uuid = uuid_map.get(ps.crs_uuid, ps.crs_uuid)

        provider.put_pointset(
            uuid=new_uuid,
            title=ps.title,
            points=ps.points,
            crs_uuid=crs_uuid,
        )
        logger.debug("Wrote pointset: %s -> %s", ps.title, new_uuid)

    # Write polylinesets
    for pls in snapshot.polylinesets:
        new_uuid = _get_uuid(pls.uuid)
        crs_uuid = uuid_map.get(pls.crs_uuid, pls.crs_uuid)

        provider.put_polylineset(
            uuid=new_uuid,
            title=pls.title,
            polylines=pls.polylines,
            closed=pls.closed,
            crs_uuid=crs_uuid,
        )
        logger.debug("Wrote polylineset: %s -> %s", pls.title, new_uuid)

    # Write triangulated surfaces
    for ts in snapshot.triangulated_surfaces:
        new_uuid = _get_uuid(ts.uuid)
        crs_uuid = uuid_map.get(ts.crs_uuid, ts.crs_uuid)

        provider.put_triangulated_set(
            uuid=new_uuid,
            title=ts.title,
            vertices=ts.vertices,
            triangles=ts.triangles,
            crs_uuid=crs_uuid,
        )
        logger.debug("Wrote triangulated surface: %s -> %s", ts.title, new_uuid)

    # Write wells (trajectories)
    for well in snapshot.wells:
        new_uuid = _get_uuid(well.uuid)
        crs_uuid = uuid_map.get(well.crs_uuid, well.crs_uuid)

        provider.put_wellbore_trajectory(
            uuid=new_uuid,
            title=well.title,
            md=well.md,
            x=well.x,
            y=well.y,
            z=well.z,
            crs_uuid=crs_uuid,
        )
        logger.debug("Wrote well trajectory: %s -> %s", well.title, new_uuid)

    # Write blocked wells
    for bw in snapshot.blocked_wells:
        new_uuid = _get_uuid(bw.uuid)
        crs_uuid = uuid_map.get(bw.crs_uuid, bw.crs_uuid)
        traj_uuid = uuid_map.get(bw.trajectory_uuid, bw.trajectory_uuid)
        grid_uuid = uuid_map.get(bw.grid_uuid, bw.grid_uuid)

        provider.put_blocked_wellbore(
            uuid=new_uuid,
            title=bw.title,
            md=bw.md,
            x=bw.x,
            y=bw.y,
            z=bw.z,
            cell_indices_i=bw.cell_indices_i,
            cell_indices_j=bw.cell_indices_j,
            cell_indices_k=bw.cell_indices_k,
            trajectory_uuid=traj_uuid,
            grid_uuid=grid_uuid,
            crs_uuid=crs_uuid,
        )
        logger.debug("Wrote blocked well: %s -> %s", bw.title, new_uuid)

    logger.info("Wrote dataspace: %s", snapshot.summary())
    return uuid_map


# ---------------------------------------------------------------------------
# Compare two snapshots
# ---------------------------------------------------------------------------


@dataclass
class Difference:
    """Single difference between two snapshot objects."""

    object_type: str
    title: str
    field: str
    detail: str


def compare_snapshots(
    a: DataspaceSnapshot,
    b: DataspaceSnapshot,
    *,
    atol: float = 1e-10,
    match_by: str = "title",
) -> List[Difference]:
    """Compare two DataspaceSnapshots for equivalence.

    Parameters
    ----------
    a, b : DataspaceSnapshot
        The two snapshots to compare.
    atol : float
        Absolute tolerance for floating-point array comparisons.
    match_by : str
        How to match objects between snapshots: "title" or "uuid".

    Returns
    -------
    list of Difference
        Empty list means the snapshots are equivalent.
    """
    diffs: List[Difference] = []

    # Compare CRS
    _compare_lists(
        a.crs_list,
        b.crs_list,
        "CRS",
        match_by,
        diffs,
        _compare_crs,
        atol=atol,
    )

    # Compare grids
    _compare_lists(
        a.grids,
        b.grids,
        "Grid",
        match_by,
        diffs,
        _compare_grids,
        atol=atol,
    )

    # Compare surfaces
    _compare_lists(
        a.surfaces,
        b.surfaces,
        "Surface",
        match_by,
        diffs,
        _compare_surfaces,
        atol=atol,
    )

    # Compare pointsets
    _compare_lists(
        a.pointsets,
        b.pointsets,
        "PointSet",
        match_by,
        diffs,
        _compare_pointsets,
        atol=atol,
    )

    # Compare polylinesets
    _compare_lists(
        a.polylinesets,
        b.polylinesets,
        "PolylineSet",
        match_by,
        diffs,
        _compare_polylinesets,
        atol=atol,
    )

    # Compare triangulated surfaces
    _compare_lists(
        a.triangulated_surfaces,
        b.triangulated_surfaces,
        "TriangulatedSurface",
        match_by,
        diffs,
        _compare_triangulated_surfaces,
        atol=atol,
    )

    # Compare wells
    _compare_lists(
        a.wells,
        b.wells,
        "Well",
        match_by,
        diffs,
        _compare_wells,
        atol=atol,
    )

    # Compare blocked wells
    _compare_lists(
        a.blocked_wells,
        b.blocked_wells,
        "BlockedWell",
        match_by,
        diffs,
        _compare_blocked_wells,
        atol=atol,
    )

    return diffs


def _compare_lists(list_a, list_b, obj_type, match_by, diffs, compare_fn, **kwargs):
    """Match objects from two lists and compare them pairwise."""
    if len(list_a) != len(list_b):
        diffs.append(
            Difference(obj_type, "", "count", f"{len(list_a)} vs {len(list_b)}")
        )

    key_fn = (lambda o: o.title) if match_by == "title" else (lambda o: o.uuid)

    index_b = {key_fn(obj): obj for obj in list_b}
    for obj_a in list_a:
        key = key_fn(obj_a)
        obj_b = index_b.get(key)
        if obj_b is None:
            diffs.append(
                Difference(obj_type, key, "missing", "not found in second snapshot")
            )
            continue
        compare_fn(obj_a, obj_b, diffs, **kwargs)


def _compare_crs(
    a: CrsSnapshot, b: CrsSnapshot, diffs: List[Difference], atol: float = 1e-10
):
    title = a.title
    if abs(a.origin_x - b.origin_x) > atol:
        diffs.append(
            Difference("CRS", title, "origin_x", f"{a.origin_x} vs {b.origin_x}")
        )
    if abs(a.origin_y - b.origin_y) > atol:
        diffs.append(
            Difference("CRS", title, "origin_y", f"{a.origin_y} vs {b.origin_y}")
        )
    if abs(a.origin_z - b.origin_z) > atol:
        diffs.append(
            Difference("CRS", title, "origin_z", f"{a.origin_z} vs {b.origin_z}")
        )
    if abs(a.areal_rotation - b.areal_rotation) > atol:
        diffs.append(
            Difference(
                "CRS",
                title,
                "areal_rotation",
                f"{a.areal_rotation} vs {b.areal_rotation}",
            )
        )
    if a.z_increasing_downward != b.z_increasing_downward:
        diffs.append(
            Difference(
                "CRS",
                title,
                "z_increasing_downward",
                f"{a.z_increasing_downward} vs {b.z_increasing_downward}",
            )
        )
    if a.projected_crs_epsg != b.projected_crs_epsg:
        diffs.append(
            Difference(
                "CRS",
                title,
                "projected_crs_epsg",
                f"{a.projected_crs_epsg} vs {b.projected_crs_epsg}",
            )
        )
    if a.vertical_crs_epsg != b.vertical_crs_epsg:
        diffs.append(
            Difference(
                "CRS",
                title,
                "vertical_crs_epsg",
                f"{a.vertical_crs_epsg} vs {b.vertical_crs_epsg}",
            )
        )


def _compare_grids(
    a: GridSnapshot, b: GridSnapshot, diffs: List[Difference], atol: float = 1e-10
):
    title = a.title
    if a.ni != b.ni or a.nj != b.nj or a.nk != b.nk:
        diffs.append(
            Difference(
                "Grid",
                title,
                "dimensions",
                f"({a.ni},{a.nj},{a.nk}) vs ({b.ni},{b.nj},{b.nk})",
            )
        )
        return

    if (
        a.coord is not None
        and b.coord is not None
        and not np.allclose(a.coord, b.coord, atol=atol, equal_nan=True)
    ):
        max_diff = np.nanmax(
            np.abs(a.coord.astype(np.float64) - b.coord.astype(np.float64))
        )
        diffs.append(Difference("Grid", title, "coord", f"max diff = {max_diff:.2e}"))

    if (
        a.zcorn is not None
        and b.zcorn is not None
        and not np.allclose(a.zcorn, b.zcorn, atol=atol, equal_nan=True)
    ):
        max_diff = np.nanmax(
            np.abs(a.zcorn.astype(np.float64) - b.zcorn.astype(np.float64))
        )
        diffs.append(Difference("Grid", title, "zcorn", f"max diff = {max_diff:.2e}"))

    if (
        a.actnum is not None
        and b.actnum is not None
        and not np.array_equal(a.actnum, b.actnum)
    ):
        n_diff = np.sum(a.actnum.flatten() != b.actnum.flatten())
        diffs.append(Difference("Grid", title, "actnum", f"{n_diff} cells differ"))

    # Compare properties
    _compare_property_lists(a.properties, b.properties, title, diffs, atol)


def _compare_property_lists(
    props_a: List[PropertySnapshot],
    props_b: List[PropertySnapshot],
    grid_title: str,
    diffs: List[Difference],
    atol: float,
):
    if len(props_a) != len(props_b):
        diffs.append(
            Difference(
                "Grid",
                grid_title,
                "property_count",
                f"{len(props_a)} vs {len(props_b)}",
            )
        )

    index_b = {p.title: p for p in props_b}
    for pa in props_a:
        pb = index_b.get(pa.title)
        if pb is None:
            diffs.append(
                Difference(
                    "Property",
                    pa.title,
                    "missing",
                    f"not in second snapshot (grid: {grid_title})",
                )
            )
            continue
        if pa.is_discrete != pb.is_discrete:
            diffs.append(
                Difference(
                    "Property",
                    pa.title,
                    "is_discrete",
                    f"{pa.is_discrete} vs {pb.is_discrete}",
                )
            )
        if pa.values is not None and pb.values is not None:
            if pa.is_discrete:
                if not np.array_equal(pa.values, pb.values):
                    n_diff = np.sum(pa.values.flatten() != pb.values.flatten())
                    diffs.append(
                        Difference(
                            "Property", pa.title, "values", f"{n_diff} cells differ"
                        )
                    )
            else:
                if not np.allclose(pa.values, pb.values, atol=atol, equal_nan=True):
                    max_diff = np.nanmax(
                        np.abs(
                            pa.values.astype(np.float64) - pb.values.astype(np.float64)
                        )
                    )
                    diffs.append(
                        Difference(
                            "Property", pa.title, "values", f"max diff = {max_diff:.2e}"
                        )
                    )


def _compare_surfaces(
    a: SurfaceSnapshot, b: SurfaceSnapshot, diffs: List[Difference], atol: float = 1e-10
):
    title = a.title
    if a.ni != b.ni or a.nj != b.nj:
        diffs.append(
            Difference(
                "Surface", title, "dimensions", f"({a.ni},{a.nj}) vs ({b.ni},{b.nj})"
            )
        )
        return
    for fname in ("origin_x", "origin_y", "di", "dj", "rotation"):
        va, vb = getattr(a, fname), getattr(b, fname)
        if abs(va - vb) > atol:
            diffs.append(Difference("Surface", title, fname, f"{va} vs {vb}"))
    if (
        a.values is not None
        and b.values is not None
        and not np.allclose(a.values, b.values, atol=atol, equal_nan=True)
    ):
        max_diff = np.nanmax(
            np.abs(a.values.astype(np.float64) - b.values.astype(np.float64))
        )
        diffs.append(
            Difference("Surface", title, "values", f"max diff = {max_diff:.2e}")
        )


def _compare_pointsets(
    a: PointSetSnapshot,
    b: PointSetSnapshot,
    diffs: List[Difference],
    atol: float = 1e-10,
):
    title = a.title
    if a.points is None or b.points is None:
        if (a.points is None) != (b.points is None):
            diffs.append(Difference("PointSet", title, "points", "one is None"))
        return
    if a.points.shape != b.points.shape:
        diffs.append(
            Difference(
                "PointSet", title, "shape", f"{a.points.shape} vs {b.points.shape}"
            )
        )
        return
    if not np.allclose(a.points, b.points, atol=atol, equal_nan=True):
        max_diff = np.nanmax(np.abs(a.points - b.points))
        diffs.append(
            Difference("PointSet", title, "points", f"max diff = {max_diff:.2e}")
        )


def _compare_polylinesets(
    a: PolylineSetSnapshot,
    b: PolylineSetSnapshot,
    diffs: List[Difference],
    atol: float = 1e-10,
):
    title = a.title
    if len(a.polylines) != len(b.polylines):
        diffs.append(
            Difference(
                "PolylineSet",
                title,
                "polyline_count",
                f"{len(a.polylines)} vs {len(b.polylines)}",
            )
        )
        return
    for i, (pa, pb) in enumerate(zip(a.polylines, b.polylines)):
        if pa.shape != pb.shape:
            diffs.append(
                Difference(
                    "PolylineSet",
                    title,
                    f"polyline[{i}].shape",
                    f"{pa.shape} vs {pb.shape}",
                )
            )
        elif not np.allclose(pa, pb, atol=atol, equal_nan=True):
            max_diff = np.nanmax(np.abs(pa - pb))
            diffs.append(
                Difference(
                    "PolylineSet", title, f"polyline[{i}]", f"max diff = {max_diff:.2e}"
                )
            )
    if a.closed != b.closed:
        diffs.append(
            Difference("PolylineSet", title, "closed", f"{a.closed} vs {b.closed}")
        )


def _compare_triangulated_surfaces(
    a: TriangulatedSurfaceSnapshot,
    b: TriangulatedSurfaceSnapshot,
    diffs: List[Difference],
    atol: float = 1e-10,
):
    title = a.title
    if a.vertices is None or b.vertices is None:
        if (a.vertices is None) != (b.vertices is None):
            diffs.append(
                Difference("TriangulatedSurface", title, "vertices", "one is None")
            )
        return
    if a.vertices.shape != b.vertices.shape:
        diffs.append(
            Difference(
                "TriangulatedSurface",
                title,
                "vertices.shape",
                f"{a.vertices.shape} vs {b.vertices.shape}",
            )
        )
        return
    if not np.allclose(a.vertices, b.vertices, atol=atol, equal_nan=True):
        max_diff = np.nanmax(np.abs(a.vertices - b.vertices))
        diffs.append(
            Difference(
                "TriangulatedSurface", title, "vertices", f"max diff = {max_diff:.2e}"
            )
        )
    if a.triangles is not None and b.triangles is not None:
        if not np.array_equal(a.triangles, b.triangles):
            diffs.append(
                Difference(
                    "TriangulatedSurface",
                    title,
                    "triangles",
                    f"shapes {a.triangles.shape} vs {b.triangles.shape}",
                )
            )


def _compare_wells(
    a: WellSnapshot,
    b: WellSnapshot,
    diffs: List[Difference],
    atol: float = 1e-10,
):
    title = a.title
    for fname in ("md", "x", "y", "z"):
        va, vb = getattr(a, fname), getattr(b, fname)
        if va is None or vb is None:
            if (va is None) != (vb is None):
                diffs.append(Difference("Well", title, fname, "one is None"))
            continue
        if va.shape != vb.shape:
            diffs.append(
                Difference("Well", title, fname, f"{va.shape} vs {vb.shape}")
            )
        elif not np.allclose(va, vb, atol=atol, equal_nan=True):
            max_diff = np.nanmax(np.abs(va - vb))
            diffs.append(
                Difference("Well", title, fname, f"max diff = {max_diff:.2e}")
            )


def _compare_blocked_wells(
    a: BlockedWellSnapshot,
    b: BlockedWellSnapshot,
    diffs: List[Difference],
    atol: float = 1e-10,
):
    title = a.title
    for fname in ("md", "x", "y", "z"):
        va, vb = getattr(a, fname), getattr(b, fname)
        if va is None or vb is None:
            if (va is None) != (vb is None):
                diffs.append(Difference("BlockedWell", title, fname, "one is None"))
            continue
        if va.shape != vb.shape:
            diffs.append(
                Difference("BlockedWell", title, fname, f"{va.shape} vs {vb.shape}")
            )
        elif not np.allclose(va, vb, atol=atol, equal_nan=True):
            max_diff = np.nanmax(np.abs(va - vb))
            diffs.append(
                Difference("BlockedWell", title, fname, f"max diff = {max_diff:.2e}")
            )
    for fname in ("cell_indices_i", "cell_indices_j", "cell_indices_k"):
        va, vb = getattr(a, fname), getattr(b, fname)
        if va is None or vb is None:
            if (va is None) != (vb is None):
                diffs.append(Difference("BlockedWell", title, fname, "one is None"))
            continue
        if not np.array_equal(va, vb):
            n_diff = np.sum(va.flatten() != vb.flatten())
            diffs.append(
                Difference("BlockedWell", title, fname, f"{n_diff} cells differ")
            )
