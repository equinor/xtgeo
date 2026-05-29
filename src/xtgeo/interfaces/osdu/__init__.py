# -*- coding: utf-8 -*-
"""XTGeo OSDU/RESQML 2.0.1 interface package.

Provides reading and writing of RESQML 2.0.1 format for all xtgeo data types:
  - IJK Grid Representations (corner-point grids)
  - Grid2D Representations (regular surfaces / maps)
  - PointSet Representations (point clouds)
  - PolylineSet Representations (polygons / fault sticks)
  - Grid properties (porosity, permeability, saturation, etc.)

Two I/O backends are supported:
  1. **EPC + HDF5 file containers** (like resqpy) - for file-based workflows
  2. **ETP 1.2 WebSocket protocol** - for live connection to OSDU Reservoir DMS (RDDMS)

Usage Examples
--------------
File-based (EPC + H5):

    >>> from xtgeo.interfaces.osdu import EpcFileProvider, ijk_grid_to_xtgeo
    >>> with EpcFileProvider("model.epc", mode="r") as provider:
    ...     grid, props = ijk_grid_to_xtgeo(provider, grid_uuid="...")

    >>> from xtgeo.interfaces.osdu import EpcFileProvider, xtgeo_grid_to_resqml
    >>> with EpcFileProvider("output.epc", mode="w") as provider:
    ...     uuids = xtgeo_grid_to_resqml(provider, grid, title="MyGrid")

ETP protocol (RDDMS):

    >>> from xtgeo.interfaces.osdu import EtpProvider, EtpConnectionConfig
    >>> config = EtpConnectionConfig(
    ...     url="wss://host/api/reservoir-ddms-etp/v2/",
    ...     token="<bearer>",
    ...     dataspace="eml:///dataspace('myproject')",
    ... )
    >>> with EtpProvider(config) as provider:
    ...     objects = provider.list_objects("IjkGrid")
    ...     grid, props = ijk_grid_to_xtgeo(provider, objects[0]["uuid"])
"""

# --- Providers ---
# --- High-level user API ---
from ._api import (
    blocked_well_from_osdu,
    blocked_well_to_osdu,
    deep_query_osdu,
    grid_from_osdu,
    grid_to_osdu,
    import_osdu,
    list_osdu_dataspaces,
    list_osdu_objects,
    points_from_osdu,
    points_to_osdu,
    polygons_from_osdu,
    polygons_to_osdu,
    query_osdu,
    query_osdu_all_dataspaces,
    search_osdu,
    surface_from_osdu,
    surface_to_osdu,
    triangulated_surface_from_osdu,
    triangulated_surface_to_osdu,
    watch_osdu_changes,
    well_from_osdu,
    well_to_osdu,
)

# --- CRS ---
from ._crs import LocalDepth3dCrs

# --- Dataspace operations ---
from ._dataspace import (
    BlockedWellSnapshot,
    CrsSnapshot,
    DataspaceSnapshot,
    Difference,
    GridSnapshot,
    PointSetSnapshot,
    PolylineSetSnapshot,
    PropertySnapshot,
    SurfaceSnapshot,
    TriangulatedSurfaceSnapshot,
    WellSnapshot,
    compare_snapshots,
    read_dataspace,
    write_dataspace,
)
from ._epc_provider import EpcFileProvider
from ._etp_provider import EtpConnectionConfig, EtpProvider

# --- Converters ---
from ._blocked_well import blocked_well_to_xtgeo, xtgeo_blocked_well_to_resqml
from ._grid2d import grid2d_to_xtgeo, xtgeo_surface_to_resqml
from ._ijk_grid import ijk_grid_to_xtgeo, xtgeo_grid_to_resqml

# --- Metadata ---
from ._metadata import (
    OsduPropertyMapping,
    OsduWorkProductMetadata,
    ecl_keyword_to_osdu,
    list_supported_properties,
    osdu_name_to_ecl_keyword,
    osdu_reference_to_mapping,
    resolve_property_mapping,
)
from ._pointset import pointset_to_xtgeo, xtgeo_points_to_resqml
from ._polyline import polylineset_to_xtgeo, xtgeo_polygons_to_resqml
from ._properties import read_grid_properties, write_grid_property
from ._provider_base import ResqmlDataProvider
from ._triangulated_surface import (
    triangulated_surface_to_xtgeo,
    xtgeo_triangulated_surface_to_resqml,
)
from ._well import well_to_xtgeo, xtgeo_well_to_resqml

# --- Enums ---
from ._resqml_enums import (
    CellShape,
    Handedness,
    IndexableElement,
    KDirection,
    PropertyKind,
    ResqmlObjectType,
)

# --- Session ---
from ._session import OsduSession

__all__ = [
    # Providers
    "EpcFileProvider",
    "EtpProvider",
    "EtpConnectionConfig",
    "ResqmlDataProvider",
    # Session
    "OsduSession",
    # CRS
    "LocalDepth3dCrs",
    # Metadata
    "OsduPropertyMapping",
    "OsduWorkProductMetadata",
    "ecl_keyword_to_osdu",
    "list_supported_properties",
    "osdu_name_to_ecl_keyword",
    "osdu_reference_to_mapping",
    "resolve_property_mapping",
    # Enums
    "CellShape",
    "Handedness",
    "IndexableElement",
    "KDirection",
    "PropertyKind",
    "ResqmlObjectType",
    # IJK Grid converters
    "ijk_grid_to_xtgeo",
    "xtgeo_grid_to_resqml",
    # Surface converters
    "grid2d_to_xtgeo",
    "xtgeo_surface_to_resqml",
    # PointSet converters
    "pointset_to_xtgeo",
    "xtgeo_points_to_resqml",
    # Polygon converters
    "polylineset_to_xtgeo",
    "xtgeo_polygons_to_resqml",
    # TriangulatedSurface converters
    "triangulated_surface_to_xtgeo",
    "xtgeo_triangulated_surface_to_resqml",
    # Well converters
    "well_to_xtgeo",
    "xtgeo_well_to_resqml",
    # BlockedWell converters
    "blocked_well_to_xtgeo",
    "xtgeo_blocked_well_to_resqml",
    # Property converters
    "read_grid_properties",
    "write_grid_property",
    # Dataspace operations
    "DataspaceSnapshot",
    "GridSnapshot",
    "SurfaceSnapshot",
    "PointSetSnapshot",
    "PolylineSetSnapshot",
    "TriangulatedSurfaceSnapshot",
    "WellSnapshot",
    "BlockedWellSnapshot",
    "PropertySnapshot",
    "CrsSnapshot",
    "Difference",
    "read_dataspace",
    "write_dataspace",
    "compare_snapshots",
    # High-level user API
    "grid_from_osdu",
    "grid_to_osdu",
    "surface_from_osdu",
    "surface_to_osdu",
    "points_from_osdu",
    "points_to_osdu",
    "polygons_from_osdu",
    "polygons_to_osdu",
    "well_from_osdu",
    "well_to_osdu",
    "blocked_well_from_osdu",
    "blocked_well_to_osdu",
    "triangulated_surface_from_osdu",
    "triangulated_surface_to_osdu",
    "list_osdu_objects",
    "list_osdu_dataspaces",
    "search_osdu",
    "query_osdu",
    "query_osdu_all_dataspaces",
    "import_osdu",
    "deep_query_osdu",
    "watch_osdu_changes",
]
