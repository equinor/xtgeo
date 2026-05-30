API Reference
=============

Complete function and class reference for the OSDU/RESQML interface,
organised by use case.

.. contents:: On this page
   :local:
   :depth: 2


Discovery & Search
------------------

Find and explore objects in a dataspace.

List objects
^^^^^^^^^^^^

.. autofunction:: xtgeo.list_osdu_objects

Search by name/type
^^^^^^^^^^^^^^^^^^^

.. autofunction:: xtgeo.search_osdu

Advanced query
^^^^^^^^^^^^^^

.. autofunction:: xtgeo.query_osdu

.. autofunction:: xtgeo.query_osdu_all_dataspaces

Import from URI
^^^^^^^^^^^^^^^

.. autofunction:: xtgeo.import_osdu


Deep Discovery & Graph Traversal
---------------------------------

Explore relationships between RESQML objects — traverse the full
dependency graph, filter by type, and follow edges.

.. autofunction:: xtgeo.deep_query_osdu

**Scope values:**

.. list-table::
   :header-rows: 1
   :widths: 20 60

   * - Scope
     - Meaning
   * - ``"targets"``
     - Objects this object references (e.g. grid → CRS)
   * - ``"sources"``
     - Objects that reference this object (e.g. properties → grid)
   * - ``"self"``
     - Only the object itself
   * - ``"targets_or_self"``
     - Targets + the starting object
   * - ``"sources_or_self"``
     - Sources + the starting object

**Common patterns:**

.. code-block:: python

    # All properties of a grid
    result = xtgeo.deep_query_osdu(session, uuid=grid_uuid, scope="sources")

    # The CRS of a grid
    result = xtgeo.deep_query_osdu(session, uuid=grid_uuid, scope="targets",
                                    object_types=["LocalDepth3dCrs"])

    # Full object tree
    result = xtgeo.deep_query_osdu(session, depth=0, include_edges=True)


Change Tracking & Notifications
-------------------------------

Monitor a dataspace for object changes using polling-based detection.

.. autofunction:: xtgeo.watch_osdu_changes

**Event format:**

Each event returned by ``.poll()`` is a dict:

.. code-block:: python

    {
        "event": "created" | "changed" | "deleted",
        "uuid": "...",
        "title": "...",
        "type": "resqml20.IjkGridRepresentation",
        "uri": "eml:///dataspace('...')/resqml20.IjkGrid...",
        "timestamp": 1716998400000000,  # microseconds
    }


Reading Objects
---------------

Grids
^^^^^

.. autofunction:: xtgeo.grid_from_osdu

Surfaces
^^^^^^^^

.. autofunction:: xtgeo.surface_from_osdu

Points
^^^^^^

.. autofunction:: xtgeo.points_from_osdu

Polygons
^^^^^^^^

.. autofunction:: xtgeo.polygons_from_osdu

Wells
^^^^^

.. autofunction:: xtgeo.well_from_osdu

Blocked Well
 Downloaded pandas
      Built xtgeo @ file:///home/runner/work/xtgeo/xtgeo
Uninstalled 7 packages in 53ms
Installed 7 packages in 32ms
I001 [*] Import block is un-sorted or un-formatted
   --> src/xtgeo/interfaces/osdu/__init__.py:42:1
    |
 40 |   # --- Providers ---
 41 |   # --- High-level user API ---
 42 | / from ._api import (
 43 | |     blocked_well_from_osdu,
 44 | |     blocked_well_to_osdu,
 45 | |     deep_query_osdu,
 46 | |     grid_from_osdu,
 47 | |     grid_to_osdu,
 48 | |     import_osdu,
 49 | |     list_osdu_dataspaces,
 50 | |     list_osdu_objects,
 51 | |     points_from_osdu,
 52 | |     points_to_osdu,
 53 | |     polygons_from_osdu,
 54 | |     polygons_to_osdu,
 55 | |     query_osdu,
 56 | |     query_osdu_all_dataspaces,
 57 | |     search_osdu,
 58 | |     surface_from_osdu,
 59 | |     surface_to_osdu,
 60 | |     triangulated_surface_from_osdu,
 61 | |     triangulated_surface_to_osdu,
 62 | |     watch_osdu_changes,
 63 | |     well_from_osdu,
 64 | |     well_to_osdu,
 65 | | )
 66 | |
 67 | | # --- CRS ---
 68 | | from ._crs import LocalDepth3dCrs
 69 | |
 70 | | # --- Dataspace operations ---
 71 | | from ._dataspace import (
 72 | |     BlockedWellSnapshot,
 73 | |     CrsSnapshot,
 74 | |     DataspaceSnapshot,
 75 | |     Difference,
 76 | |     GridSnapshot,
 77 | |     PointSetSnapshot,
 78 | |     PolylineSetSnapshot,
 79 | |     PropertySnapshot,
 80 | |     SurfaceSnapshot,
 81 | |     TriangulatedSurfaceSnapshot,
 82 | |     WellSnapshot,
 83 | |     compare_snapshots,
 84 | |     read_dataspace,
 85 | |     write_dataspace,
 86 | | )
 87 | | from ._epc_provider import EpcFileProvider
 88 | | from ._etp_provider import EtpConnectionConfig, EtpProvider
 89 | |
 90 | | # --- Converters ---
 91 | | from ._blocked_well import blocked_well_to_xtgeo, xtgeo_blocked_well_to_resqml
 92 | | from ._grid2d import grid2d_to_xtgeo, xtgeo_surface_to_resqml
 93 | | from ._ijk_grid import ijk_grid_to_xtgeo, xtgeo_grid_to_resqml
 94 | |
 95 | | # --- Metadata ---
 96 | | from ._metadata import (
 97 | |     OsduPropertyMapping,
 98 | |     OsduWorkProductMetadata,
 99 | |     ecl_keyword_to_osdu,
100 | |     list_supported_properties,
101 | |     osdu_name_to_ecl_keyword,
102 | |     osdu_reference_to_mapping,
103 | |     resolve_property_mapping,
104 | | )
105 | | from ._pointset import pointset_to_xtgeo, xtgeo_points_to_resqml
106 | | from ._polyline import polylineset_to_xtgeo, xtgeo_polygons_to_resqml
107 | | from ._properties import read_grid_properties, write_grid_property
108 | | from ._provider_base import ResqmlDataProvider
109 | | from ._triangulated_surface import (
110 | |     triangulated_surface_to_xtgeo,
111 | |     xtgeo_triangulated_surface_to_resqml,
112 | | )
113 | | from ._well import well_to_xtgeo, xtgeo_well_to_resqml
114 | |
115 | | # --- Enums ---
116 | | from ._resqml_enums import (
117 | |     CellShape,
118 | |     Handedness,
119 | |     IndexableElement,
120 | |     KDirection,
121 | |     PropertyKind,
122 | |     ResqmlObjectType,
123 | | )
124 | |
125 | | # --- Session ---
126 | | from ._session import OsduSession
    | |_________________________________^
127 |
128 |   __all__ = [
    |
help: Organize imports

SIM102 Use a single `if` statement instead of nested `if` statements
    --> src/xtgeo/interfaces/osdu/_dataspace.py:1148:5
     |
1146 |               )
1147 |           )
1148 | /     if a.triangles is not None and b.triangles is not None:
1149 | |         if not np.array_equal(a.triangles, b.triangles):
     | |________________________________________________________^
1150 |               diffs.append(
1151 |                   Difference(
     |
help: Combine `if` statements using `and`

F401 [*] `._resqml_enums.NS_COMMON20` imported but unused
    --> src/xtgeo/interfaces/osdu/_etp_provider.py:2482:36
     |
2480 |         from lxml import etree
2481 |
2482 |         from ._resqml_enums import NS_COMMON20, NS_RESQML20, RESQML_NS_MAP
     |                                    ^^^^^^^^^^^
2483 |
2484 |         qualified_type = "resqml20.WellboreFeature"
     |
help: Remove unused import: `._resqml_enums.NS_COMMON20`

F401 [*] `._resqml_enums.NS_COMMON20` imported but unused
    --> src/xtgeo/interfaces/osdu/_etp_provider.py:2505:36
     |
2503 |         from lxml import etree
2504 |
2505 |         from ._resqml_enums import NS_COMMON20, NS_RESQML20, RESQML_NS_MAP
     |                                    ^^^^^^^^^^^
2506 |
2507 |         qualified_type = "resqml20.WellboreInterpretation"
     |
help: Remove unused import: `._resqml_enums.NS_COMMON20`

F401 [*] `._resqml_enums.NS_COMMON20` imported but unused
    --> src/xtgeo/interfaces/osdu/_etp_provider.py:2533:36
     |
2531 |         from lxml import etree
2532 |
2533 |         from ._resqml_enums import NS_COMMON20, NS_RESQML20, RESQML_NS_MAP
     |                                    ^^^^^^^^^^^
2534 |
2535 |         qualified_type = "resqml20.obj_MdDatum"
     |
help: Remove unused import: `._resqml_enums.NS_COMMON20`

I001 [*] Import block is un-sorted or un-formatted
  --> tests/test_interfaces/test_osdu/test_epc_compliance.py:14:1
   |
12 |   """
13 |
14 | / import pathlib
15 | | import tempfile
16 | |
17 | | import numpy as np
18 | | import pandas as pd
19 | | import pytest
20 | | from hypothesis import given, settings
21 | |
22 | | import xtgeo
23 | | from xtgeo.interfaces.osdu import EpcFileProvider
24 | | from xtgeo.interfaces.osdu._grid2d import grid2d_to_xtgeo, xtgeo_surface_to_resqml
25 | | from xtgeo.interfaces.osdu._ijk_grid import ijk_grid_to_xtgeo, xtgeo_grid_to_resqml
26 | | from xtgeo.interfaces.osdu._pointset import pointset_to_xtgeo, xtgeo_points_to_resqml
27 | | from xtgeo.interfaces.osdu._polyline import (
28 | |     polylineset_to_xtgeo,
29 | |     xtgeo_polygons_to_resqml,
30 | | )
31 | | from xtgeo.interfaces.osdu._triangulated_surface import (
32 | |     triangulated_surface_to_xtgeo,
33 | |     xtgeo_triangulated_surface_to_resqml,
34 | | )
35 | | from xtgeo.interfaces.osdu._well import well_to_xtgeo, xtgeo_well_to_resqml
36 | | from xtgeo.interfaces.osdu._blocked_well import (
37 | |     blocked_well_to_xtgeo,
38 | |     xtgeo_blocked_well_to_resqml,
39 | | )
   | |_^
   |
help: Organize imports

I001 [*] Import block is un-sorted or un-formatted
   --> tests/test_interfaces/test_osdu/test_osdu_unit.py:921:1
    |
919 |   # ---------------------------------------------------------------------------
920 |
921 | / from xtgeo.interfaces.osdu import (  # noqa: E402, F811
922 | |     grid_from_osdu,
923 | |     grid_to_osdu,
924 | |     points_from_osdu,
925 | |     points_to_osdu,
926 | |     polygons_from_osdu,
927 | |     polygons_to_osdu,
928 | |     surface_from_osdu,
929 | |     surface_to_osdu,
930 | | )
931 | | from xtgeo.interfaces.osdu._resqml_meta import _get_resqml_meta, _set_resqml_meta  # noqa: E402, F811
    | |_________________________________________________________________________________^
    |
help: Organize imports

F401 [*] `typing.List` imported but unused
  --> tests/test_interfaces/test_osdu/test_rddms.py:17:31
   |
15 | import uuid as _uuid
16 | from pathlib import Path
17 | from typing import Any, Dict, List
   |                               ^^^^
18 |
19 | import numpy as np
   |
help: Remove unused import: `typing.List`

E501 Line too long (99 > 88)
   --> tests/test_interfaces/test_osdu/test_rddms.py:299:89
    |
297 |                 n_diff = np.sum(a != b)
298 |                 print(
299 |                     f"  [{label}] {name}: DIFFER max_diff={max_diff:.2e}, n_diff={n_diff}/{a.size}"
    |                                                                                         ^^^^^^^^^^^
300 |                 )
301 |                 raise AssertionError(f"{name} not bitwise identical in {label}")
    |

E501 Line too long (99 > 88)
   --> tests/test_interfaces/test_osdu/test_rddms.py:306:89
    |
304 |                 n_diff = np.sum(a != b)
305 |                 print(
306 |                     f"  [{label}] {name}: DIFFER max_diff={max_diff:.2e}, n_diff={n_diff}/{a.size}"
    |                                                                                         ^^^^^^^^^^^
307 |                 )
308 |                 np.testing.assert_allclose(
    |

SIM105 Use `contextlib.suppress(Exception)` instead of `try`-`except`-`pass`
   --> tests/test_interfaces/test_osdu/test_rddms.py:486:5
    |
484 |       p_wr = EtpProvider(cfg_dst)
485 |       p_wr.open()
486 | /     try:
487 | |         p_wr.put_dataspace(target_ds)
488 | |     except Exception:
489 | |         pass
    | |____________^
490 |
491 |       # ===== Cycle 1: Write → Read =====
    |
help: Replace `try`-`except`-`pass` with `with contextlib.suppress(Exception): ...`

F541 [*] f-string without any placeholders
   --> tests/test_interfaces/test_osdu/test_rddms.py:519:11
    |
517 |     # ===== Compare =====
518 |     print(f"\n{'=' * 60}")
519 |     print(f"COMPARE: cycle0 vs cycle1 (format conversion tolerance)")
    |           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
520 |     print(f"{'=' * 60}")
521 |     ok_01 = _compare_snapshots(snap_0, snap_1, "cycle0↔cycle1", strict=False)
    |
help: Remove extraneous `f` prefix

F541 [*] f-string without any placeholders
   --> tests/test_interfaces/test_osdu/test_rddms.py:524:11
    |
523 |     print(f"\n{'=' * 60}")
524 |     print(f"COMPARE: cycle1 vs cycle2 (MUST BE BITWISE IDENTICAL)")
    |           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
525 |     print(f"{'=' * 60}")
526 |     ok_12 = _compare_snapshots(snap_1, snap_2, "cycle1↔cycle2", strict=True)
    |
help: Remove extraneous `f` prefix

SIM105 Use `contextlib.suppress(Exception)` instead of `try`-`except`-`pass`
   --> tests/test_interfaces/test_osdu/test_rddms.py:588:5
    |
586 |       p0 = EtpProvider(cfg)
587 |       p0.open()
588 | /     try:
589 | |         p0.put_dataspace(target_ds)
590 | |     except Exception:
591 | |         pass
    | |____________^
592 |       p0.put_crs(crs_uuid, "SyntheticCRS", 0, 0, 0, 0, True)
    |
help: Replace `try`-`except`-`pass` with `with contextlib.suppress(Exception): ...`

E501 Line too long (89 > 88)
   --> tests/test_interfaces/test_osdu/test_rddms.py:623:89
    |
621 |     print(
622 |         f"  Cycle 2: BITWISE IDENTICAL "
623 |         f"({tri2['vertices'].shape[0]} vertices, {tri2['triangles'].shape[0]} triangles)"
    |                                                                                         ^
624 |     )
625 |     print("  PASS: Synthetic TriangulatedSet double roundtrip BITWISE IDENTICAL")
    |

F811 Redefinition of unused `etp_config` from line 674
    --> tests/test_interfaces/test_osdu/test_rddms.py:1096:5
     |
1095 | @pytest.fixture
1096 | def etp_config():
     |     ^^^^^^^^^^ `etp_config` redefined here
1097 |     """Base ETP config for local RDDMS."""
1098 |     import uuid as _uuid
     |
    ::: tests/test_interfaces/test_osdu/test_rddms.py:674:5
     |
 673 | @pytest.fixture
 674 | def etp_config():
     |     ---------- previous definition of `etp_config` here
 675 |     """Config with a unique test dataspace."""
 676 |     ds_path = f"xtgeo/test_disc_{_uuid.uuid4().hex[:8]}"
     |
help: Remove definition: `etp_config`

F811 Redefinition of unused `provider` from line 684
    --> tests/test_interfaces/test_osdu/test_rddms.py:1108:5
     |
1107 | @pytest.fixture
1108 | def provider(etp_config):
     |     ^^^^^^^^ `provider` redefined here
1109 |     """ETP provider with a fresh test dataspace."""
1110 |     cfg, ds_path = etp_config
     |
    ::: tests/test_interfaces/test_osdu/test_rddms.py:684:5
     |
 683 | @pytest.fixture
 684 | def provider(etp_config):
     |     -------- previous definition of `provider` here
 685 |     """ETP provider with a fresh test dataspace."""
 686 |     cfg, ds_path = etp_config
     |
help: Remove definition: `provider`

Found 17 errors.
[*] 9 fixable with the `--fix` option.
Error: Process completed with exit code 1.
0s
0s
1s
0s
0s

^^^^^^^^^^^^^

.. autofunction:: xtgeo.blocked_well_from_osdu

Triangulated Surfaces
^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: xtgeo.triangulated_surface_from_osdu


Writing Objects
---------------

Grids
^^^^^

.. autofunction:: xtgeo.grid_to_osdu

Surfaces
^^^^^^^^

.. autofunction:: xtgeo.surface_to_osdu

Points
^^^^^^

.. autofunction:: xtgeo.points_to_osdu

Polygons
^^^^^^^^

.. autofunction:: xtgeo.polygons_to_osdu

Wells
^^^^^

.. autofunction:: xtgeo.well_to_osdu

Blocked Wells
^^^^^^^^^^^^^

.. autofunction:: xtgeo.blocked_well_to_osdu

Triangulated Surfaces
^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: xtgeo.triangulated_surface_to_osdu


Dataspace Management
--------------------

List dataspaces
^^^^^^^^^^^^^^^

.. autofunction:: xtgeo.list_osdu_dataspaces

Bulk operations
^^^^^^^^^^^^^^^

.. autofunction:: xtgeo.interfaces.osdu.read_dataspace

.. autofunction:: xtgeo.interfaces.osdu.write_dataspace

.. autofunction:: xtgeo.interfaces.osdu.compare_snapshots

.. autoclass:: xtgeo.interfaces.osdu.DataspaceSnapshot
   :members:


Session & Authentication
------------------------

.. autoclass:: xtgeo.interfaces.osdu.OsduSession
   :members: access_token, etp_config, create_dataspace_rest, create_dataspace_etp,
             list_dataspaces, get_dataspace, delete_dataspace, list_objects_rest,
             search_objects_rest, get_object_metadata_rest, switch_dataspace,
             save, load, from_env, list_profiles


Providers (Low-level)
---------------------

These are the backend implementations. Most users should use the high-level
functions above; providers are useful for advanced or custom workflows.

ETP Provider
^^^^^^^^^^^^

.. autoclass:: xtgeo.interfaces.osdu.EtpProvider
   :members: open, close, list_objects, discover, get_related_objects,
             get_deleted_resources, subscribe_notifications

.. autoclass:: xtgeo.interfaces.osdu.EtpConnectionConfig
   :members:

EPC File Provider
^^^^^^^^^^^^^^^^^

.. autoclass:: xtgeo.interfaces.osdu.EpcFileProvider
   :members: open, close, list_objects


Low-level Converters
--------------------

Direct converter functions for custom pipelines.

IJK Grid
^^^^^^^^

.. autofunction:: xtgeo.interfaces.osdu.ijk_grid_to_xtgeo

.. autofunction:: xtgeo.interfaces.osdu.xtgeo_grid_to_resqml

Grid2D (Surface)
^^^^^^^^^^^^^^^^

.. autofunction:: xtgeo.interfaces.osdu.grid2d_to_xtgeo

.. autofunction:: xtgeo.interfaces.osdu.xtgeo_surface_to_resqml

PointSet
^^^^^^^^

.. autofunction:: xtgeo.interfaces.osdu.pointset_to_xtgeo

.. autofunction:: xtgeo.interfaces.osdu.xtgeo_points_to_resqml

PolylineSet
^^^^^^^^^^^^

.. autofunction:: xtgeo.interfaces.osdu.polylineset_to_xtgeo

.. autofunction:: xtgeo.interfaces.osdu.xtgeo_polygons_to_resqml

TriangulatedSet
^^^^^^^^^^^^^^^

.. autofunction:: xtgeo.interfaces.osdu.triangulated_surface_to_xtgeo

.. autofunction:: xtgeo.interfaces.osdu.xtgeo_triangulated_surface_to_resqml

Well (Trajectory + Logs)
^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: xtgeo.interfaces.osdu.well_to_xtgeo

.. autofunction:: xtgeo.interfaces.osdu.xtgeo_well_to_resqml

BlockedWell
^^^^^^^^^^^

.. autofunction:: xtgeo.interfaces.osdu.blocked_well_to_xtgeo

.. autofunction:: xtgeo.interfaces.osdu.xtgeo_blocked_well_to_resqml


CRS & Metadata
--------------

.. autoclass:: xtgeo.interfaces.osdu.LocalDepth3dCrs
   :members:

.. autofunction:: xtgeo.interfaces.osdu.resolve_property_mapping

.. autofunction:: xtgeo.interfaces.osdu.ecl_keyword_to_osdu

.. autofunction:: xtgeo.interfaces.osdu.osdu_name_to_ecl_keyword


Enumerations
------------

.. autoclass:: xtgeo.interfaces.osdu.ResqmlObjectType
   :members:

.. autoclass:: xtgeo.interfaces.osdu.PropertyKind
   :members:

.. autoclass:: xtgeo.interfaces.osdu.CellShape
   :members:

.. autoclass:: xtgeo.interfaces.osdu.IndexableElement
   :members:
