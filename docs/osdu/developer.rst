Developer Guide
===============

Architecture, data model details, protocol internals, testing, and
contributor guidance.

.. contents:: On this page
   :local:
   :depth: 2


Architecture
------------

The package is layered to separate concerns and enable testing at each level:

.. code-block:: text

    ┌─────────────────────────────────────────────────────────────────────┐
    │                       USER API LAYER                                 │
    │  grid_from_osdu · surface_from_osdu · deep_query_osdu · ...         │
    │  All accept (session_or_path, ...) → xtgeo objects                  │
    └───────────────────────────────┬─────────────────────────────────────┘
                                    │
    ┌───────────────────────────────▼─────────────────────────────────────┐
    │                    CONVERTER LAYER                                    │
    │  _ijk_grid.py · _grid2d.py · _pointset.py · _polyline.py           │
    │  _properties.py · _crs.py · _metadata.py                            │
    │  Pure logic: xtgeo ↔ RESQML geometry/property transforms            │
    └───────────────────────────────┬─────────────────────────────────────┘
                                    │
    ┌───────────────────────────────▼─────────────────────────────────────┐
    │                    PROVIDER ABSTRACTION                               │
    │  ResqmlDataProvider (ABC)                                            │
    │    ├─ EpcFileProvider  (EPC ZIP + HDF5 file I/O)                    │
    │    └─ EtpProvider      (ETP 1.2 WebSocket binary protocol)          │
    │  Common interface: get/put geometry, properties, list objects         │
    └───────────────────────────────┬─────────────────────────────────────┘
                                    │
    ┌───────────────────────────────▼─────────────────────────────────────┐
    │                    SESSION & AUTH                                     │
    │  OsduSession – OAuth2, profiles, REST admin wrappers                 │
    │  EtpConnectionConfig – WebSocket connection parameters               │
    └─────────────────────────────────────────────────────────────────────┘


Module Inventory
----------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Module
     - Responsibility
   * - ``__init__.py``
     - Package exports and usage docstring
   * - ``_api.py``
     - User-facing functions (grid_from_osdu, deep_query_osdu, watch_osdu_changes, etc.)
   * - ``_session.py``
     - OsduSession: auth, profile persistence, REST admin wrappers
   * - ``_provider_base.py``
     - ABC defining the provider interface
   * - ``_epc_provider.py``
     - EPC+HDF5 file provider (zipfile + h5py)
   * - ``_etp_provider.py``
     - ETP 1.2 WebSocket provider (energistics library + asyncio)
   * - ``_ijk_grid.py``
     - IJK Grid ↔ xtgeo Grid converter
   * - ``_grid2d.py``
     - Grid2D ↔ xtgeo RegularSurface converter
   * - ``_pointset.py``
     - PointSet ↔ xtgeo Points converter
   * - ``_polyline.py``
     - PolylineSet ↔ xtgeo Polygons converter
   * - ``_triangulated_surface.py``
     - TriangulatedSetRepresentation ↔ xtgeo TriangulatedSurface converter
   * - ``_well.py``
     - WellboreTrajectory + WellboreFrame ↔ xtgeo Well converter
   * - ``_blocked_well.py``
     - BlockedWellboreRepresentation ↔ xtgeo BlockedWell converter
   * - ``_properties.py``
     - Grid property read/write (multi-property support)
   * - ``_crs.py``
     - LocalDepth3dCrs handling
   * - ``_metadata.py``
     - OSDU property name mapping (Eclipse ↔ OSDU)
   * - ``_resqml_enums.py``
     - RESQML enumeration types
   * - ``_resqml_meta.py``
     - Metadata attachment helpers (UUID preservation on xtgeo objects)
   * - ``_dataspace.py``
     - Bulk dataspace operations (read/write/compare entire datasets)


Design Principles
-----------------

Provider Abstraction
^^^^^^^^^^^^^^^^^^^^

All data access goes through ``ResqmlDataProvider``. This enables:

- Swapping between file and network backends transparently
- Testing converters against EPC files without a live server
- Future backends (e.g., REST-only OSDU v3)

.. code-block:: python

    class ResqmlDataProvider(ABC):
        def open(self) -> None: ...
        def close(self) -> None: ...

        # Discovery
        def list_objects(self, object_type=None) -> List[Dict]: ...

        # Geometry (get/put per object type)
        def get_ijk_grid_geometry(self, uuid) -> Dict[str, np.ndarray]: ...
        def put_ijk_grid(self, uuid, title, xml, arrays) -> None: ...
        def get_grid2d_geometry(self, uuid) -> Dict[str, Any]: ...
        # ... etc.

Exact Geometry Preservation
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The converter layer guarantees bit-exact roundtrips:

- Corner-point pillar coordinates: ``coordsv`` shape ``(ni+1, nj+1, 6)`` — float64
- Cell corners: ``zcornsv`` shape ``(ni+1, nj+1, nk+1, 4)`` — float64
- Activity mask: ``actnumsv`` shape ``(ni, nj, nk)`` — int32
- No interpolation, no resampling, no precision loss

Property Axis Convention
^^^^^^^^^^^^^^^^^^^^^^^^

- **xtgeo**: properties shaped ``(ni, nj, nk)`` in IJK (column-major) order
- **RESQML/resqpy**: properties in ``(nk, nj, ni)`` KJI (row-major) order
- Conversion: ``values.reshape(nk, nj, ni).transpose(2, 1, 0)``
- Verified to machine precision in all roundtrip tests

UUID Preservation
^^^^^^^^^^^^^^^^^

All converters follow the same UUID pattern:

1. On **write**: check if the xtgeo object has RESQML metadata attached
   (via ``_resqml_meta.py``). If a UUID exists, reuse it (update-in-place).
   Otherwise generate a new ``uuid4()``.
2. On **read**: attach the RESQML UUID, CRS UUID, and other references to
   the xtgeo object's metadata dict, so subsequent writes preserve identity.
3. **Cross-references** (CRS, HDF proxy, Feature/Interpretation links) use
   UUID attributes on child elements — consistent across all types.

CRS Handling
^^^^^^^^^^^^

All 8 write converters share the same CRS creation pattern:

- If no ``crs_uuid`` is passed, create a ``LocalDepth3dCrs`` with the given
  EPSG code and default origin ``(0, 0, 0)``
- CRS is referenced from within the object's ``Geometry`` element
- One CRS per EPC/dataspace is typical (shared across objects)

Metadata & Citation
^^^^^^^^^^^^^^^^^^^

All types write a ``Citation`` element with ``Title``. The title is the
user-supplied ``title=`` argument. Additional Citation fields (``Originator``,
``Creation``, ``Description``) are not currently written but are preserved
if present when reading third-party files.

resqpy Interoperability
^^^^^^^^^^^^^^^^^^^^^^^

The module follows resqpy conventions where possible:

- **HDF5 per-patch naming**: ``points_patch0``, ``triangles_patch0`` for
  TriangulatedSets (resqpy's standard naming)
- **WellboreFeature → WellboreInterpretation → Representation** chain
  (required for resqpy/Petrel discovery)
- **SurfaceRole** element on TriangulatedSets
- **Read fallback**: readers try per-patch names first, then fall back to
  generic names for backward compatibility with older files


Data Model: xtgeo vs RESQML
----------------------------

This section documents the key structural differences between xtgeo's internal
data model and the RESQML 2.0.1 standard. Understanding these is essential for
contributors working on converters or for interoperability with other tools.

IJK Grid Geometry
^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Aspect
     - xtgeo
     - RESQML 2.0.1
   * - Coordinate storage
     - ``_coordsv``: ``(ni+1, nj+1, 6)`` — one pillar line per node (top+bottom XYZ)
     - ``Points``: ``(nk+1, nj+1, ni+1, 3)`` — explicit XYZ at every K-layer boundary
   * - Z-corners
     - ``_zcornsv``: ``(ni+1, nj+1, nk+1, 4)`` — 4 values per pillar node per layer
     - Embedded in the 3D Points array as the Z-coordinate
   * - Axis order
     - IJK (column-major): I fastest, K slowest
     - KJI (row-major): K outermost, I innermost
   * - Handedness
     - Implicit (depends on grid construction)
     - Explicit via ``GridIsRighthanded`` flag
   * - Split pillars
     - **Not supported** — ``_coordsv`` stores one pillar per node; true XY-splits
       cannot be represented
     - Full support via ``PillarIndices`` + ``ColumnsPerSplitCoordinateLine`` +
       ``SplitCoordinateLines`` arrays

**Split pillar limitation:** xtgeo's internal geometry stores one pillar line
per ``(i, j)`` node. This means true XY-split pillars (where different cells
sharing a pillar node have different X/Y coordinates) are structurally
impossible. Only Z-discontinuities (faults with vertical throw) are
representable. When exporting to RESQML, a warning is emitted if
Z-discontinuities are detected at interior pillar nodes, since the RESQML
output uses unsplit pillar geometry:

.. code-block:: text

    WARNING: Grid has Z‑discontinuities at N interior pillar nodes that
    suggest faulted geometry. The exported RESQML uses unsplit pillar
    coordinates — lateral fault geometry (XY‑split pillars) is not
    representable in the xtgeo data model.

K-direction gaps
^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Aspect
     - xtgeo
     - RESQML
   * - Layer connectivity
     - Layers share top/bottom surfaces (forced connected on import)
     - ``KGaps`` flag + boolean array allow vertical separation between layers
   * - Data loss
     - K-gaps collapsed on Eclipse GRDECL import, with warning
     - Preserved natively

Properties
^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Aspect
     - xtgeo
     - RESQML
   * - Value type
     - ``GridProperty.values``: masked float64 array
     - ``ContinuousProperty`` (float64) or ``DiscreteProperty`` (int32)
   * - Inactive cells
     - ``np.ma.MaskedArray`` — masked entries for ACTNUM=0
     - Separate ``PatchOfValues`` + supporting representation activity mask
   * - Name mapping
     - Eclipse keyword (``PORO``, ``PERMX``, etc.) or free-form
     - OSDU PropertyNameType reference URI + human-readable name
   * - Time series
     - Not directly supported (each timestep is a separate ``GridProperty``)
     - ``TimeIndex`` + ``TimeSeries`` objects for temporal properties
   * - Realizations
     - Not supported as a first-class concept
     - ``RealizationIndex`` on property patches

Surfaces
^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Aspect
     - xtgeo
     - RESQML
   * - Type
     - ``RegularSurface`` — regular 2D grid
     - ``Grid2dRepresentation`` with ``Grid2dPatch``
   * - Origin
     - ``xori``, ``yori`` (lower-left corner)
     - ``Origin`` XYZ point (may differ from lower-left depending on axis directions)
   * - Offset axes
     - ``xinc``, ``yinc``, ``rotation`` (rotation in degrees from north)
     - Two ``Point3d`` offset vectors with spacing + count (explicit direction)
   * - Missing values
     - ``np.nan`` in masked arrays
     - NaN in HDF5 arrays (same convention)

Points & Polygons
^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Aspect
     - xtgeo
     - RESQML
   * - Points storage
     - DataFrame with X, Y, Z columns
     - ``PointSetRepresentation`` with HDF5 array ``(N, 3)``
   * - Polygon storage
     - DataFrame with X, Y, Z, POLY_ID columns
     - ``PolylineSetRepresentation`` with ``NodeCountPerPolyline`` int array +
       concatenated coordinates ``(total_nodes, 3)``
   * - Closure
     - Not explicit (user convention)
     - ``ArePatched`` flag indicates closed vs open polylines

Wells
^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Aspect
     - xtgeo
     - RESQML
   * - Trajectory storage
     - DataFrame with X_UTME, Y_UTMN, Z_TVDSS, MD columns
     - ``WellboreTrajectoryRepresentation`` with MD + XYZ HDF5 arrays
   * - Well logs
     - Additional DataFrame columns (GR, PORO, etc.)
     - ``WellboreFrameRepresentation`` with log properties
   * - Object hierarchy
     - Flat (single Well object)
     - ``WellboreFeature`` → ``WellboreInterpretation`` → ``Trajectory``
   * - Blocked well
     - ``BlockedWell`` with I/J/K index columns
     - ``BlockedWellboreRepresentation`` with cell index arrays
   * - MD datum
     - Implicit (first MD value)
     - Explicit ``MdDatum`` object (not yet written)

Triangulated Surfaces
^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Aspect
     - xtgeo
     - RESQML
   * - Vertex storage
     - ``vertices``: ``(N, 3)`` float64 array
     - ``points_patch0`` HDF5 dataset (per-patch naming)
   * - Triangle storage
     - ``triangles``: ``(M, 3)`` int32 array
     - ``triangles_patch0`` HDF5 dataset (per-patch naming)
   * - Surface role
     - Not explicit
     - ``SurfaceRole`` element (``"map"`` default)
   * - Multi-patch
     - Single patch only
     - Schema supports multiple patches (read fallback supported)


Supported vs Unsupported RESQML Features
-----------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 35 15 50

   * - Feature
     - Status
     - Notes
   * - IJK Grid (explicit Points array)
     - Full
     - Read + write, exact roundtrip
   * - Grid properties (Continuous + Discrete)
     - Full
     - Float64 continuous, int32 discrete, multi-property
   * - Grid2d (regular surface)
     - Full
     - Origin, increments, rotation preserved
   * - PointSet
     - Full
     - N-point arrays with XYZ
   * - PolylineSet
     - Full
     - Multiple polylines per representation
   * - TriangulatedSetRepresentation
     - Full
     - Vertices + triangle indices, per-patch HDF5 naming (resqpy-compatible)
   * - WellboreTrajectoryRepresentation
     - Full
     - MD + XYZ arrays, WellboreFeature → Interpretation chain auto-created
   * - WellboreFrameRepresentation
     - Full
     - Well logs stored as frame properties
   * - BlockedWellboreRepresentation
     - Full
     - Cell I/J/K indices + properties, linked to trajectory
   * - LocalDepth3dCrs
     - Full
     - EPSG-based, auto-created
   * - Split coordinate lines
     - Read only
     - Read if expanded to full Points array; write always unsplit
   * - K-gaps
     - Lossy
     - Collapsed on import
   * - Parametric geometry
     - None
     - Only explicit geometry supported
   * - Unstructured grids
     - None
     - Not in xtgeo data model
   * - WellboreTrajectory / WellboreFrame
     - Full
     - Read + write with Feature → Interpretation → Representation chain
   * - BlockedWellboreRepresentation
     - Full
     - I/J/K cell indices + discrete/continuous properties
   * - TriangulatedSetRepresentation
     - Full
     - Per-patch HDF5 naming, SurfaceRole, resqpy-interoperable
   * - Time-series properties
     - None
     - Each timestep is a separate xtgeo property
   * - EPC relationship parts
     - Partial
     - ``_rels/.rels`` written; ``[Content_Types].xml`` included
   * - RESQML 2.2
     - None
     - Only 2.0.1 schemas


Property Name Mapping (Implementation Details)
-----------------------------------------------

This section documents the mapping logic for developers. For the user-facing
reference table, see :ref:`property-mapping-table` in the User Guide.

Resolution order
^^^^^^^^^^^^^^^^

When reading a RESQML property, the name is resolved in this order:

1. **Title match** — the XML ``Citation/Title`` is normalised and looked up in
   the mapping table and title synonyms (e.g., ``"NET/GROSS"`` → ``NTG``)
2. **PropertyKind match** — the RESQML ``PropertyKind`` string is normalised
   and matched via the kind synonym table, with optional facet direction
   (e.g., ``"permeability rock"`` + facet ``I`` → ``PERMX``)
3. **Fallback** — if no mapping is found, the original title is used as-is

The implementation lives in ``_metadata.py:resolve_property_mapping()``.

Title synonyms
^^^^^^^^^^^^^^

These aliases allow RMS-style and free-form names to resolve to canonical
Eclipse keywords:

.. list-table::
   :header-rows: 1
   :widths: 40 30

   * - Input alias
     - Resolves to
   * - ``NET/GROSS``, ``NET_GROSS``, ``NET TO GROSS``
     - ``NTG``
   * - ``PERM_X``, ``KLOGH``
     - ``PERMX``
   * - ``PERM_Y``
     - ``PERMY``
   * - ``PERM_Z``
     - ``PERMZ``
   * - ``SW``
     - ``SWAT``
   * - ``SO``
     - ``SOIL``
   * - ``SG``
     - ``SGAS``
   * - ``FACIES_CODE``
     - ``FACIES``
   * - ``ZONE_LOG``
     - ``ZONE``

RESQML PropertyKind synonyms
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These handle the ``PropertyKind`` field from RESQML XML:

.. list-table::
   :header-rows: 1
   :widths: 50 30

   * - RESQML PropertyKind
     - Eclipse keyword
   * - ``porosity``
     - ``PORO``
   * - ``net to gross ratio``
     - ``NTG``
   * - ``permeability rock``, ``permeability``, ``permeability thickness``
     - ``PERMX`` / ``PERMY`` / ``PERMZ`` (via facet)
   * - ``pressure``, ``pore pressure``
     - ``PRESSURE``
   * - ``water saturation``
     - ``SWAT``
   * - ``oil saturation``
     - ``SOIL``
   * - ``gas saturation``
     - ``SGAS``
   * - ``depth``
     - ``DEPTH``
   * - ``thickness``, ``cell thickness``
     - ``DZ``
   * - ``temperature``
     - ``TEMP``
   * - ``transmissibility``
     - ``TRANX`` / ``TRANY`` / ``TRANZ`` (via facet)
   * - ``facies``
     - ``FACIES``
   * - ``rock type``
     - ``ROCKNUM``
   * - ``zone``
     - ``ZONE``
   * - ``active``
     - ``ACTNUM``
   * - ``region``
     - ``FIPNUM``

For directional properties (permeability, transmissibility), the facet
direction (``I``/``J``/``K`` or ``X``/``Y``/``Z``) selects the variant.


ETP Protocol Details
--------------------

The ``EtpProvider`` communicates via the `Energistics Transfer Protocol (ETP) 1.2
<https://www.energistics.org/etp-specification/>`_ over WebSockets.

Protocols used
^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 10 20 50

   * - ID
     - Protocol
     - Usage
   * - 3
     - Discovery
     - ``GetResources`` — list/search objects, deep traversal
   * - 4
     - Store
     - ``GetDataObjects``, ``PutDataObjects`` — XML metadata
   * - 9
     - DataArray
     - ``GetDataArrays``, ``PutDataArrays`` — bulk numeric arrays
   * - 18
     - Transaction
     - Atomic write groups
   * - 24
     - Dataspace
     - ``GetDataspaces``, ``PutDataspaces``, ``DeleteDataspaces``

Deep Discovery (Protocol 3)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``discover()`` method uses ``GetResources`` with extended parameters:

.. code-block:: python

    ContextInfo(
        uri=target_uri,
        depth=N,                 # 1=direct, N=hops, large=full graph
        data_object_types=[...], # Type filter
        navigable_edges=PRIMARY, # or BOTH, SECONDARY
        include_secondary_targets=bool,
        include_secondary_sources=bool,
    )
    GetResources(
        context=context,
        scope=ContextScopeKind.TARGETS,  # or SOURCES, SELF, etc.
        include_edges=True,              # Return Edge objects
    )

**Response fields:**

- ``Resource``: uri, name, source_count, target_count, last_changed, store_created, active_status
- ``Edge``: source_uri, target_uri, relationship_kind

Change Detection
^^^^^^^^^^^^^^^^

Since the ``energistics`` Python library does not include ETP Protocol 5
(StoreNotification) message classes, change detection is implemented via
polling:

1. Take a baseline snapshot of all resources (using ``discover(depth=0)``)
2. On each ``.poll()`` call, re-query and compare timestamps
3. Detect created (new UUID), changed (different ``last_changed``), deleted (missing UUID)
4. Also query ``GetDeletedResources`` for authoritative deletion tracking

This provides the same semantics as true notifications, with a polling cost.

Connection & Auth Flow
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: text

    1. WebSocket connect (ws:// or wss://)
       Headers: Authorization: Bearer <token>
                Sec-WebSocket-Protocol: etp12.energistics.org

    2. ETP Handshake (Avro binary):
       Client → RequestSession (requested_protocols, supported_data_objects)
       Server → OpenSession (session_id, supported_protocols)

    3. Operations (multiplexed on single connection):
       Client → GetResources / GetDataObjects / PutDataObjects / ...
       Server → GetResourcesResponse / GetDataObjectsResponse / ...

    4. Close:
       Client → CloseSession
       Server → WebSocket close


REST API Endpoints
------------------

Used by ``OsduSession`` for administrative operations:

.. list-table::
   :header-rows: 1
   :widths: 10 40 30

   * - Method
     - Endpoint
     - Purpose
   * - POST
     - ``/api/reservoir-ddms/v2/dataspaces``
     - Create dataspace
   * - GET
     - ``/api/reservoir-ddms/v2/dataspaces``
     - List dataspaces
   * - GET
     - ``/api/reservoir-ddms/v2/dataspaces/{path}``
     - Get dataspace info
   * - DELETE
     - ``/api/reservoir-ddms/v2/dataspaces/{path}``
     - Delete dataspace
   * - GET
     - ``/api/reservoir-ddms/v2/objects?dataspace=...``
     - List objects
   * - POST
     - ``/api/reservoir-ddms/v2/objects/search``
     - Search objects
   * - GET
     - ``/api/reservoir-ddms/v2/objects/{uuid}``
     - Get object metadata


Dependency Graph
----------------

.. code-block:: text

    xtgeo (core)
    ├── numpy
    ├── h5py (EPC HDF5 access)
    ├── lxml (RESQML XML construction/parsing)
    ├── packaging
    └── xtgeo[osdu] (optional extras):
        ├── pyetp  ← provides the `energistics` Python module
        └── websockets

**Graceful degradation:**

- Without ``pyetp``/``websockets``: ETP operations raise ``ImportError`` with installation instructions
- Without ``h5py``: EPC operations raise ``ImportError``
- Without ``resqpy``: Interop tests skip

pyetp — Role & Scope
^^^^^^^^^^^^^^^^^^^^^

The ``pyetp`` package (PyPI: `pyetp <https://pypi.org/project/pyetp/>`_,
GitHub: `equinor/pyetp <https://github.com/equinor/pyetp>`_) provides the
``energistics`` Python module. It is an Equinor-maintained package that
auto-generates typed Pydantic models from the official ETP 1.2 Avro schemas.

**What pyetp provides (used as-is by xtgeo):**

- ``avro_handler.encode_message()`` / ``decode_message()`` — Avro binary
  serialization via ``fastavro``
- Pydantic models for every ETP 1.2 message type (``RequestSession``,
  ``GetDataObjects``, ``PutDataArrays``, etc.)
- Datatype models (``MessageHeader``, ``DataArrayIdentifier``, ``Resource``,
  ``Dataspace``, etc.)
- ``Protocol`` / ``Role`` enums
- Numpy array bridge (``DataArray.from_numpy_array()`` / ``.to_numpy_array()``)

**What pyetp does NOT provide:**

- No WebSocket client or connection management
- No session lifecycle handling
- No RESQML XML object models
- No domain-aware methods or high-level API

**What xtgeo builds on top:**

.. list-table::
   :header-rows: 1
   :widths: 30 50

   * - Layer
     - Implementation
   * - WebSocket ETP client
     - Custom sync-over-async client in ``_etp_provider.py`` using ``websockets``
   * - RESQML 2.0.1 XML
     - Hand-built ``lxml.etree`` construction/parsing for IjkGrid, Grid2D,
       PointSet, PolylineSet, ContinuousProperty, DiscreteProperty, CRS
   * - EPC file I/O
     - Full ZIP/OPC container + HDF5 handling in ``_epc_provider.py`` (no resqpy)
   * - xtgeo ↔ RESQML converters
     - Bidirectional geometry transforms (pillar arrays ↔ coord/zcorn, etc.)
   * - Session / auth
     - OAuth2 token management with profile persistence
   * - High-level API
     - ``grid_from_osdu()``, ``surface_to_osdu()``, ``list_osdu_objects()``, etc.

In summary, pyetp is a **pure schema/codec layer** — xtgeo uses it for
message construction and binary serialization, and builds everything else
(client, XML models, converters, auth) from scratch.

**Known limitation:** pyetp does not include Protocol 5 (StoreNotification)
message classes. Change detection is therefore implemented via timestamp-based
polling (see `Change Detection`_ above).


Testing
-------

Setting Up Local RDDMS
^^^^^^^^^^^^^^^^^^^^^^^

The integration tests require a local OSDU Reservoir DDMS Docker stack.

**Docker Compose setup:**

.. code-block:: yaml

    # docker-compose.yml
    services:
      postgres:
        image: postgres:15
        environment:
          POSTGRES_DB: rddms
          POSTGRES_USER: rddms
          POSTGRES_PASSWORD: rddms
        healthcheck:
          test: pg_isready -U rddms
          interval: 2s
          timeout: 5s
          retries: 5

      etp-server:
        image: community.opengroup.org:5555/osdu/platform/domain-data-mgmt-services/reservoir/reservoir-ddms-etp:latest
        ports:
          - "9002:9002"
        environment:
          RDDMS_DB_HOST: postgres
          RDDMS_DB_NAME: rddms
          RDDMS_DB_USER: rddms
          RDDMS_DB_PASSWORD: rddms
          RDDMS_ETP_PORT: 9002
          RDDMS_AUTH_ENABLED: "false"
        depends_on:
          postgres:
            condition: service_healthy
        healthcheck:
          test: ["CMD", "curl", "-f", "http://localhost:9002/health"]
          interval: 3s
          timeout: 5s
          retries: 10

Start it:

.. code-block:: bash

    docker compose up -d
    docker compose ps   # wait for healthy

Running the Tests
^^^^^^^^^^^^^^^^^

.. code-block:: bash

    cd /path/to/xtgeo

    # CI-safe tests (no RDDMS needed)
    pytest -m "not requires_rddms" tests/test_interfaces/test_osdu/

    # All tests (requires Docker RDDMS or Azure eqndev credentials)
    pytest tests/test_interfaces/test_osdu/

    # Only RDDMS integration tests
    pytest -m "requires_rddms" tests/test_interfaces/test_osdu/

Test Structure
^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 12 58

   * - File
     - RDDMS?
     - What it tests
   * - ``test_epc_compliance.py``
     - No
     - All object types through EPC roundtrip: grids (rotated, faulted,
       pinched, asymmetric, hypothesis-fuzzed), surfaces, points, polygons,
       wells, blocked wells, triangulated surfaces
   * - ``test_epc_operations.py``
     - No
     - Post-roundtrip xtgeo operations (bulk volume, cell dims, XYZ,
       dataframe, crop, grid quality, reduce-to-one-layer, surface extraction)
   * - ``test_api.py``
     - No
     - High-level API functions (grid/surface/points/polygons from/to OSDU)
   * - ``test_osdu_unit.py``
     - No
     - Unit tests for internals (metadata mapping, CRS transforms, property
       resolution, UUID preservation)
   * - ``test_resqpy.py``
     - No
     - Cross-library compatibility with resqpy + pipeline patterns
       (extract_box, coarsen, fault connection sets)
   * - ``test_rddms.py``
     - **Yes**
     - Double roundtrip (Sleipner, Drogon, synthetic TriSet), deep discovery,
       notifications, dataspace snapshot API

**Offline tests** (all except ``test_rddms.py``) run in CI without any
infrastructure. They exercise the full converter and EPC file I/O stack.

**RDDMS tests** (``test_rddms.py``) are marked with ``requires_rddms`` and
excluded from CI via ``pytest -m "not requires_rddms"``. They require either
a local Docker RDDMS (``ws://localhost:9002``) or Azure eqndev credentials.


Contributing
------------

Adding a new object type
^^^^^^^^^^^^^^^^^^^^^^^^

1. Create a converter module (e.g., ``_wellbore.py``)
2. Implement ``resqml_to_xtgeo()`` and ``xtgeo_to_resqml()``
3. Add provider methods (``get_*_geometry``, ``put_*``)
4. Add user API functions (``well_from_osdu``, ``well_to_osdu``)
5. Export from ``__init__.py`` and top-level ``xtgeo/__init__.py``
6. Write EPC roundtrip tests + ETP integration tests

Adding a new provider
^^^^^^^^^^^^^^^^^^^^^

1. Subclass ``ResqmlDataProvider``
2. Implement all abstract methods
3. Register in ``_api._open_provider()``
4. Write tests against the existing converter test fixtures


Future Considerations
---------------------

- **OSDU v3 REST data API**: Add a ``RestProvider`` when REST data endpoints stabilize
- **Streaming large grids**: ETP supports chunked DataArray transfers (currently full-buffer)
- **Property collections**: Group by realization/timestep for ensemble workflows
- **Well data**: RESQML WellboreTrajectory/WellboreFrame → xtgeo Well
- **True Protocol 5 notifications**: If/when the energistics library adds StoreNotification message classes
- **Split coordinate lines (write)**: Full XY-split pillar export when xtgeo's data model supports it
- **Time-series properties**: Temporal property support with ``TimeIndex`` / ``TimeSeries``


External References
-------------------

Standards
^^^^^^^^^

- `RESQML 2.0.1 Specification <https://www.energistics.org/resqml-data-standards/>`_
  — The data model for subsurface objects (grids, properties, surfaces)
- `ETP 1.2 Specification <https://www.energistics.org/etp-specification/>`_
  — Energistics Transfer Protocol (WebSocket + Avro binary)
- `OSDU Technical Standard <https://community.opengroup.org/osdu/>`_
  — Open Subsurface Data Universe platform specification

Libraries
^^^^^^^^^

- `energistics (pyetp) <https://github.com/equinor/pyetp>`_
  — Python implementation of ETP 1.2 message schemas
- `resqpy <https://github.com/bp/resqpy>`_
  — Pure-Python RESQML 2.0.1 read/write (for cross-validation)
- `h5py <https://www.h5py.org/>`_
  — HDF5 file access for array data in EPC containers

OSDU Services
^^^^^^^^^^^^^

- `OSDU Reservoir DDMS <https://community.opengroup.org/osdu/platform/domain-data-mgmt-services/reservoir/>`_
  — The backend service providing ETP 1.2 access to subsurface data
