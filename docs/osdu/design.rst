Design & Development
====================

Architecture, protocol details, and guidance for contributors.

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

Install the optional OSDU dependencies with:

.. code-block:: bash

    pip install xtgeo[osdu]

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
