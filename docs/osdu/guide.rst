User Guide
==========

This guide walks you through using xtgeo with OSDU Reservoir DDMS servers
and RESQML EPC files. No prior knowledge of RESQML or ETP protocols is needed —
the API follows the same patterns as the rest of xtgeo.

.. contents:: On this page
   :local:
   :depth: 2


Installation
------------

Install xtgeo with the optional OSDU dependencies:

.. code-block:: bash

    pip install xtgeo[osdu]

This installs:

- ``energistics`` — ETP 1.2 protocol (Avro binary WebSocket)
- ``h5py`` — HDF5 array storage for EPC files

Quick Start Recipes
-------------------

Cloud connection via environment variables
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    # Set these in your shell, CI pipeline, or .env file:
    export OSDU_HOSTNAME="equinorswedev.energy.azure.com"
    export OSDU_TENANT_ID="3aa4a235-..."
    export OSDU_CLIENT_ID="21b442a9-..."
    export OSDU_CLIENT_SECRET="<secret>"
    export OSDU_SCOPE="bd0c9d90-.../.default"
    export OSDU_DATA_PARTITION="dev"
    export OSDU_DATASPACE="myteam/project"
    export OSDU_LEGAL_TAG="dev-equinor-private-default"
    export OSDU_ACL_OWNERS="data.default.owners@dev.dataservices.energy"
    export OSDU_ACL_VIEWERS="data.default.viewers@dev.dataservices.energy"

.. code-block:: python

    from xtgeo.interfaces.osdu import OsduSession

    # Picks up all OSDU_* env vars automatically
    session = OsduSession.from_env()
    # → etp_url derived from OSDU_HOSTNAME
    # → token_url derived from OSDU_TENANT_ID
    # → auth_mode auto-detected as "client_credentials"

    # Create dataspace with ACL/legal (required for cloud RDDMS)
    session.create_dataspace_rest()

    # Now use normally
    import xtgeo
    xtgeo.grid_to_osdu(session, grid, title="MyGrid", crs_epsg=23031)

Read a grid, modify, write back
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import xtgeo

    session = OsduSession.from_env()

    # Read grid + all properties
    grid, props = xtgeo.grid_from_osdu(session, name="Drogon")

    # Modify a property
    poro = next(p for p in props if p.name == "PORO")
    poro.values *= 0.9  # Apply cutoff

    # Write to a new dataspace
    session.switch_dataspace("myteam/modified")
    session.create_dataspace_rest()
    xtgeo.grid_to_osdu(session, grid, title="Drogon_mod", properties=props, crs_epsg=23031)

Eclipse GRDECL → OSDU roundtrip (numerically exact)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import xtgeo
    import numpy as np
    from xtgeo.interfaces.osdu import OsduSession

    session = OsduSession()  # local dev

    # Import from Eclipse format
    grid = xtgeo.grid_from_file("GRID.grdecl", fformat="grdecl")
    poro = xtgeo.gridproperty_from_file("PORO.grdecl", fformat="grdecl",
                                         name="PORO", grid=grid)

    # Store in OSDU
    xtgeo.grid_to_osdu(session, grid, title="EclGrid",
                       properties=[poro], crs_epsg=23031)

    # Read back — geometry is bit-for-bit identical
    grid2, props2 = xtgeo.grid_from_osdu(session, name="EclGrid")
    assert np.array_equal(grid2._coordsv, grid._coordsv)
    assert np.array_equal(grid2._zcornsv, grid._zcornsv)

    # Export back to Eclipse — no information loss
    grid2.to_file("GRID_roundtrip.grdecl", fformat="grdecl")

Save and reuse a profile
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from xtgeo.interfaces.osdu import OsduSession

    # First time: configure and save
    session = OsduSession(
        profile="equinor-dev",
        etp_url="wss://edl.equinor.com/api/reservoir-ddms-etp/v2/",
        rest_base_url="https://edl.equinor.com",
        token_url="https://login.microsoftonline.com/<tenant>/oauth2/v2.0/token",
        client_id="<client-id>",
        auth_mode="client_credentials",
        data_partition="equinor-dev",
        dataspace="myteam/project",
        legal_tag="equinor-dev-equinor-private-default",
        owners=["data.default.owners@equinor-dev.dataservices.energy"],
        viewers=["data.default.viewers@equinor-dev.dataservices.energy"],
    )
    session.save()  # → ~/.config/xtgeo/osdu/equinor-dev.toml

    # Later (secrets come from XTGEO_OSDU_CLIENT_SECRET env var):
    session = OsduSession.load("equinor-dev")

List available profiles:

.. code-block:: python

    print(OsduSession.list_profiles())  # ['equinor-dev', 'local']


Connecting to a Server
----------------------

There are three ways to configure a connection, from simplest to most explicit:

From environment variables (recommended for CI/scripts)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from xtgeo.interfaces.osdu import OsduSession

    session = OsduSession.from_env()

This reads ``OSDU_*`` environment variables (see table below) and auto-detects
the auth mode. For local development with no env vars set, it defaults to
``ws://localhost:9002`` with no authentication.

.. list-table:: Supported environment variables
   :header-rows: 1
   :widths: 35 65

   * - Variable
     - Purpose
   * - ``OSDU_HOSTNAME``
     - Derives ETP URL (``wss://<host>/api/reservoir-ddms-etp/v2/``) and REST base
   * - ``OSDU_TENANT_ID``
     - Derives Azure AD token URL
   * - ``OSDU_CLIENT_ID``
     - OAuth2 client ID
   * - ``OSDU_CLIENT_SECRET``
     - OAuth2 client secret
   * - ``OSDU_SCOPE``
     - OAuth2 scope (defaults to ``<client_id>/.default``)
   * - ``OSDU_DATA_PARTITION``
     - OSDU data partition (e.g. ``"dev"``, ``"equinor-dev"``)
   * - ``OSDU_DATASPACE``
     - ETP dataspace path (e.g. ``"myteam/project"``)
   * - ``OSDU_LEGAL_TAG``
     - Default legal tag for new objects
   * - ``OSDU_ACL_OWNERS``
     - Comma-separated ACL owner groups
   * - ``OSDU_ACL_VIEWERS``
     - Comma-separated ACL viewer groups
   * - ``OSDU_COUNTRIES``
     - Comma-separated ISO country codes (default: ``"NO"``)

Higher-priority overrides use the ``XTGEO_OSDU_`` prefix (e.g.
``XTGEO_OSDU_ETP_URL``, ``XTGEO_OSDU_CLIENT_SECRET``).

From a saved profile (recommended for interactive use)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    session = OsduSession.load("equinor-dev")

Profiles are TOML files at ``~/.config/xtgeo/osdu/<name>.toml``. Secrets
should always come from environment variables, not the file.

Explicit construction (for programmatic control)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Local development (Docker RDDMS, no authentication):
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import xtgeo
    from xtgeo.interfaces.osdu import OsduSession

    # Start local Docker RDDMS, then:
    session = OsduSession()  # defaults to ws://localhost:9002, no auth

    grid = xtgeo.grid_from_file("mymodel.grdecl", fformat="grdecl")
    xtgeo.grid_to_osdu(session, grid, title="MyGrid", crs_epsg=23031)

    grid2, props = xtgeo.grid_from_osdu(session, name="MyGrid")
    assert grid2.ncol == grid.ncol  # exact roundtrip

.. code-block:: python

    from xtgeo.interfaces.osdu import OsduSession

    session = OsduSession(
        etp_url="ws://localhost:9002",
        dataspace="myteam/project",
    )

Cloud OSDU (e.g. Equinor Energy DataLake with Azure AD):

.. code-block:: python

    session = OsduSession(
        etp_url="wss://edl.equinor.com/api/reservoir-ddms-etp/v2/",
        rest_base_url="https://edl.equinor.com",
        token_url="https://login.microsoftonline.com/<tenant>/oauth2/v2.0/token",
        client_id="<client-id>",
        client_secret=os.environ["OSDU_CLIENT_SECRET"],
        auth_mode="client_credentials",
        data_partition="equinor-dev",
        dataspace="myteam/project",
        legal_tag="equinor-dev-equinor-private-default",
        owners=["data.default.owners@equinor-dev.dataservices.energy"],
        viewers=["data.default.viewers@equinor-dev.dataservices.energy"],
    )

Reload a saved profile:

.. code-block:: python

    session = OsduSession.load("equinor-dev")


Working with EPC Files
----------------------

Every function that takes a ``session`` also accepts a path to an ``.epc`` file.
This makes it easy to work offline or exchange data as files:

.. code-block:: python

    import xtgeo

    # Read from an EPC file
    grid, props = xtgeo.grid_from_osdu("model.epc", name="Drogon")
    surf = xtgeo.surface_from_osdu("maps.epc", name="TopVolantis")

    # Write to an EPC file
    xtgeo.grid_to_osdu("output.epc", grid, title="MyGrid", crs_epsg=23031)


Discovering Objects
-------------------

List everything in a dataspace:

.. code-block:: python

    import xtgeo

    objects = xtgeo.list_osdu_objects(session)
    for obj in objects:
        print(f"{obj['type']:25s} {obj['uuid'][:8]}... {obj['title']}")

Filter by type:

.. code-block:: python

    grids = xtgeo.list_osdu_objects(session, object_type="grid")
    surfaces = xtgeo.list_osdu_objects(session, object_type="surface")
    properties = xtgeo.list_osdu_objects(session, object_type="property")

Search by name pattern (shell-style wildcards):

.. code-block:: python

    results = xtgeo.search_osdu(session, name="*Drogon*")
    results = xtgeo.search_osdu(session, name="PORO", object_type="property")


Reading Data
------------

Grids with properties
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Read by name (first match)
    grid, props = xtgeo.grid_from_osdu(session, name="Drogon")
    print(grid)  # Grid: ncol=46, nrow=112, nlay=22
    for p in props:
        print(f"  {p.name}: {p.values.min():.3f} - {p.values.max():.3f}")

    # Read by UUID
    grid, props = xtgeo.grid_from_osdu(session, uuid="12345678-abcd-...")

    # Grid geometry only (skip properties)
    grid, _ = xtgeo.grid_from_osdu(session, name="Drogon", load_properties=False)

Surfaces
^^^^^^^^

.. code-block:: python

    surf = xtgeo.surface_from_osdu(session, name="TopVolantis")
    # RegularSurface: ncol=100, nrow=100, xinc=25.0, yinc=25.0

Points and Polygons
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    pts = xtgeo.points_from_osdu(session, name="WellTops")
    polys = xtgeo.polygons_from_osdu(session, name="FaultTraces")


Writing Data
------------

.. code-block:: python

    import xtgeo
    import numpy as np

    # Create test data
    grid = xtgeo.create_box_grid((10, 10, 5))
    poro = xtgeo.GridProperty(grid, name="PORO", values=np.random.rand(10, 10, 5))

    # Write grid + properties to OSDU
    uuids = xtgeo.grid_to_osdu(
        session, grid,
        title="TestGrid",
        properties=[poro],
        crs_epsg=23031,
    )
    print(uuids)
    # {'TestGrid': 'uuid-...', 'CRS': 'uuid-...', 'PORO': 'uuid-...'}

    # Write a surface
    surf = xtgeo.RegularSurface(
        ncol=50, nrow=50, xinc=25, yinc=25,
        values=np.zeros((50, 50))
    )
    xtgeo.surface_to_osdu(session, surf, title="FlatSurface", crs_epsg=23031)

    # Write points / polygons
    pts = xtgeo.Points(values=np.array([[1, 2, 3], [4, 5, 6]]))
    xtgeo.points_to_osdu(session, pts, title="MyPoints", crs_epsg=23031)


Deep Discovery
--------------

Explore relationships between objects — find what's connected to what,
traverse the full object graph, or filter by type across the entire dataspace.

.. code-block:: python

    # Discover everything in the dataspace (full graph)
    result = xtgeo.deep_query_osdu(session, depth=0)
    for r in result['resources']:
        print(f"{r['type']:30s} {r['title']}")

    # Find all properties referencing a specific grid
    result = xtgeo.deep_query_osdu(session, uuid=grid_uuid, scope="sources")
    for r in result['resources']:
        print(f"  Property: {r['title']}")

    # Only discover specific types
    result = xtgeo.deep_query_osdu(
        session, depth=0,
        object_types=["IjkGridRepresentation"],
    )

    # Include relationship edges
    result = xtgeo.deep_query_osdu(session, depth=2, include_edges=True)
    for edge in result['edges']:
        print(f"  {edge['source_uri']} → {edge['target_uri']}")


Change Tracking
---------------

Monitor a dataspace for changes — detect when objects are created, modified,
or deleted:

.. code-block:: python

    # Subscribe to all changes
    sub = xtgeo.watch_osdu_changes(session)

    # ... time passes, someone modifies data ...

    # Poll for changes
    events = sub.poll()
    for e in events:
        print(f"{e['event']:8s} {e['type']:25s} {e['title']}")
    # created  ContinuousProperty        PERMX
    # changed  IjkGridRepresentation     Drogon

    # Filter to specific types
    sub = xtgeo.watch_osdu_changes(session, object_types=["IjkGridRepresentation"])

    # Use a callback
    def on_change(event_type, info):
        print(f"  {event_type}: {info['title']}")

    sub = xtgeo.watch_osdu_changes(session, callback=on_change)
    sub.poll()  # Fires callback for each event

    # Stop watching
    sub.stop()

    # Or use as context manager
    with xtgeo.watch_osdu_changes(session) as sub:
        events = sub.poll()


Dataspace Management
--------------------

.. code-block:: python

    # Create a new dataspace
    session.create_dataspace_etp("myteam/experiment")   # via ETP (local)
    session.create_dataspace_rest("myteam/experiment")  # via REST (cloud)

    # List all dataspaces
    for ds in session.list_dataspaces():
        print(ds["Path"])

    # Switch working dataspace
    session.switch_dataspace("myteam/experiment")

    # Delete (permanent!)
    session.delete_dataspace("myteam/experiment")


Bulk Dataspace Operations
-------------------------

Copy entire dataspaces (all objects) in one go:

.. code-block:: python

    from xtgeo.interfaces.osdu import (
        EtpProvider, read_dataspace, write_dataspace, compare_snapshots,
    )

    config = session.etp_config()

    # Read everything from source
    with EtpProvider(config) as provider:
        snapshot = read_dataspace(provider)

    print(f"Read {len(snapshot.grids)} grids, {len(snapshot.surfaces)} surfaces")

    # Write to new dataspace
    session.switch_dataspace("myteam/copy")
    session.create_dataspace_etp("myteam/copy")
    config2 = session.etp_config()

    with EtpProvider(config2) as provider:
        write_dataspace(provider, snapshot, preserve_uuids=True)

    # Verify exact match
    with EtpProvider(config2) as provider:
        snapshot2 = read_dataspace(provider)

    diffs = compare_snapshots(snapshot, snapshot2, atol=1e-10)
    if not diffs:
        print("Perfect roundtrip!")


Metadata Preservation
---------------------

UUIDs and RESQML metadata are preserved automatically through roundtrips:

.. code-block:: python

    # Write a grid
    xtgeo.grid_to_osdu(session, grid, title="MyGrid", crs_epsg=23031)

    # Read it back — the UUID is attached to the xtgeo object
    grid2, _ = xtgeo.grid_from_osdu(session, name="MyGrid")
    print(grid2.metadata["resqml_uuid"])  # Same UUID as written

    # Re-write: UUID is preserved (updates in place)
    xtgeo.grid_to_osdu(session, grid2, title="MyGrid", crs_epsg=23031)


Property Name Mapping
---------------------

Eclipse keywords are mapped to OSDU property names automatically:

.. code-block:: python

    from xtgeo.interfaces.osdu import ecl_keyword_to_osdu, osdu_name_to_ecl_keyword

    mapping = ecl_keyword_to_osdu("PORO")
    print(mapping.osdu_name)  # "Porosity"
    print(mapping.uom)        # "v/v"

    kw = osdu_name_to_ecl_keyword("Porosity")
    print(kw)  # "PORO"


Multi-Dataspace Queries
-----------------------

Search or query across all accessible dataspaces at once:

.. code-block:: python

    # Query specific field/type across all dataspaces
    results = xtgeo.query_osdu(session, name="*Drogon*", object_type="grid")

    # Import an object from its URI into the local session
    obj = xtgeo.import_osdu(session, uri=results[0]["uri"])


Format Compatibility
--------------------

Exact roundtrip (lossless)
^^^^^^^^^^^^^^^^^^^^^^^^^^

The following conversions are **numerically exact** — no information is lost:

- **Eclipse GRDECL ↔ xtgeo ↔ RESQML/ETP** — pillar coordinates, Z-corners,
  and activity masks are bit-for-bit identical after roundtrip. Tested with
  heavily faulted grids (117/121 pillars split).

- **ROFF ↔ xtgeo ↔ RESQML/ETP** — exact for grids with ``split_enz = 1``
  (unsplit) and ``split_enz = 4`` (lateral faults). These cover the vast
  majority of production grids.

- **RegularSurface ↔ RESQML Grid2D** — origin, increments, rotation, and
  Z-values all preserved exactly.

- **Grid properties** — float64 values preserved through ETP protocol.

Best-effort (lossy)
^^^^^^^^^^^^^^^^^^^

These cases require approximation:

- **K-direction gaps/overlaps** — Eclipse ZCORN allows vertical splits between
  layers (cells don't share top/bottom surfaces). XTGeo forces layers
  connected on import (with a warning). The RESQML roundtrip preserves
  whatever xtgeo stores, but the original K-gaps are lost on Eclipse import.

- **ROFF split_enz = 2 or 8** — K-direction splits or full 8-way splits are
  not supported by the C conversion code. Only lateral splits (``split_enz = 4``)
  are preserved.

Reading grids from other tools
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Grids written by resqpy, Petrel, or other RESQML-compliant tools that use the
standard ``Point3dHdf5Array`` format (shape ``(nk+1, nj+1, ni+1, 3)``) can be
read. The KJI→IJK axis reordering is handled automatically.

.. note::

   Split coordinate lines (fault geometry stored separately from main pillar
   array via ``PillarIndices`` + ``ColumnsPerSplitCoordinateLine``) are not
   yet supported for reading. Grids with this geometry type need to be
   pre-processed through resqpy to get the full explicit Points array.

RESQML compliance
^^^^^^^^^^^^^^^^^

Written RESQML objects include:

- ``GridIsRighthanded`` — inferred from actual grid geometry
- ``PillarShape`` — always ``"straight"`` (corner-point grids)
- ``KDirection`` — default ``"down"``
- Proper Grid2D offset axis ordering (1st = J/slowest, 2nd = I/fastest)

These ensure interoperability with other RESQML readers (resqpy, Petrel, etc.).
