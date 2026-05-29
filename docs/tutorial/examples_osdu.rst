OSDU / RESQML examples
======================

This tutorial covers reading and writing xtgeo objects to and from
OSDU Reservoir DDMS (RDDMS) servers and RESQML EPC file containers.

.. seealso::

   - :doc:`/osdu/guide` — Full user guide with connection options, environment
     variables, profiles, and property name mapping reference
   - :doc:`/osdu/api` — Complete API reference
   - :doc:`/osdu/demos` — More runnable examples (change tracking, bulk copy)
   - :doc:`/osdu/developer` — Architecture, data model details, testing

Prerequisites
-------------

Install the optional OSDU dependencies:

.. code-block:: bash

    pip install xtgeo[osdu]

This installs ``pyetp`` (ETP 1.2 Avro protocol schemas) and ``websockets``.
The core dependencies ``lxml`` and ``h5py`` (for EPC files) are already
included in xtgeo.

Session setup
-------------

For local development (Docker RDDMS, no auth):

.. code-block:: python

    from xtgeo.interfaces.osdu import OsduSession

    session = OsduSession(
        etp_url="ws://localhost:9002",
        dataspace="myteam/project",
        auth_mode="none",
    )

For cloud OSDU via environment variables (recommended for CI/scripts):

.. code-block:: python

    from xtgeo.interfaces.osdu import OsduSession

    # Reads OSDU_HOSTNAME, OSDU_TENANT_ID, OSDU_CLIENT_ID, etc.
    session = OsduSession.from_env()

For cloud OSDU with explicit configuration:

.. code-block:: python

    import os
    from xtgeo.interfaces.osdu import OsduSession

    session = OsduSession(
        profile="my-cloud",
        etp_url="wss://your-osdu-host.energy.azure.com/api/reservoir-ddms-etp/v2/",
        rest_base_url="https://your-osdu-host.energy.azure.com",
        token_url="https://login.microsoftonline.com/<tenant>/oauth2/v2.0/token",
        client_id="<your-client-id>",
        client_secret=os.environ["OSDU_CLIENT_SECRET"],
        auth_mode="client_credentials",
        data_partition="<your-partition>",
        dataspace="myteam/project",
        legal_tag="<partition>-private-default",
        owners=["data.default.owners@<partition>.dataservices.energy"],
        viewers=["data.default.viewers@<partition>.dataservices.energy"],
    )
    session.save()  # Persists to ~/.config/xtgeo/osdu/my-cloud.toml

Later, reload with:

.. code-block:: python

    session = OsduSession.load("my-cloud")

See :ref:`connecting-to-a-server` in the User Guide for the full list of
environment variables and configuration options.

Discovering objects
-------------------

.. code-block:: python

    import xtgeo

    # List everything in a dataspace
    objects = xtgeo.list_osdu_objects(session)
    for obj in objects:
        print(f"{obj['type']:20s} {obj['uuid'][:8]}... {obj['title']}")

    # Filter by type
    grids = xtgeo.list_osdu_objects(session, object_type="grid")
    surfaces = xtgeo.list_osdu_objects(session, object_type="surface")
    properties = xtgeo.list_osdu_objects(session, object_type="property")

    # Search by name pattern (wildcards supported)
    results = xtgeo.search_osdu(session, name="*Drogon*")
    results = xtgeo.search_osdu(session, name="PORO", object_type="property")

Reading data
------------

Grids with properties
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Read by name (first match)
    grid, props = xtgeo.grid_from_osdu(session, name="Drogon")
    print(grid)  # Grid: ncol=46, nrow=112, nlay=22
    for p in props:
        print(f"  {p.name}: {p.values.min():.3f} - {p.values.max():.3f}")

    # Read by UUID (exact)
    grid, props = xtgeo.grid_from_osdu(session, uuid="12345678-...")

    # Grid only, skip properties
    grid, _ = xtgeo.grid_from_osdu(session, name="Drogon", load_properties=False)

Surfaces
^^^^^^^^

.. code-block:: python

    surf = xtgeo.surface_from_osdu(session, name="TopVolantis")
    print(surf)
    # RegularSurface: ncol=100, nrow=100, xinc=25.0, yinc=25.0, rotation=30.0

Points and polygons
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    pts = xtgeo.points_from_osdu(session, name="WellTops")
    polys = xtgeo.polygons_from_osdu(session, name="FaultTraces")

From EPC files
^^^^^^^^^^^^^^

All the same functions work with EPC file paths:

.. code-block:: python

    grid, props = xtgeo.grid_from_osdu("exported_model.epc", name="Drogon")
    surf = xtgeo.surface_from_osdu("maps.epc", name="TopVolantis")

Writing data
------------

.. code-block:: python

    import xtgeo
    import numpy as np

    # Create some data
    grid = xtgeo.create_box_grid((10, 10, 5))
    poro = xtgeo.GridProperty(grid, name="PORO", values=np.random.rand(10, 10, 5))

    # Write to OSDU/RDDMS
    uuids = xtgeo.grid_to_osdu(
        session,
        grid,
        title="TestGrid",
        properties=[poro],
        crs_epsg=23031,
    )
    print(uuids)  # {'TestGrid': 'uuid-...', 'CRS': 'uuid-...', 'PORO': 'uuid-...'}

    # Write surface
    surf = xtgeo.RegularSurface(ncol=50, nrow=50, xinc=25, yinc=25, values=np.zeros((50, 50)))
    xtgeo.surface_to_osdu(session, surf, title="FlatSurface", crs_epsg=23031)

    # Write to EPC file
    xtgeo.grid_to_osdu("output.epc", grid, title="MyGrid", properties=[poro], crs_epsg=23031)

Dataspace management
--------------------

Creating and managing dataspaces (requires REST API or ETP):

.. code-block:: python

    # Create dataspace via ETP (local RDDMS)
    session.create_dataspace_etp("myteam/experiment")

    # Create via REST (cloud OSDU)
    session.create_dataspace_rest("myteam/experiment")

    # List all dataspaces (REST)
    for ds in session.list_dataspaces():
        print(ds["Path"])

    # Switch working dataspace
    session.switch_dataspace("myteam/experiment")

    # Search objects in a dataspace (REST)
    results = session.search_objects_rest("porosity", object_type="Property")

    # Delete (WARNING: permanent)
    session.delete_dataspace("myteam/experiment")

Bulk dataspace copy
-------------------

Copy an entire dataspace (all objects) to a new location:

.. code-block:: python

    from xtgeo.interfaces.osdu import (
        OsduSession, EtpProvider,
        read_dataspace, write_dataspace, compare_snapshots,
    )

    session = OsduSession(etp_url="ws://localhost:9002", dataspace="myteam/project")
    config = session.etp_config()

    # Read everything from source
    with EtpProvider(config) as provider:
        snapshot = read_dataspace(provider)

    print(f"Read {len(snapshot.grids)} grids, {len(snapshot.surfaces)} surfaces")

    # Write to new dataspace
    session.switch_dataspace("myteam/project-copy")
    session.create_dataspace_etp("myteam/project-copy")
    config2 = session.etp_config()

    with EtpProvider(config2) as provider:
        write_dataspace(provider, snapshot, preserve_uuids=True)

    # Verify exact match
    with EtpProvider(config2) as provider:
        snapshot2 = read_dataspace(provider)

    diffs = compare_snapshots(snapshot, snapshot2, atol=1e-10)
    if not diffs:
        print("Perfect roundtrip!")
    else:
        for d in diffs:
            print(f"  DIFF: {d}")

Property name mapping
---------------------

Eclipse keywords are automatically mapped to OSDU property names when
reading and writing:

.. code-block:: python

    from xtgeo.interfaces.osdu import ecl_keyword_to_osdu, osdu_name_to_ecl_keyword

    mapping = ecl_keyword_to_osdu("PORO")
    print(mapping.osdu_name)    # "Porosity"
    print(mapping.uom_family)   # "fraction"

    kw = osdu_name_to_ecl_keyword("Porosity")
    print(kw)  # "PORO"

    # List all 40 supported mappings
    from xtgeo.interfaces.osdu import list_supported_properties
    for m in list_supported_properties():
        print(f"{m.ecl_keyword:12s} → {m.osdu_name:30s} ({m.uom_family})")

Common aliases like ``SW`` → ``SWAT``, ``KLOGH`` → ``PERMX``, and
``NET/GROSS`` → ``NTG`` are also resolved automatically.

Unmapped property names are stored as-is and round-trip correctly, but
won't have standardised OSDU metadata.

See :ref:`property-mapping-table` in the User Guide for the complete
reference table.
