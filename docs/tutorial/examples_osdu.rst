OSDU / RESQML examples
======================

This tutorial covers reading and writing xtgeo objects to and from
OSDU Reservoir DDMS (RDDMS) servers and RESQML EPC file containers.

Prerequisites
-------------

Install the optional OSDU dependencies:

.. code-block:: bash

    pip install xtgeo[osdu]

This installs ``energistics`` (ETP 1.2 protocol) and ``h5py``.
For file-based EPC access, ``resqpy`` is also recommended.

Session setup
-------------

For local development (Docker RDDMS, no auth):

.. code-block:: python

    from xtgeo.interfaces.osdu import OsduSession

    session = OsduSession(
        etp_url="ws://localhost:9002",
        dataspace="maap/drogon",
        auth_mode="none",
    )

For cloud OSDU (Equinor Energy DataLake, Azure AD auth):

.. code-block:: python

    from xtgeo.interfaces.osdu import OsduSession

    session = OsduSession(
        profile="equinor-dev",
        etp_url="wss://edl.equinor.com/api/reservoir-ddms-etp/v2/",
        rest_base_url="https://edl.equinor.com",
        token_url="https://login.microsoftonline.com/<tenant>/oauth2/v2.0/token",
        client_id="<your-client-id>",
        auth_mode="refresh_token",
        data_partition="equinor-dev",
        dataspace="myteam/project",
    )
    # Set secrets via environment:
    # export XTGEO_OSDU_REFRESH_TOKEN=...
    # export XTGEO_OSDU_CLIENT_SECRET=...

    session.save()  # Persists to ~/.config/xtgeo/osdu/equinor-dev.toml

Later, reload with:

.. code-block:: python

    session = OsduSession.load("equinor-dev")

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

    session = OsduSession(etp_url="ws://localhost:9002", dataspace="maap/drogon")
    config = session.etp_config()

    # Read everything from source
    with EtpProvider(config) as provider:
        snapshot = read_dataspace(provider)

    print(f"Read {len(snapshot.grids)} grids, {len(snapshot.surfaces)} surfaces")

    # Write to new dataspace
    session.switch_dataspace("maap/drogon-copy")
    session.create_dataspace_etp("maap/drogon-copy")
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

The module maps between Eclipse keywords and OSDU property names:

.. code-block:: python

    from xtgeo.interfaces.osdu import ecl_keyword_to_osdu, osdu_name_to_ecl_keyword

    mapping = ecl_keyword_to_osdu("PORO")
    print(mapping.osdu_name)  # "Porosity"
    print(mapping.uom)        # "v/v"

    kw = osdu_name_to_ecl_keyword("Porosity")
    print(kw)  # "PORO"
