Demos & Examples
================

Copy-paste runnable examples for common workflows.

.. contents:: On this page
   :local:
   :depth: 2


Quick Demo: Local RDDMS
------------------------

A complete workflow against a local Docker RDDMS (no auth required):

.. code-block:: python

    import xtgeo
    import numpy as np
    from xtgeo.interfaces.osdu import OsduSession

    # 1. Connect
    session = OsduSession(
        etp_url="ws://localhost:9002",
        dataspace="demo/quickstart",
        auth_mode="none",
    )

    # 2. Write a grid with properties
    grid = xtgeo.create_box_grid((10, 8, 5), origin=(460000, 5930000, 1000))
    poro = xtgeo.GridProperty(grid, name="PORO", values=np.random.rand(10, 8, 5))
    perm = xtgeo.GridProperty(grid, name="PERMX", values=np.random.rand(10, 8, 5) * 500)

    uuids = xtgeo.grid_to_osdu(
        session, grid,
        title="DemoGrid",
        properties=[poro, perm],
        crs_epsg=23031,
    )
    print(f"Written: {uuids}")

    # 3. Discover what's in the dataspace
    objects = xtgeo.list_osdu_objects(session)
    print(f"\n{'Type':30s} {'Title':20s} UUID")
    print("-" * 80)
    for obj in objects:
        print(f"{obj['type']:30s} {obj['title']:20s} {obj['uuid'][:8]}...")

    # 4. Deep discovery — find property→grid relationships
    result = xtgeo.deep_query_osdu(session, depth=0, include_edges=True)
    print(f"\nGraph: {len(result['resources'])} objects, {len(result['edges'])} edges")

    # 5. Read it back
    grid2, props2 = xtgeo.grid_from_osdu(session, name="DemoGrid")
    print(f"\nRead back: {grid2}")
    for p in props2:
        print(f"  {p.name}: min={p.values.min():.4f} max={p.values.max():.4f}")

    # 6. Verify exact roundtrip
    assert grid2.ncol == grid.ncol
    assert np.allclose(grid2._coordsv, grid._coordsv)
    print("\n✓ Exact geometry roundtrip verified")


Demo: Change Tracking
---------------------

Watch for changes in a dataspace:

.. code-block:: python

    import xtgeo
    import numpy as np
    from xtgeo.interfaces.osdu import OsduSession, EtpProvider

    session = OsduSession(etp_url="ws://localhost:9002", dataspace="demo/watch")
    config = session.etp_config()

    with EtpProvider(config) as provider:
        # Start watching
        sub = provider.subscribe_notifications()

        # Simulate a change: write a new surface
        from xtgeo.interfaces.osdu import xtgeo_surface_to_resqml
        surf = xtgeo.RegularSurface(ncol=10, nrow=10, xinc=25, yinc=25,
                                     values=np.random.rand(10, 10))
        xtgeo_surface_to_resqml(provider, surf, title="WatchedSurface")

        # Detect the change
        events = sub.poll()
        for e in events:
            print(f"  {e['event']:8s} {e['type']:30s} {e['title']}")
        # Output: created  resqml20.Grid2dRepresentation  WatchedSurface

        sub.stop()


Demo: Bulk Dataspace Copy
-------------------------

Copy all objects between dataspaces with integrity verification:

.. code-block:: python

    from xtgeo.interfaces.osdu import (
        OsduSession, EtpProvider, EtpConnectionConfig,
        read_dataspace, write_dataspace, compare_snapshots,
    )

    # Source
    src_cfg = EtpConnectionConfig(
        url="ws://localhost:9002",
        dataspace="eml:///dataspace('source/project')",
    )

    with EtpProvider(src_cfg) as src:
        snapshot = read_dataspace(src)
        print(f"Source: {len(snapshot.grids)} grids, "
              f"{len(snapshot.surfaces)} surfaces, "
              f"{len(snapshot.properties)} properties")

    # Destination
    dst_cfg = EtpConnectionConfig(
        url="ws://localhost:9002",
        dataspace="eml:///dataspace('dest/project')",
    )

    with EtpProvider(dst_cfg) as dst:
        dst.put_dataspace("dest/project")
        write_dataspace(dst, snapshot, preserve_uuids=True)

    # Verify
    with EtpProvider(dst_cfg) as dst:
        snapshot2 = read_dataspace(dst)

    diffs = compare_snapshots(snapshot, snapshot2, atol=1e-10)
    if not diffs:
        print("✓ Perfect copy")
    else:
        for d in diffs:
            print(f"  DIFF: {d}")


Demo: EPC File Workflow
-----------------------

Pure offline workflow — no server needed:

.. code-block:: python

    import xtgeo
    import numpy as np

    # Write to EPC
    grid = xtgeo.create_box_grid((5, 5, 3))
    poro = xtgeo.GridProperty(grid, name="PORO", values=np.random.rand(5, 5, 3))
    xtgeo.grid_to_osdu("model.epc", grid, title="MyGrid",
                        properties=[poro], crs_epsg=23031)

    # Read from EPC
    grid2, props = xtgeo.grid_from_osdu("model.epc", name="MyGrid")
    print(grid2, props[0].name)

    # Share with resqpy users
    import resqpy.model
    model = resqpy.model.Model("model.epc")
    print(model.titles())  # ['MyGrid', 'PORO', 'Default CRS']


See Also
--------

- :doc:`guide` — Full user guide with connection setup and property mappings
- :doc:`developer` — Testing setup, Docker, architecture details
