OSDU / RESQML Interface
-----------------------

.. note::

   The full OSDU documentation — user guide, API reference, developer guide,
   and demos — lives in a dedicated section. See :doc:`osdu/index`.

.. toctree::
   :hidden:

   osdu/index

.. versionadded:: 4.x

This module provides reading and writing of xtgeo objects to and from
`OSDU <https://community.opengroup.org/osdu>`_ Reservoir DDMS servers
(via ETP 1.2 WebSocket protocol) and RESQML 2.0.1 EPC+HDF5 file containers.

Quick start
^^^^^^^^^^^

.. code-block:: python

    import xtgeo
    from xtgeo.interfaces.osdu import OsduSession

    # Local dev (Docker RDDMS, zero config)
    session = OsduSession()

    # Or cloud (reads OSDU_* env vars automatically)
    session = OsduSession.from_env()

    # Or reload a saved profile
    session = OsduSession.load("my-cloud")

    # Search for objects
    grids = xtgeo.search_osdu(session, name="*Drogon*", object_type="grid")

    # Read with full properties
    grid, props = xtgeo.grid_from_osdu(session, name="Drogon")
    surf = xtgeo.surface_from_osdu(session, name="TopVolantis")

    # Write back (numerically exact roundtrip for Eclipse grids)
    xtgeo.grid_to_osdu(session, grid, title="MyGrid", properties=props, crs_epsg=23031)
    xtgeo.surface_to_osdu(session, surf, title="TopReek", crs_epsg=23031)

    # Or use EPC files (same API, just pass a path)
    grid, props = xtgeo.grid_from_osdu("model.epc", name="Drogon")

For the complete API reference, user guide, and developer documentation, see
the :doc:`osdu/index` section.
