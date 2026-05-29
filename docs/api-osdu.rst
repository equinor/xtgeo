OSDU / RESQML Interface
-----------------------

.. note::

   The OSDU documentation has been reorganised into a dedicated section with
   user guide, API reference, design docs, and demos. See :doc:`osdu/index`.

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
    session = OsduSession.load("equinor-dev")

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

High-level functions
^^^^^^^^^^^^^^^^^^^^

Discovery
"""""""""

.. autofunction:: xtgeo.list_osdu_objects
   :no-index:

.. autofunction:: xtgeo.search_osdu
   :no-index:

Reading
"""""""

.. autofunction:: xtgeo.grid_from_osdu
   :no-index:

.. autofunction:: xtgeo.surface_from_osdu
   :no-index:

.. autofunction:: xtgeo.points_from_osdu
   :no-index:

.. autofunction:: xtgeo.polygons_from_osdu
   :no-index:

Writing
"""""""

.. autofunction:: xtgeo.grid_to_osdu
   :no-index:

.. autofunction:: xtgeo.surface_to_osdu
   :no-index:

.. autofunction:: xtgeo.points_to_osdu
   :no-index:

.. autofunction:: xtgeo.polygons_to_osdu
   :no-index:


Session management
^^^^^^^^^^^^^^^^^^

.. autoclass:: xtgeo.interfaces.osdu.OsduSession
   :no-index:
   :members: access_token, etp_config, create_dataspace_rest, create_dataspace_etp,
             list_dataspaces, get_dataspace, delete_dataspace, list_objects_rest,
             search_objects_rest, get_object_metadata_rest, switch_dataspace,
             save, load, from_env, list_profiles

Providers
^^^^^^^^^

.. autoclass:: xtgeo.interfaces.osdu.EtpProvider
   :no-index:
   :members: open, close, list_objects

.. autoclass:: xtgeo.interfaces.osdu.EpcFileProvider
   :no-index:
   :members: open, close, list_objects

.. autoclass:: xtgeo.interfaces.osdu.EtpConnectionConfig
   :no-index:

Dataspace operations
^^^^^^^^^^^^^^^^^^^^

For advanced bulk operations (copy entire dataspaces, compare datasets):

.. autofunction:: xtgeo.interfaces.osdu.read_dataspace
   :no-index:

.. autofunction:: xtgeo.interfaces.osdu.write_dataspace
   :no-index:

.. autofunction:: xtgeo.interfaces.osdu.compare_snapshots
   :no-index:

.. autoclass:: xtgeo.interfaces.osdu.DataspaceSnapshot
   :no-index:

Low-level converters
^^^^^^^^^^^^^^^^^^^^

.. autofunction:: xtgeo.interfaces.osdu.ijk_grid_to_xtgeo
   :no-index:

.. autofunction:: xtgeo.interfaces.osdu.xtgeo_grid_to_resqml
   :no-index:

.. autofunction:: xtgeo.interfaces.osdu.grid2d_to_xtgeo
   :no-index:

.. autofunction:: xtgeo.interfaces.osdu.xtgeo_surface_to_resqml
   :no-index:

.. autofunction:: xtgeo.interfaces.osdu.pointset_to_xtgeo
   :no-index:

.. autofunction:: xtgeo.interfaces.osdu.xtgeo_points_to_resqml
   :no-index:

.. autofunction:: xtgeo.interfaces.osdu.polylineset_to_xtgeo
   :no-index:

.. autofunction:: xtgeo.interfaces.osdu.xtgeo_polygons_to_resqml
   :no-index:

CRS and metadata
^^^^^^^^^^^^^^^^

.. autoclass:: xtgeo.interfaces.osdu.LocalDepth3dCrs
   :no-index:

.. autofunction:: xtgeo.interfaces.osdu.resolve_property_mapping
   :no-index:

.. autofunction:: xtgeo.interfaces.osdu.ecl_keyword_to_osdu
   :no-index:

.. autofunction:: xtgeo.interfaces.osdu.osdu_name_to_ecl_keyword
   :no-index:
