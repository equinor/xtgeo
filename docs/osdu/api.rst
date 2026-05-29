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
