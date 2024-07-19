==================================
API Reference
==================================


Surfaces (maps)
---------------

RegularSurface
^^^^^^^^^^^^^^

Functions
"""""""""

.. autofunction:: xtgeo.surface_from_file

.. autofunction:: xtgeo.surface_from_cube

.. autofunction:: xtgeo.surface_from_grid3d

.. autofunction:: xtgeo.surface_from_roxar

Classes
"""""""

.. autoclass:: xtgeo.RegularSurface

    .. autoclasstoc::

Surfaces
^^^^^^^^

Classes
"""""""

.. autoclass:: xtgeo.Surfaces

    .. autoclasstoc::


Points and Polygons
-------------------

Points
^^^^^^

Functions
"""""""""

.. autofunction:: xtgeo.points_from_file

.. autofunction:: xtgeo.points_from_roxar

.. autofunction:: xtgeo.points_from_surface

.. autofunction:: xtgeo.points_from_wells

.. autofunction:: xtgeo.points_from_wells_dfrac

Classes
"""""""

.. autoclass:: xtgeo.Points

    .. autoclasstoc::

Polygons
^^^^^^^^

Functions
"""""""""

.. autofunction:: xtgeo.polygons_from_file

.. autofunction:: xtgeo.polygons_from_roxar

.. autofunction:: xtgeo.polygons_from_wells

Classes
"""""""

.. autoclass:: xtgeo.Polygons

    .. autoclasstoc::


Wells
-----

Well (single)
^^^^^^^^^^^^^

Functions
"""""""""

.. autofunction:: xtgeo.well_from_file

.. autofunction:: xtgeo.well_from_roxar

Classes
"""""""

.. autoclass:: xtgeo.Well

    .. autoclasstoc::

Wells (multiple)
^^^^^^^^^^^^^^^^

Classes
"""""""

.. autoclass:: xtgeo.Wells

    .. autoclasstoc::

Blocked well (single)
^^^^^^^^^^^^^^^^^^^^^

Functions
"""""""""

.. autofunction:: xtgeo.blockedwell_from_file

.. autofunction:: xtgeo.blockedwell_from_roxar

Classes
"""""""

.. autoclass:: xtgeo.BlockedWell

    .. autoclasstoc::

Blocked wells (multiple)
^^^^^^^^^^^^^^^^^^^^^^^^

Functions
"""""""""

.. autofunction:: xtgeo.blockedwells_from_roxar

Classes
"""""""

.. autoclass:: xtgeo.BlockedWells

    .. autoclasstoc::


Cubes (e.g. seismic)
--------------------

Cube
^^^^

Functions
"""""""""

.. autofunction:: xtgeo.cube_from_file

.. autofunction:: xtgeo.cube_from_roxar

Classes
"""""""

.. autoclass:: xtgeo.Cube

    .. autoclasstoc::

3D grids and properties
-----------------------

Grid
^^^^

Functions
"""""""""

.. autofunction:: xtgeo.grid_from_file

.. autofunction:: xtgeo.grid_from_roxar

Classes
"""""""

.. autoclass:: xtgeo.Grid

    .. autoclasstoc::

Grid property (single)
^^^^^^^^^^^^^^^^^^^^^^

Functions
"""""""""

.. autofunction:: xtgeo.gridproperty_from_file

.. autofunction:: xtgeo.gridproperty_from_roxar

Classes
"""""""

.. autoclass:: xtgeo.GridProperty

    .. autoclasstoc::

Grid properties (multiple)
^^^^^^^^^^^^^^^^^^^^^^^^^^

Classes
"""""""

.. autoclass:: xtgeo.GridProperties

    .. autoclasstoc::


Other
-----

Roxar utilities
^^^^^^^^^^^^^^^

RoxUtils
""""""""

.. autoclass:: xtgeo.RoxUtils

    .. autoclasstoc::

Metadata (experimental)
^^^^^^^^^^^^^^^^^^^^^^^

MetadataRegularSurface
""""""""""""""""""""""

.. autoclass:: xtgeo.MetaDataRegularSurface

    .. autoclasstoc::

MetaDataRegularCube
"""""""""""""""""""

.. autoclass:: xtgeo.MetaDataRegularCube

    .. autoclasstoc::

MetaDataCPGeometry
""""""""""""""""""

.. autoclass:: xtgeo.MetaDataCPGeometry

    .. autoclasstoc::

MetaDataCPProperty
""""""""""""""""""

.. autoclass:: xtgeo.MetaDataCPProperty

    .. autoclasstoc::

MetaDataWell
""""""""""""

.. autoclass:: xtgeo.MetaDataWell

    .. autoclasstoc::
