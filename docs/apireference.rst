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
    :members:
    :show-inheritance:
    :inherited-members:
    :special-members: __init__

    .. autoclasstoc::

Surfaces
^^^^^^^^

Classes
"""""""

.. autoclass:: xtgeo.Surfaces
    :members:
    :show-inheritance:
    :inherited-members:
    :special-members: __init__

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
    :members:
    :show-inheritance:
    :inherited-members:
    :special-members: __init__

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
    :members:
    :show-inheritance:
    :inherited-members:
    :special-members: __init__

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
    :members:
    :show-inheritance:
    :inherited-members:
    :special-members: __init__

    .. autoclasstoc::

Wells (multiple)
^^^^^^^^^^^^^^^^

Classes
"""""""

.. autoclass:: xtgeo.Wells
    :members:
    :show-inheritance:
    :inherited-members:
    :special-members: __init__

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
    :members:
    :show-inheritance:
    :inherited-members:
    :special-members: __init__

    .. autoclasstoc::

Blocked wells (multiple)
^^^^^^^^^^^^^^^^^^^^^^^^

Functions
"""""""""

.. autofunction:: xtgeo.blockedwells_from_roxar

Classes
"""""""

.. autoclass:: xtgeo.BlockedWells
    :members:
    :show-inheritance:
    :inherited-members:
    :special-members: __init__

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
    :members:
    :show-inheritance:
    :inherited-members:
    :special-members: __init__

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
    :members:
    :show-inheritance:
    :inherited-members:
    :special-members: __init__

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
    :members:
    :show-inheritance:
    :inherited-members:
    :special-members: __init__

    .. autoclasstoc::

Grid properties (multiple)
^^^^^^^^^^^^^^^^^^^^^^^^^^

Classes
"""""""

.. autoclass:: xtgeo.GridProperties
    :members:
    :show-inheritance:
    :inherited-members:
    :special-members: __init__

    .. autoclasstoc::


Other
-----

Roxar utilities
^^^^^^^^^^^^^^^

RoxUtils
""""""""

.. autoclass:: xtgeo.RoxUtils
    :members:
    :show-inheritance:
    :inherited-members:
    :special-members: __init__

    .. autoclasstoc::

Metadata (experimental)
^^^^^^^^^^^^^^^^^^^^^^^

MetadataRegularSurface
""""""""""""""""""""""

.. autoclass:: xtgeo.MetaDataRegularSurface
    :members:
    :show-inheritance:
    :inherited-members:
    :special-members: __init__

    .. autoclasstoc::

MetaDataRegularCube
"""""""""""""""""""

.. autoclass:: xtgeo.MetaDataRegularCube
    :members:
    :show-inheritance:
    :inherited-members:
    :special-members: __init__

    .. autoclasstoc::

MetaDataCPGeometry
""""""""""""""""""

.. autoclass:: xtgeo.MetaDataCPGeometry
    :members:
    :show-inheritance:
    :inherited-members:
    :special-members: __init__

    .. autoclasstoc::

MetaDataCPProperty
""""""""""""""""""

.. autoclass:: xtgeo.MetaDataCPProperty
    :members:
    :show-inheritance:
    :inherited-members:
    :special-members: __init__

    .. autoclasstoc::

MetaDataWell
""""""""""""

.. autoclass:: xtgeo.MetaDataWell
    :members:
    :show-inheritance:
    :inherited-members:
    :special-members: __init__

    .. autoclasstoc::

Plot (to be deprecated)
^^^^^^^^^^^^^^^^^^^^^^^

XSection
""""""""

.. autoclass:: xtgeo.plot.XSection
    :members:
    :show-inheritance:
    :inherited-members:
    :special-members: __init__

    .. autoclasstoc::

Map
"""

.. autoclass:: xtgeo.plot.Map
    :members:
    :show-inheritance:
    :inherited-members:
    :special-members: __init__

    .. autoclasstoc::

Grid3DSlice
"""""""""""

.. autoclass:: xtgeo.plot.Grid3DSlice
    :members:
    :show-inheritance:
    :inherited-members:
    :special-members: __init__

    .. autoclasstoc::
