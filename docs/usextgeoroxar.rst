.. highlight:: python

===================
Use of XTGeo in RMS
===================

XTGeo can be incorporated within the RMS user interface and share
data with RMS. The integration will be continuosly improved.
Note that all these script examples are assumed to be ran inside
a python job within RMS.

Surface data
------------

Here are some simple examples on how to use XTGeo to interact with
RMS data, and e.g. do quick exports to files.

Export a surface in RMS to irap binary format
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: python

    import xtgeo

    # import (transfer) data from RMS to XTGeo and export
    surf = xtgeo.surface_from_roxar(project, 'TopReek', 'DS_extracted')

    surf.to_file('topreek.gri')

    # modify surface, add 1000 to all map nodes
    surf.values += 1000

    # store in RMS (category must exist)
    surf.to_roxar(project, 'TopReek', 'DS_whatever')


Export a surface in RMS to zmap ascii format
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Note here that an automatic resampling to a nonrotated regular
grid will be done in case the RMS map has a rotation.

.. code-block:: python

    import xtgeo as xt

    # surface names
    hnames = ['TopReek', 'MiddleReek', 'LowerReek']

    # loop over stratigraphy
    for name in hnames:
        surf = xt.surface_from_roxar(project, name, 'DS_extracted')
        fname = name.lower()  # lower case file name
        surf.to_file(fname + '.zmap', fformat='zmap_ascii')

    print('Export done')

Take a surface in RMS and multiply values with 2:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import xtgeo

    surf = xtgeo.surface_from_roxar(project, 'TopReek', 'DS_tmp')

    surf.values *= 2  # values is the masked 2D numpy array property

    # store the surface back to RMS
    surf.to_roxar(project, 'TopReek', 'DS_tmp')


3D grid data
------------

Exporting geometry to ROFF file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import xtgeo

    # import (transfer) data from RMS to XTGeo and export
    mygrid = xtgeo.grid_from_roxar(project, 'Geomodel')

    mygrid.to_file('topreek.roff')  # roff binary is default format


Edit a porosity in a 3D grid
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import xtgeo

    # import (transfer) data from RMS to XTGeo
    myporo = xtgeo.gridproperty_from_roxar(project, 'Geomodel', 'Por')

    # now I want to limit porosity to 0.35 for values above 0.35:

    poro.values[poro_values > 0.35] = 0.35

    # store to another icon
    poro.to_roxar(project, 'Geomodel', 'PorNew')

Edit a 3D grid porosity inside polygons
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Example where I want to read a 3D grid porosity, and set value
   # to 99 inside polygons

   import xtgeo

   mygrid = xtgeo.grid_from_roxar(project, 'Reek_sim')
   myprop = xtgeo.gridproperty_from_roxar(project, 'Reek_sim', 'PORO')

   # read polygon(s), from Horizons, Faults, Zones or Clipboard
   mypoly = xtgeo.polygons_from_roxar(project, 'TopUpperReek', 'DL_test')

   # need to connect property to grid geometry when using polygons
   myprop.geometry = mygrid

   myprop.set_inside(mypoly, 99)

   # Save in RMS as a new icon
   myprop.to_roxar(project, 'Reek_sim', 'NEWPORO_setinside')


Cube data
---------

Slicing a surface in a cube
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Examples to come...

Well data
---------

Examples to comes...


Line point data
---------------

Examples to comes...
