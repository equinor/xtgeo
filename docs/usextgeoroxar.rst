.. highlight:: python

===================
Use of XTGeo in RMS
===================

XTGeo can be incorporated within the  RMS user interface and share
some data with RMS. The integration will be continuosly improved.
Note that all these script examples are assumed to be ran inside
a python job in RMS.

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

    mygrid.to_file('topreek.gri')


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
