.. highlight:: python

===================
Use of XTGeo in RMS
===================

XTGeo can be incorporated with RMS and share some data with RMS. The integration
will be continuosly improved.

Surface data
------------

Here are some examples.

Export a surface in RMS to irap binary format
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: python

    import xtgeo

    # import (transfer) data from RMS to XTGeo and export
    surf = xtgeo.surface_from_roxar(project, 'TopReek', 'DS_extracted')

    surf.to_file('topreek.gri')

Export a surface in RMS to zmap ascii format
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Note here that an automatic resampling to a regular grid will be
done in case the RMS map has rotation.

.. code-block:: python

    import xtgeo

    surf = xtgeo.surface_from_roxar(project, 'TopReek', 'DS_extracted')

    surf.to_file('topreek.zmap', fformat='zmap_ascii')

Take a surface in RMS and multiply values with 2:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import xtgeo

    surf = xtgeo.surface_from_roxar(project, 'TopReek', 'DS_tmp')

    surf.values *= 2  # values is the masked 2D numpy array

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
