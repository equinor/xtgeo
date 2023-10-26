.. highlight:: python

==============================
Examples in standalone scripts
==============================

.. comments
   These examples are ran in Jupyter notebook...

XTGeo is Python library to work with surfaces, grids, cubes, wells, etc,
possibly in combinations. It is easy to make small user scripts that runs from
the command line in Linux, Mac and Windows.

------------------
Surface operations
------------------

See class :class:`~xtgeo.surface.regular_surface.RegularSurface` for details on
available methods and attributes.

Initialising a Surface object (instance)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import xtgeo

   # initialising a RegularSurface object:

   # this makes a surface from scratch
   surf = xtgeo.RegularSurface(ncol=33, nrow=50,
                               xori=34522.22, yori=6433231.21,
                               xinc=40.0, yinc=40.0, rotation=30,
                               values=np.zeros((33,50)))

   # a more common method is to make an instance from file:

   surf = xtgeo.surface_from_file("somename.gri")


Surface object properties
^^^^^^^^^^^^^^^^^^^^^^^^^

A Surface object will have a number of so-called properties,
see :class:`~xtgeo.surface.regular_surface.RegularSurface`. Some
of these properties can be changed, which actually changes the map

.. code-block:: python

   import xtgeo

   surf3 =xtgeo.surface_from_file('reek.gri')

   print(surf3)  # will show a description

   print(surf3.xinc, surf3.yinc)

   print(surf3.rotation)

   # change the rotation:
   surf3.rotation = 45.0

   # move the surface 1000 m to west:
   surf3.xori -= 1000.0

   # export the modified surface
   surf3.to_file('changedsurface.gri')  # irap binary is default

   # Note that changing `nrow` and `ncol` is not possible to do directly.


Sample a surface from a 3D grid
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../examples/surface_slice_grid3d.py
   :language: python

Sample a surface or a window attribute from a cube
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../examples/surface_slice_cube.py
   :language: python

---------------
Cube operations
---------------

Taking diff of two cubes and export, in SEGY
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It is easy to take the difference bewteen two cubes and export to SEGY
format. Here is an example:

.. code-block:: python

   import xtgeo

   # make two cube objects from file import:
   cube1 = xtgeo.cube_from_file('cube1.segy')
   cube2 = xtgeo.cube_from_file('cube2.segy')

   # subtract the two numpy arrays
   cube1.values = cube1.values - cube2.values

   # export the updated cube1 to SEGY
   cube1.to_file('diff.segy')


Reduce cube (e.g. SEGY) data by thinning and cropping
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Here is a big data set which gets heavily reduced by thinning and cropping.
These are very quick operations! Note that cropping has two numbers (tuple) for each
direction, e.g. (20, 30) means removal of 20 columns from front,
and 30 from back. The applied order of these routines matters...

.. code-block:: python

   import xtgeo

   big = xtgeo.cube_from_file("troll.segy")
   big.do_thinning(2, 2, 1)  # keep every second inline and xline
   big.do_cropping((20, 30), (250, 20), (0, 0))  # crop ilines and xlines

   # export a much smaller file to SEGY
   big.to_file("much_smaller.segy")


Reduce or change cube (e.g. SEGY) data by resampling
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Here is a big data set which gets heavily reduced by making a new cube
with every second inline and xline, and then resample values prior to export.

Also, another small cube with another rotation is made:

.. code-block:: python




   import xtgeo

   big = xtgeo.cube_from_file("troll.segy")

   # make a cube of every second iline and xline
   newcube = xtgeo.Cube(xori=big.xori, yori=big.yori, zori=big.zori,
                        xinc=big.xinc * 2,
                        yinc=big.yinc * 2,
                        zinc=big.zinc,
                        ncol=int(big.ncol / 2),
                        nrow=int(big.nrow / 2),
                        nlay=big.nlay,
                        rotation=big.rotation,
                        yflip=big.yflip)

   newcube.resample(big)

   newcube.to_file('newcube.segy')

   # you can also make whatever cube you want with e.g. another rotation

   smallcube = xtgeo.Cube(xori=523380, yori=6735680, zori=big.zori,
                           xinc=50,
                           yinc=50,
                           zinc=12,
                           ncol=100,
                           nrow=200,
                           nlay=100,
                           rotation=0.0,
                           yflip=big.yflip)

   smallcube.resample(big)

   smallcube.to_file('smallcube.segy')

------------------------------------
Combined Surface and Cube operations
------------------------------------

To sample cube values into a surface can be quite useful. Both direct
sampling, and interval sampling (over a window, or between two surfaces)
is supported. For the interval sampling, various attributes can be
extracted.

Sampling a surface from a cube
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Here is sampling a regular surface from a cube. The two objects can have
different rotation. See :meth:`xtgeo.surface.RegularSurface.slice_cube` method

.. code-block:: python

   import xtgeo

   # make two cube objects from file import:
   surf = xtgeo.surface_from_file('top.gri')
   cube = xtgeo.cube_from_file('cube2.segy')

   surf.slice_cube(cube)

   # export the updated to RMS binary map format
   surf.to_file('myslice.gri')


Sampling the root-mean-square surface over a window from a cube
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The root mean scquare (rms) value over a surface, +- 10 units
(e.g. metres if depth), see `slice_cube_window` method.

.. code-block:: python

   import xtgeo

   # slice within a window (vertically) around surface:
   surf = xtgeo.surface_from_file('top.gri')
   cube = xtgeo.cube_from_file('cube.segy')

   surf.slice_cube_window(cube, zrange=10, attribute='rms')

   # export the updated to Irap (RMS) ascii map format
   surf.to_file('rmsaverage.fgr', fformat='irap_ascii')

----------------
3D grid examples
----------------


Crop a 3D grid with properties
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../examples/grid3d_crop.py
   :language: python


Extract Pandas dataframe from 3D grid and props
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../examples/grid3d_get_df.py
   :language: python


Compute a grid property average and stdev
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this example, how to extract Mean ans Stddev from
some geo properties, filtered on facies. An RMS inside
version is also shown.

.. literalinclude:: ../examples/grid3d_properties_qc.py
   :language: python


Compute a grid property average across realisations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this example, a technique that keeps memory usage
under control when computing averages is also presented.

.. literalinclude:: ../examples/grid3d_compute_stats.py
   :language: python

Make a CSV file from Eclipse INIT data (aka ERT ECL)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Example on how to create a CSV file from all INIT
properties. Example is for Eclipse format, but shall
work also with ROFF input.

.. literalinclude:: ../examples/grid3d_print_init_csv.py
   :language: python
