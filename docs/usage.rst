.. highlight:: python

=====
Usage
=====

.. comments
   These examples are ran in Jupyter notebook...

XTGeo is python libraray to work with surfaces, grids, cubes, wells, etc,
possibly in combinations.

------------------
Surface operations
------------------

Initialising a Surface object (instance)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import xtgeo

   # initialising a RegularSurface object:

   # this makes a surface from scratch
   surf = xtgeo.surface.RegularSurface(ncol=33, nrow=50,
                                       xori=34522.22, yori=6433231.21,
                                       xinc=40.0, yinc=40.0, rotation=30,
                                       values=np.zeros((33,55))

   # a more common method is to make an instance from file
   # there are some variant on how to to this:

   # 1)
   surf1 = xtgeo.surface.RegularSurface()
   surf1.from_file('reek.gri', fformat='irap_binary')

   # 2)
   surf2 = xtgeo.surface.RegularSurface('reek.gri')  # irap binary is default

   # 3)  problably simplest
   surf3 =xtge.surface_from_file('reek.gri')


Surface object properties
^^^^^^^^^^^^^^^^^^^^^^^^^

A Surface object will have a number of so-called properties.
See `xtgeo.surface.RegularSurface`. Some
of these properties can be changed, which actually changes the map

.. code-block:: python

   import xtgeo

   surf3 =xtge.surface_from_file('reek.gri')

   print(surf3.xinc, surf3.yinc)

   print(surf3.rotation)

   # change the rotation:
   surf3.rotation = 45.0

   # move the surface 1000 m to west:
   surf3.xori -= 1000.0

   # export the modified surface
   surf3.to_file('changedsurface.gri')  # irap binary is default

   # Note that changing nrow and ncol is not possible to do directly.
