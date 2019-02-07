=============================
The *XTGeo* library
=============================
.. image:: https://travis-ci.com/equinor/xtgeo.svg?token=c9LYyqv6MtDXz4Cxbq9H&branch=master
    :target: https://travis-ci.com/equinor/xtgeo


In-house Python class library for surfaces, wells, 3D grids, etc
mostly in relation with RMS and geo work.


Features
--------

Easy access to basic data, with manipulation in numpy and/or pandas. E.g.

::

   from xtgeo.surface import RegularSurface

   # create an instance of a surface, read from file
   mysurf = RegularSurface('myfile.gri')  # Irap binary as default

   print('Mean is {}'.format(mysurf.mean()))

   # change date so all values less than 2000 becomes 2000
   # The values attribute gives the Numpy array

   mysurface.values[mysurface.values < 2000] = 2000

   # export the modified surface:
   mysurface.to_file('newfile.gri')
