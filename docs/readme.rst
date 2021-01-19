
Introduction
============

XTGeo is a `LGPL`_ licensed Python library with C backend to support
manipulation of (oil industry) subsurface reservoir modelling. Typical
users are geoscientist and reservoir engineers working with reservoir
modelling.


XTGeo main environment is as stand-alone Python script or notebook, but can
*optionally* be ran inside `RMS`_'s python. XTGeo is developed in `Equinor`_.

Feature summary
---------------

-  Python 3.6+ (Linux, Windows and MacOS).
-  Focus on high speed, using `numpy`_ and `pandas`_ with C backend
-  Regular surfaces, i.e. 2D maps with regular sampling and rotation
-  3D grids (corner-point), supporting several formats such as RMS and
   Eclipse
-  Support of seismic cubes, using `segyio`_ as backend for SEGY format
-  Support of well data, line and polygons (still somewhat immature)
-  Operations between the data types listed above; e.g. slice a surface
   with a seismic cube
-  Integration to ROXAR API python for several data types is supported
   (see note later)

Quick Installation
------------------

PYPI installation is enabled for all supported platforms:

.. code:: bash

   pip install xtgeo

For detailed installation instructions (implies C compiling), see
:doc:`installation`.


Getting started
---------------

.. code:: python

   import xtgeo

   # create an instance of a surface, read from file
   mysurf = xtgeo.RegularSurface("myfile.gri")  # Irap binary as default

   print(f"Mean is {mysurf.values.mean()}")

   # change date so all values less than 2000 becomes 2000
   # The values attribute gives the Numpy array

   mysurface.values[mysurface.values < 2000] = 2000

   # export the modified surface:
   mysurface.to_file("newfile.gri")

Note on RMS Roxar API integration
---------------------------------

The following applies to the part of the XTGeo API that is connected to
Roxar API (RMS):

*RMS is neither an open source software nor a free software and any use of it needs
a software license agreement in place.*

.. _segyio: https://github.com/equinor/segyio
.. _Equinor: https://equinor.com
.. _pandas: https://pandas.pydata.org/
.. _numpy: https://numpy.org/
.. _LGPL: https://en.wikipedia.org/wiki/GNU_Lesser_General_Public_License
.. _RMS: https://www.emerson.com/no-no/automation/operations-business-management/reservoir-management-software
