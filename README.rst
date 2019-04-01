=============================
The *XTGeo* library
=============================
.. image:: https://travis-ci.com/equinor/xtgeo.svg?token=c9LYyqv6MtDXz4Cxbq9H&branch=master
    :target: https://travis-ci.com/equinor/xtgeo


XTGeo is Python class library with a C backend for subsurface work. It handles
many data types, such as surfaces, well data, 3D grids, etc. The usage is primarely
targeted to geoscientist and reservoir engineers working with reservoir modelling,
in relation with RMS.


Building XTGeo
--------------

XTGeo is developed on Linux. To build XTGeo you need:

* A C99 compatible C compiler (gcc is recommended)
* The SWIG library (http://www.swig.org/) version 2 or higher.
* Python 2.7, 3.4 or higher. Furter Python requirements
  are listed in requirements.txt, requirements_dev.txt and setup.py
* Detailed instructions are provided in installation.rst



Features
--------

To goal of XTGeo is to provide easy access to basic data, with manipulation in
numpy and/or pandas. For example:

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
