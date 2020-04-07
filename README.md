![XTGeo](https://github.com/equinor/xtgeo/blob/master/docs/images/xtgeo-logo-wide.png)

[![Build Status](https://travis-ci.com/equinor/xtgeo.svg?branch=master)](https://travis-ci.com/equinor/xtgeo)
[![Build status](https://ci.appveyor.com/api/projects/status/g795rox7g4k23l1q?svg=true)](https://ci.appveyor.com/project/jcrivenaes/xtgeo)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/c209aeed6a2a40b08ea859aeadf31cb0)](https://www.codacy.com/app/jcrivenaes/xtgeo?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=equinor/xtgeo&amp;utm_campaign=Badge_Grade)
[![codecov](https://codecov.io/gh/equinor/xtgeo/branch/master/graph/badge.svg)](https://codecov.io/gh/equinor/xtgeo)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)
[![PyPI version](https://badge.fury.io/py/xtgeo.svg)](https://badge.fury.io/py/xtgeo)
[![Documentation Status](https://readthedocs.org/projects/xtgeo/badge/?version=latest)](https://xtgeo.readthedocs.io/en/latest/?badge=latest)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/xtgeo.svg)
![PyPI - License](https://img.shields.io/pypi/l/xtgeo.svg)

## Introduction ##

XTGeo is a LGPL licensed Python library with C backend to support
manipulation of (oil industry) subsurface reservoir modelling. Typical
users are geoscientist and reservoir engineers working with
reservoir modelling, in relation with RMS. XTGeo is developed in Equinor.

Detailed documentation for [XTGeo at Read *the* Docs](https://xtgeo.readthedocs.io)

## Feature summary ##

   * Python 2.7+ and 3.5+ support
   * Focus on high speed, using numpy and pandas with C backend
   * Regular surfaces, i.e. 2D maps with regular sampling and rotation
   * 3D grids (corner-point), supporting several formats such as
     RMS and Eclipse
   * Support of seismic cubes, using
     [segyio](https://github.com/equinor/segyio) as backend for SEGY format
   * Support of well data, line and polygons (still somewhat immature)
   * Operations between the data types listed above; e.g. slice a surface
     with a seismic cube
   * Optional integration with ROXAR API python for several data types
     (see note later)
   * Linux is main development platform, but Windows and MacOS (64 bit) are supported
     and PYPI wheels are provided.

## Installation ##

For Linux, Windows and MacOS 64bit, PYPI installation is enabled:

```
pip install xtgeo
```
For detailed installation instructions (implies C compiling), see
the documentation.

## Getting started ##

```python
from xtgeo.surface import RegularSurface

# create an instance of a surface, read from file
mysurf = RegularSurface('myfile.gri')  # Irap binary as default

print('Mean is {}'.format(mysurf.values.mean()))

# change date so all values less than 2000 becomes 2000
# The values attribute gives the Numpy array

mysurface.values[mysurface.values < 2000] = 2000

# export the modified surface:
mysurface.to_file('newfile.gri')
```

## Note on RMS Roxar API integration ##

The following applies to the part of the XTGeo API that is
connected to Roxar API (RMS):

> RMS is neither an open source software nor a free software and
> any use of it needs a software license agreement in place.
