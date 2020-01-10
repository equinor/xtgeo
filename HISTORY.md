# History

## Version 2

### 2.4.2
* Fix a bug that occurs when reading Eclipse properties from E300 runs

### 2.4.1
* Push to trigger travis build and deploy

### 2.4.0

* New features:
  * Added a general kwargs to `savefig()` in plot module, so e.g. dpi keyword can be passed to matplotlib
* Bug fixes:
  * More robust on reading saturations from UNRST files from Eclipse 300 and IX, where "IPHS" metadata
    (describing phases present) is unreliable.
* Fixes for developers:
  *  setup can now be ran in "develop mode"

### 2.3.1

* Preliminary support for Python 3.8 (Linux only)
* Several bug fixes:
  * User warning when requested colour map is not found
  * Printing of a Points of Polygons instance shall now work
  * UNDEF values in property grdecl or bgrdecl export shall now be 0.0, not a large number
  * Name in GridProperty `to_file(name=...)` is fixed
  * If `fformat` in GridProperty import is mispelled, an exception will be raised

### 2.3.0

* Added support for MacOS on PYPI (Python 3.6, 3.7)
* Added functionality on grid slices as method ()
* More flexible reading on phases present in Eclipse/IX UNRST files
* Several minor bugfixes and improvements

### 2.2.2

* Several smaller bug fixes
* Use of realisation in gridproperty_from_roxar() was not working

### 2.2.1

* Full C code and compile restructuring, now using scikit-build!
* Use of realisation in gridproperty_from_roxar() was not working

### 2.2.0

Several fixes and new features, most important:

  * Well() class
    * Added tvd interval for rescaling of well logs.
    * When sampling a discrete property to well, it will now be a discrete log
    * Added a isdiscrete() method
  * RegularSurface() class
    * Support for read from bytestrings (memory) in addition to files (Irap binary format supported)
    * Fast load of surfaces (will only read metadata) if requested
    * Support for threading/multiprocessing (concurrent.futures) when importing surfaces from Irap binary.
  * Grid() class
    * Improvements and fixes for dual porosity and/or dual permeability models from Eclipse


### 2.1.0

Several fixes and new features, most important:

  * Cube() class
    * A general get_randomline() methods
  * Grid() class
    * Make a rectular shoebox grid
    * Get a randomline (sampling) along a 3D grid with property
    * More robust support for binary GRDECL format
    * Possible to input dual porosity models from Eclipse (EGRID, INIT, UNRST)
  * Surfaces
    * Added a class for Surfaces(), a collection of RegularSurface instances
    * Generate surface from 3D grid
    * Lazy load of RegularSurfaces (if ROFF/RMS binary) for fast scan of metadata
    * Clipboard support in from_roxar() and to_roxar() methods
    * fill(), fast infill of undefined values
    * smooth(), median smoothing
    * get_randomline() method (more general and flexible)

  * Points/polygons
    * Added copy() method
    * Added snap to surface method (snap_surface)
    * Several other methods related to xsections from polygons

  * Well() class
    * Get polygon and and improved fence from well trajectory
    * Look up IJK indices in 3D grid from well path

### 2.0.8

* Fixed a backward compatibility issue with `filter` vs `pfilter` for points/polygons `to_file`

### 2.0.7

* (merged into 2.0.8)

### 2.0.6

* Corrected issues with matplotlib when loading xtgeo in RMS

### 2.0.5

* Fixed a bug when reading grids in ROXAR API, the subgrids were missing
* Improved logo and documentation runs
* Allow for xtgeo.ClassName() as well as xtgeo.submodule.ClassName()
* A number of smaller Fixes
* More badges

### 2.0.4

* Technical fixes regarding numpy versions vs py version, swig setup and setup.py

### 2.0.3

* Deploy to python 3.4 and 3.5 also. Numpy versions tuned to match roxar library.

### 2.0.2

* Adding services for code improvements (codacy, bandit)

### 2.0.1

* Minor improvements in setup and documentation
* Travis automatic deploy works now

### 2.0.0

* First version after Open Sourcing to LGPL v3+

## Version 0 and 1

See github for commit and tag history:

https://github.com/equinor/xtgeo
