# Release notes

## Version 2.15

* Pending API changes:
  * The initialisation of different objects in xtgeo currently manipulates an
    instance of itself.  We want to deprecate said behaviour and instead return a
    new instance on each creation. This work have now started, but currently it will
    only produce a warning. In the next major we will deprecate relevant behaviour.
  * The following classes will undergo changes: `Grid`, `GridProperty`,
    `RegularSurface`, `Points`, `Polygons`, `Well` and `Cube`, but at this stage only
    `RegularSurface` is affected.
  * For example:

    There are currently multiple ways of creating e.g. a `RegularSurface` in `XTGeo`:

    1) Creating an instance and then populating it by reading a file:

      >> `surf = RegularSurface()`
      >> `surf.from_file('myfile.gri')`

    2) Providing a file (any supported file format can be given):

      >> `surf = RegularSurface('myfile.gri')`

    3) Using one of the wrapper functions (similarly with cube, roxar, grid):
      >> `surf = xtgeo.surface_from_file('myfile.gri')`

    These methods are consistent for all classes in `XTGeo`.

    Methods 1 and 2 will be deprecated, while using the wrapper function (3) will
    be the preferred approach.

  * Until version 2.14, creating an empty `RegularSurface` provided a default
    surface with `ncol=5`, `nrow=3`, `xinc=25`, `yinc=25` and the following values:
    `[[1, 6, 11], [2, 7, 12], [3, 8, 1e33], [4, 9, 14], [5, 10, 15]]`.
    Creating an empty `RegularSurface` in this version will still provide the default
    structure, but in version 3 values will be set to `0` instead.
    Creating an empty `RegularSurface` will however display a deprecation warning, and
    warns that `xinc`, `yinc`, `ncol` and `nrow` should be set explicitly.


* New features:
  * Python 3.9 support, with PYPI wheel.
  * Add the possibility to write Well to Roxar API #528
  * Read and write ROFF on ASCII now supported #549
  * Add warning for roxar grid with dual index #495 #491 #306

* Improvements and bug fixes:
  * GRDECL file format reading now handles repeat counts. #435
  * Read and write of ROFF files have been refactored and moved to separate
    package (roffio). Speed is improved, and ASCII roff is now supported.
    This solved several bugs and issues related to roff reading, such as handling
    large file sizes and long keywords. #549 #548 #537 #536
  * Fix: Reading a discrete property from file and save in Roxar API is
    now corrected #531.
  * Fix: handling of file extensions with multiple "." in name #592
  * Fix: MAPAXES being ignored when last in file in Eclipse formats #581
  * Fix: handling of subgrids in hybrid grids #553
  * Several cases of non-recoverable exit have now been replaced by an
    Exception #552 #545
  * Fix: a memory access violation issue in translate_coordinates for 3D grids #550
  * Fix: invalid read in surf_get_z_from_ij (C backend) #551
  * Fix: Segfaulting when using get_value_from_xy (C backend) #496
  * Fix: Checks on indices in set_grid in roxar api #578
  * Change datatype for counters, related to #537
  * Fix: Check validity on colnames when setting dataframe for Polygons #506
  * Fix: GridProperty accepts arbitrary line and keyword length when reading GRDECL #504
  * Fix: GridProperty would not accept cases with inactive cells when
    reading GRDECL (introduced in XTGeo 2.6-2.10) #507
  * Fix: Write to roxapi discrete prop #531

### Version 2.15.1

* Bugfixes:
  * General improvements when i/o with binary GRDECL and EGRID formats
  * Fixing that egrid sometimes omitted ACTNUM #599
  * Fix rounding error in ZINC when exporting cubes to SEGY #616

## Version 2.14

* New features:
  * For some methods in RegularSurface(), a ``sampling`` key is added which in
    addition to default `bilinear` interpolation also can use `nearest node` sampling.
    The latter can be useful e.g. for discrete maps such as facies. Closes #462.
  * For Wells(), add a method ``create_surf_distance_log`` that makes a well log
    that is distance to a surface, #461.
  * For Wells(), add a method that can remove (mask) data based on discrete logs, i.e.
    remove shoulder-bed effects, #457.
  * A pre-release (i.e. experimental!) support for free-form metadata and new formats:
    * A separate MetaData class for each major data-type.
    * Native XTGeo formats for some data types. Should be very fast and support metadata.
    * Support for HDF-5 formats, should be quite/very fast, support metadata and
      compression.
    * In relation to this, more automatic format detection is initiated. This means
      in practice that the key ``fformat`` and/or file extension rules will not be
      needed in many cases, as xtgeo will detect the file format by inspection the file
      itself. This is in particular useful for binary memory streams.
    * As a side note, the HDF support change the external requirements for xtgeo as e.g.
      the `h5py` package and more are now needed.

* Improvements and bug fixes:
  * Improvements on missing or wrong ``values`` key when dealing with RegularSurface(),
    relates to bug report #450.
  * When reading a discrete 3D property from Roxar API the codes may in
      some cases return an empty dictionary. This is now fixed, #465.
  * Fix use of realisations key from Roxar API (3D grids), #443.
  * General improvements in documentation, e.g. clearer guidelines for contributions.

### 2.14.1
* Technical build and docs (logo on RTD) issues, not affecting end users.


## Version 2.13

From this version support for Python 2.7 and 3.5 is dropped. Now only Python
3.6+ is supported

* New features:
  * Grid quality indicators for Grid(): ``get_gridquality_properties()``.
  * Support for BytesIO i/o for ``RegularSurface()`` formats irap ascii and zmap ascii.
  * A `fmt` key is now available in `GridProperty().to_file()` for Eclipse grdecl
    format, and default format is now scientific %e for Floats.
  * Added a key `datesonly` for GridProperties().``scan_dates`` for getting a
    simple list of dates existing in a given restart file.
  * Added a ``strict`` key to ``GridProperties().from_file()`` method.
  * Added ``Points().from_dataframe()`` method.
  * Added bulk volumetrics ``Grid()``: ``get_cell_volume()`` and ``get_bulk_volume()``
    as experimental methods.
  * Added ``percentiles`` option for ``Surfaces().statistics()``.

* Improvements and bug fixes
  * Reorganizing internal storage of 3D corner point grids.
  * Faster read and write to ROFF binary for Grid() (read almost 4 times faster).
  * Faster read/write to Roxar API for grid geometries.
  * Import from Eclipse is improved, in particular using "all" alias for
    ``initprops``, ``restartprops`` and ``restartdates``.
  * Fix issues  #414, #415, #418, #421, #423, #426, #436, #439.

### 2.13.1
* Fixed a manifest bug which occurs when importing a grid inside RMS, cf #448.

### 2.13.2
* Fix CI testing for Komodo (not affecting end user)

### 2.13.3
* Adjust SWIG install in github actions (not affecting end user)

### 2.13.4
* Adjust CI install in github actions (not affecting end user)

## Version 2.12

* New features:
  * Added method ``rename_subgrids`` for Grid() class.
  * Added key ``casting`` to method ``to_roxar()`` for GridProperty() class
  * Added key ``faciescodes`` to method ``from_roxar`` and
    ``gridproperty_from_roxar()`` for GridProperty() class
  * It is now possible to write blocked wells and ordinary wells to Roxar API
  * Added a ``autocrop()`` function for RegularSurface()

### 2.12.3
* Postfix release backport, adress #439

### 2.12.2
* Postfix release backport, adress #436

### 2.12.1
* Fix of ``get_dataframe()`` for ``RegularSurface`` which fails in some case, cf issue #415


## Version 2.11
* New features:
  * Added keys ``perflogrange`` and ``filterlogrange`` in Grid()
    ``report_zone_mismatch()``



## Version 2.10
* New features:
  * Added interpolation option in xsection when plotting 3D grids #401
* Fixes:
  * Improvements in Roxar API interface, e.g. behaviour on when projects are saved or not
  * Fix on surface values, which data that can be accepted, issue #405
  * Some other minor fixes in code and documentation

## Version 2.9
* Full rewrite of surface slice cube methods, they will now be much faster #354
* Added `activeonly` key in `make_ijk_from_grid` for Well()
* Improving points in cell detection algorithm (full rewrite)
* Fix bug in cube orientation when importing fro Roxar API #359
* Introducing new faster reading of roff grids (will not be default until later)
* Fix of xinc/yinc for PMD map format #367
* Improvements in various plot routines, in particular xsections
* Changed CI and deploy from travis/appveyor to Github actions using `cibuildwheel`

### 2.9.2
* Postfix release backport, adress #436 and #439

### 2.9.0 and 2.9.1
* Initial release, 2.9.1 replaced 2.9.0. for technical reasons

## Version 2.8
* New features:
  * Added gridlink option in GridProperty import, #329
  * More keyword options in Grid get_ijk_from_points() #327
  * Well method report_zone_mismatch() rewritten and improved
  * Well: added get_surface_picks()
  * Initialise a new GridProperty instance from existing GridProperty
  * Grid(): added name as attribute #319
* Bug fixes:
  * The gridproperty list in GridProperty() is now unique
  * Fixed bug in Well: get_zonation_points
  * More fixes on pathlib (general rewrite) #332
* Fixes for developers:
  * Replace logging methods in the C part, and relocated clib folder
  * Added code coverage in travis CI

### 2.8.3
* Fix a bug for renaming points and polygons coordinate columns, ref #349
* Added and "F" in SPECGRID when exporting 3D grid to GRDECL format

### 2.8.2
* Fix a bug wrt writing bytestream instances on non-Linux, #342

### 2.8.1
* Fix a clib related issue that made XTGeo import feil on RHEL6 in Python2, #346

## Version 2.7
* New features:
  * Support for petromod binary format for RegularSurface()
  * Added name attribute for Grid()
  * Enhanced plotting for well logs
  * The arrays stored Grid() are no longer SWIG C pointers, but numpy arrays. This simplifies pickling.
* Bug fixes:
  * File names used in e.g. from_file should now handle pathlib instances
  * Improved error messages if issues with e.g. file names used in export/import
  * Fix of excessive logger output in
* Fixes for developers:
  * General refactorizion of C code, to improve speed and stability. Also change logger method in C (still ongoing)

### 2.7.1
* Bugfig:
  * Issue with pathlib solved, #332


## Version 2.6
* New features:
  * A Grid() instance can now be "numpified" so that pickling can be done, method `numpify_carrays()`
  * An existing GridProperty() instance should now accept scalar input which will be broadcasted to
    the full array
  * Added a method so one can create a GridProperty instance directly for a Grid() instance #291
  * Added several alternatives to instantate Points(), e.g. from a list of tuples
  * A general method that finds the IJK indices in a 3D grid from from Points() is made `get_ijk_from_points` #287
  * For RegularSurface(), the `fill()` methid will now accept an optional fill_value (constant) #294
* Bug fixes:
  * Making surface write to BytesIO stream threading safe (Irap binary format)
  * Assigning a GridProperty() inside/outside a polygon is now more robust.
  * Many internal build fixes and improves, including requirements.txt
  * For surfaces, some operator overload function changed unintentionally the `other` instance #295
  * For surfaces, operator overload on instances with same topology will not unintentionally trigger resampling


## Version 2.5
* New features:
  * Be able to write surfaces to BytesIO (memory streams), Linux only
  * Add the ability for 3D grids to detect and swap handedness of a 3D grid.
  * Available on Python 3.8 on all platforms
* Fixes for developers
  * Now backward compatible to cmake 2.8.12
  * Many internal build fixes and improves, including requirements.txt

## Version 2.4

* New features:
  * Added a general kwargs to `savefig()` in plot module, so e.g. dpi keyword can be passed to matplotlib
* Bug fixes:
  * More robust on reading saturations from UNRST files from Eclipse 300 and IX, where "IPHS" metadata
    (describing phases present) is unreliable.
* Fixes for developers:
  * Setup can now be ran in "develop mode"

### 2.4.3
* Fix of bugs when exporting points/polygons to Roxar API
* Fix (for developers) various setup in cmake/swig etc so that cmake can be downgraded to 3.13.3 and hence a ``manylinux1`` image is available in PYPI for Linux (Python versions < 3.7)

### 2.4.2
* Fix a bug that occurs when reading Eclipse properties from E300 runs

### 2.4.1
* Push to trigger travis build and deploy


## Version 2.3

* Added support for MacOS on PYPI (Python 3.6, 3.7)
* Added functionality on grid slices as method ()
* More flexible reading on phases present in Eclipse/IX UNRST files
* Several minor bugfixes and improvements

### 2.3.1

* Preliminary support for Python 3.8 (Linux only)
* Several bug fixes:
  * User warning when requested colour map is not found
  * Printing of a Points of Polygons instance shall now work
  * UNDEF values in property grdecl or bgrdecl export shall now be 0.0, not a large number
  * Name in GridProperty `to_file(name=...)` is fixed
  * If `fformat` in GridProperty import is mispelled, an exception will be raised


## Version 2.2

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


### 2.2.2

* Several smaller bug fixes
* Use of realisation in gridproperty_from_roxar() was not working

### 2.2.1

* Full C code and compile restructuring, now using scikit-build!
* Use of realisation in gridproperty_from_roxar() was not working


## Version 2.1

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


## Version 2.0

* First version after Open Sourcing to LGPL v3+

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


## Version 0 and 1

See github for commit and tag history:

https://github.com/equinor/xtgeo
