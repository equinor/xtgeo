=================================
XTGeo data models and i/o formats
=================================

Here is a description of the internal datamodels used for various datatypes in
XTGeo, and the input/output formats that are supported in the current version.

-----------------------
Surface: RegularSurface
-----------------------

See class :class:`xtgeo.surface.RegularSurface` for details on available
methods and attributes.

A surface can in principle be represented in various ways. Currently, XTGeo
supports a `RegularSurface` which is commonly used in the oil
industry. Due to the regular layout, such surfaces are quite fast to work
with and requires small storage (only the Z values array is stored).

Description
^^^^^^^^^^^

A RegularSurface is described by:

* An origin in UTM coordinates, defined as ``xori`` and ``yori``

* An increment in each direction, defined as ``xinc`` and ``yinc``

* A number of columns and rows, where columns follow X and rows follow Y, if
  a rotation is zero, as ``ncol`` and ``nrow``.

* A ``rotation`` of the X axis; in XTGeo the rotation is counter-clockwise
  from the X (East) axis, in degrees.

* An ``yflip`` indicator. Normally the system is left-handed (with Z axis
  positive down). If yflip is -1, then the map is right-handed.

* A 2D array (masked numpy) of ``values``, for a total of ncol * nrow entries.
  Undefined map nodes are masked. The 2D numpy array is stored in C-order
  (row-major). Default is 64 bit Float.

.. figure:: images/datamodel_regsurface.svg

Within the C code (backend for python functions), arrays are stored in 1D,
C-order and are usually named ``p_map_v`` in the code.


RegularSurface supported import/export formats
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table:: RegularSurface format support
   :widths: 20 8 8 20 30
   :header-rows: 1

   * - Data format/source
     - Import
     - Export
     - Format limitations
     - Comment
   * - Irap/RMS Binary
     - Yes
     - Yes
     -
     - This is the default format
   * - Irap/RMS ASCII
     - Yes
     - Yes
     -
     -
   * - IJXYZ OW)
     - Yes
     - Yes
     -
     - 5 columns: I, J, X, Y, Z
   * - ZMAP+ ASCII
     - No
     - Yes
     - No map rotation
     - Output auto derotated
   * - Petromod binary pmd
     - Yes
     - Yes
     -
     -
   * - Storm binary
     - No
     - Yes
     - No map rotation
     - Output auto derotated
   * - Inside RMS (ROXAPI)
     - Yes
     - Yes
     -
     -

---------
Cube data
---------

A Cube is described by:

* An origin in UTM coordinates, defined as ``xori``, ``yori`` and ``zori``

* An increment in each direction, defined as ``xinc``, ``yinc`` and ``zinc``

* A number of columns, rows and layers, where columns follow X and rows follow Y, if
  a rotation is zero, as ``ncol`` and ``nrow``. Vertically ``nlay``

* A ``rotation`` of the X axis; in XTGeo the rotation is counter-clockwise
  from the X (East) axis, in degrees.

* An ``yflip`` indicator. Normally the system is left-handed (with Z axis
  positive down). If yflip is -1, then the cube is right-handed.

* A 3D array (numpy) of ``values``, for a total of ncol * nrow * nlay entries.
  All nodes are defined. The 3D numpy array is stored in C-order
  (row-major). Default is 32 bit Float.

Within the C code (backend for python functions), arrays are stored in 1D,
C-order and are usually named ``p_cube_v`` in the code.


Description of cubes
^^^^^^^^^^^^^^^^^^^^

Cubes are quite similar to maps, only that a third dimension is added.

Cube supported import/export formats
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table:: Cube format support
   :widths: 20 8 8 20 30
   :header-rows: 1

   * - Data format/source
     - Import
     - Export
     - Format limitations
     - Comment
   * - SEGY
     - Yes
     - Yes
     -
     - Export: ASCII not EBDIC
   * - RMS regular grid
     - Yes
     - Yes
     -
     -
   * - Storm regular grid
     - Yes
     - No
     -
     -
   * - Inside RMS, ROXAPI
     - Yes
     - Yes
     -
     -

----------------------
3D Grid and properties
----------------------

A 3D grid consists of two parts, a geometry which follows a simplified version of
a corner-point grid (Grid class), and a number of assosiated properties
(GridProperty class):

* The geometry is stored as numpy arrays; but currently the geometry itself is not
  directly accessible in Python. The corresponding C arrays are usually named
  coordsv, zcornsv, and actnumsv. They are one dimensionial arrays in internal
  _xtgformat=1, or multidimensional arrays in xtgformat=2 (details on demand).

* The grid dimensions are given by ``ncol * nrow * nlay``

* The properties are stored as 3D masked numpy arrays in python. Undefined cells are
  masked. The machine order in python is C-order. For historical reasons, the memory
  order of the property arrays in C code (when applied) is F-order for _xtgformat=1,
  while _xtgformat=2 uses C order.


Description of 3D grid formats
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In prep.

Supported import/export formats
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table:: 3D grid geometry format support
   :widths: 20 8 8 20 30
   :header-rows: 1

   * - Data format/source
     - Import
     - Export
     - Format limitations
     - Comment
   * - ROFF binary
     - Yes
     - Yes
     -
     - Default
   * - ROFF ASCII
     - No
     - Yes
     -
     -
   * - XTGeo binary
     - Yes
     - Yes
     -
     - Fast binary format that allows metadata
   * - Eclipse ASCII GRDECL
     - Yes
     - Yes
     -
     -
   * - Eclipse binary GRDECL
     - Yes
     - Yes
     -
     -
   * - Eclipse EGRID
     - Yes
     - Yes
     -
     -
   * - Eclipse GRID
     - No
     - No
     -
     - Rarely applied nowadays?
   * - Pandas dataframes
     - No
     - Yes
     -
     - Indirect CSV format
   * - Inside RMS, ROXAPI
     - Yes
     - Yes §
     -
     - § Improved in RMS 11.1

The Pandas dataframe format is limited in the sense that only centerpoint
coordinates are applied.

.. list-table:: 3D grid property format support
   :widths: 20 8 8 20 30
   :header-rows: 1

   * - Data format/source
     - Import
     - Export
     - Format limitations
     - Comment
   * - ROFF binary
     - Yes
     - Yes
     -
     - Default
   * - ROFF ASCII
     - No
     - Yes
     -
     -
   * - Ecl ASCII GRDECL
     - Yes
     - Yes
     - Discrete coding missing
     -
   * - Ecl binary GRDECL
     - Yes
     - Yes
     - Discrete coding missing
     -
   * - Ecl bin INIT, UNRST
     - Yes
     - No
     - Discrete coding missing
     -
   * - Pandas dataframes
     - No
     - Yes
     -
     - Indirect CSV format
   * - Inside RMS, ROXAPI
     - Yes
     - Yes
     -
     -

---------
Well data
---------

Well data is stored in python as Pandas dataframe plus some additional
metadata.

A special subclass is Blocked Well data.

Well data supported import/export formats
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In prep.

-----------------------------
XYZ data, Points and Polygons
-----------------------------

In general, Points and Polygons are XYZ data with possible atttributes.

Points and Polygons data is stored in python as Pandas dataframe plus some additional
metadata.

The term "Polygons" here is not precise perhaps, at it refers to connected lines which
can either form an open polyline or are closed polygon. A Polygons() instance may
have a number of individual polygon "pieces", which are defined by
a ``POLY_ID`` (default name) column. This design is borrowed from RMS.

XYZ data supported import/export formats
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In prep.


--------------------
XTGeo binary formats
--------------------

(in prep)

XTGeo is developing it own binary data storage due to limitations in existing systems.
Examples of such limitions are:

* Cube data: SEGY format is a complex, old and difficult format, for the purpose
  of Cube data in XTGeo

* 3D grid data: There is no existing format that handles named subgrids

* General lack of support for metadata, where metadata shall both have mandatory
  and optional members

* Efficient data structures for read and write

XTGeo general binary layout
^^^^^^^^^^^^^^^^^^^^^^^^^^^

In prep.

XTGeo 3D grid format
^^^^^^^^^^^^^^^^^^^^

.. rst-class:: dl-parameters

Record 1:
  An 4 byte integer. This integer is an "endian" indicator and will read as 1
  if no bytewap is required.

Record 2:
  A 16 (15 char + null) byte magic word, which serves as a type and version
  information. For 3D grids, this word is ">XTGGRIDCP3D_V1"

Records 3 to 5:
  Three 8 byte (C type long) integer numbers for `ncol`, `nrow`, `nlay`

Record 6:
  Coordinates with shape (ncol + 1) * (nrow + 1) * 6, C order.

Record 7:
  Zcorners, with shape (ncol + 1) * (nrow + 1) * (nlay + 1) * 4.

Record 8:
  A 16 byte magic word. THis is used to verify that data structures are
  read correctly. For 3D grids, this word is "<XTGGRIDCP3D_V1"

Record 9:
  A JSON text structure of free length (until 100,000 bytes) that represents
  metadata


XTGeo layout for other data types
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In prep.
