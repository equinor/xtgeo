.. highlight:: python
.. automodule:: xtgeo
   :members:

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
with and requires minimum storage.

Description
^^^^^^^^^^^

In principle, a RegularSurface is described by:

* An origin in UTM coordinates, defined as ``xori`` and ``yori``

* An increment in each direction, defined as ``xinc`` and ``yinc``

* A number of columns and rows, where columns follow X and rows follow Y, if
  a rotation is zero, as ``ncol`` and ``nrow``.

* A ``rotation`` of the X axis; in XTGeo the rotation is counter-clockwise
  from the X (East) axis, in degrees.

* An ``yflip`` indicator. Normally the system is left-handed (with Z axis
  positive down). If yflip is -1, then the map is right-handed.

* A 2D array (masked numpy) of ``values``, for a total of ncol * nrow entries.
  Undefined map nodes are masked. The 2D array is stored in C-order
  (row-major). Default is 64 bit Float.

.. figure:: images/datamodel_regsurface.svg

Within the C code (backend for python functions), arrays are stored in 1D,
C-order and are usually named ``p_map_v`` in the code.

Supported import/export formats
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

In prep.

Description
^^^^^^^^^^^

In prep.

Supported import/export formats
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Supported import/export formats
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

In prep.

Description
^^^^^^^^^^^

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
     - No?
     - Yes
     -
     -
   * - Eclipse ASCII GRDECL
     - Yes
     - Yes
     -
     -
   * - Eclipse binary GRDECL
     - Yes?
     - Yes?
     -
     -
   * - Eclipse EGRID
     - Yes
     - No
     -
     -
   * - Eclipse GRID
     - Yes?
     - No
     -
     - Rarely applied?
   * - Inside RMS, ROXAPI
     - Yes
     - No
     -
     -

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
     - No?
     - Yes?
     -
     -
   * - Ecl ASCII GRDECL
     - Yes
     - Yes
     - Discrete coding missing
     -
   * - Ecl binary GRDECL
     - Yes?
     - Yes?
     - Discrete coding missing
     -
   * - Ecl bin INIT, UNRST
     - Yes
     - No
     - Discrete coding missing
     -
   * - Inside RMS, ROXAPI
     - Yes
     - Yes
     -
     -
