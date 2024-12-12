"""Module/class for regular surfaces with XTGeo.

Regular surfaces have a constant distance between nodes (xinc, yinc),
and this simplifies computations a lot. A regular surface is
defined by an origin (xori, yori)
in UTM, a number of columns (along X axis, if no rotation), a number of
rows (along Y axis if no rotation), and increment (distance between nodes).

The map itself is an array of values.

Rotation is allowed and is measured in degrees, anticlock from X axis.

Note that an instance of a regular surface can be made directly with::

 >>> import xtgeo
 >>> mysurf = xtgeo.surface_from_file(surface_dir + '/topreek_rota.gri')

or::

 mysurf = xtgeo.surface_from_roxar('some_rms_project', 'TopX', 'DepthSurface')

"""

# --------------------------------------------------------------------------------------
# Comment on 'asmasked' vs 'activeonly:
# 'asmasked'=True will return a np.ma array, with some fill_value if
# if asmasked = False
#
# while 'activeonly' will filter
# out maked entries, or use np.nan if 'activeonly' is False
#
# For functions with mask=... ,the should be replaced with asmasked=...
# --------------------------------------------------------------------------------------

from __future__ import annotations

import functools
import math
import numbers
import warnings
from copy import deepcopy
from types import FunctionType
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Tuple, Type, Union

import numpy as np
import numpy.ma as ma
import pandas as pd
import scipy.ndimage

from xtgeo.common.constants import (
    UNDEF,
    UNDEF_LIMIT,
    VERYLARGENEGATIVE,
    VERYLARGEPOSITIVE,
)
from xtgeo.common.exceptions import InvalidFileFormatError
from xtgeo.common.log import null_logger
from xtgeo.common.sys import generic_hash
from xtgeo.common.xtgeo_dialog import XTGDescription, XTGeoDialog
from xtgeo.io._file import FileFormat, FileWrapper
from xtgeo.metadata.metadata import MetaDataRegularSurface
from xtgeo.xyz.points import Points

from . import (
    _regsurf_boundary,
    _regsurf_cube,
    _regsurf_cube_window,
    _regsurf_export,
    _regsurf_grid3d,
    _regsurf_gridding,
    _regsurf_import,
    _regsurf_oper,
    _regsurf_roxapi,
    _regsurf_utils,
)

if TYPE_CHECKING:
    import io
    import pathlib

    from xtgeo.cube.cube1 import Cube
    from xtgeo.grid3d.grid import Grid, GridProperty


xtg = XTGeoDialog()
logger = null_logger(__name__)

# valid argumentts for seismic attributes
ValidAttrs = Literal[
    "all",
    "max",
    "min",
    "rms",
    "mean",
    "var",
    "maxpos",
    "maxneg",
    "sumpos",
    "sumneg",
    "meanabs",
    "meanpos",
    "meanneg",
]

# ======================================================================================
# METHODS as wrappers to class init + import


def surface_from_file(
    mfile,
    fformat=None,
    template=None,
    values=True,
    engine: Optional[str] = "cxtgeo",
    dtype: Union[Type[np.float64], Type[np.float32]] = np.float64,
):
    """Make an instance of a RegularSurface directly from file import.

    Args:
        mfile (str): Name of file
        fformat: File format, None/guess/irap_binary/irap_ascii/ijxyz/petromod/
            zmap_ascii/xtg/hdf is currently supported. If None or guess, the file
            'signature' is used to guess format first, then file extension.
        template: Only valid if ``ijxyz`` format, where an existing Cube or
            RegularSurface instance is applied to get correct topology.
        values (bool): If True (default), surface values will be read (Irap binary only)
        engine (str): Some import methods are implemnted in both C and Python.
            The C method ``cxtgeo`` is default. Alternative use ``python``
        dtype: Requsted numpy dtype of values; default is float64, alternatively float32

    Example::

        >>> import xtgeo
        >>> mysurf = xtgeo.surface_from_file(surface_dir + '/topreek_rota.gri')

    .. versionchanged:: 2.1
      Key "values" for Irap binary maps added

    .. versionchanged:: 2.13 Key "engine" added
    """

    return RegularSurface._read_file(
        mfile,
        fformat=fformat,
        load_values=values,
        engine=engine,
        template=template,
        dtype=dtype,
    )


def surface_from_roxar(
    project,
    name,
    category,
    stype="horizons",
    realisation=0,
    dtype: Union[Type[np.float64], Type[np.float32]] = np.float64,
):
    """This makes an instance of a RegularSurface directly from roxar input.

    Args:
        project (str or special): Name of project (as folder) if
            outside RMS, og just use the magic project word if within RMS.
        name (str): Name of surface/map
        category (str): For horizons/zones or clipboard/general2d_data:
            for example 'DS_extracted'. For clipboard/general2d_data this can
            be empty or None, or use '/' for multiple folder levels (e.g. 'fld/subfld').
            For 'trends', the category is not applied.
        stype (str): RMS folder type, 'horizons' (default), 'zones', 'clipboard',
            'general2d_data' or 'trends'
        realisation (int): Realisation number, default is 0
        dtype: Requested numpy dtype for array; default is 64 bit

    Example::

        # inside RMS:
        import xtgeo
        mysurf = xtgeo.surface_from_roxar(project, 'TopEtive', 'DepthSurface')

    Note::

        When dealing with surfaces to and from ``stype="trends"``, the surface must
        exist in advance, i.e. the Roxar API do not allow creating new surfaces.
        Actually trends are read only, but a workaround using ``load()`` in Roxar
        API makes it possible to overwrite existing surface trends. In addition,
        ``realisation`` is not applied in trends.

    """

    return RegularSurface._read_roxar(
        project, name, category, stype=stype, realisation=realisation, dtype=dtype
    )


def surface_from_cube(cube, value):
    """Make RegularSurface directly from a cube instance with a constant value.

    The surface geometry will be exactly the same as for the Cube.

    Args:
        cube(xtgeo.cube.Cube): A Cube instance
        value (float): A constant value for the surface

    Example::

       >>> import xtgeo
       >>> mycube = xtgeo.cube_from_file(cube_dir + "/ib_test_cube2.segy")
       >>> mymap = xtgeo.surface_from_cube(mycube, 2700)

    """
    return RegularSurface._read_cube(cube, value)


def surface_from_grid3d(
    grid: Grid,
    template: RegularSurface | str | None = None,
    where: str | int = "top",
    property: str | GridProperty = "depth",
    rfactor: float = 1.0,
    index_position: Literal["center", "top", "base"] = "center",
    **kwargs,
) -> RegularSurface | List[np.ndarray]:
    """This makes an instance of a RegularSurface directly from a Grid() instance.

    Args:
        grid: XTGeo 3D Grid instance, describing the corner point grid geometry
        template: Optional, to use an existing surface as
            template for map geometry, or by using "native" which returns a surface that
            approximate the 3D grid layout (same number of rows and columns, and same
            rotation). If None (default) a non-rotated surface will be made
            based on a refined estimation from the current grid
            resolution (see ``rfactor``).
        where: Cell layer number, or if property is "depth", use "top" or "base" to get
            a surface sampled from the very top or very base of the grid (including
            inactive cells!). Otherwise use the syntax "2_top" where 2
            is layer no. 2 and _top indicates top of cell, while "_base"
            indicates base of cell. Cell layer numbering starts from 1. Default position
            in a cell layer is "top" if layer is given as pure number and "depth" is
            the property. If a grid property is given, the position is always found
            the center depth within in a cell.
        property: Which property to return. Choices are "depth", "i"
            (columns) or "j" (rows) or, more generic, a GridProperty instance
            which belongs to the given grid geometry. Alle these returns a
            RegularSurface. A special variant is "raw" which
            returns a list of 2D numpy arrays. See details in the Note section.
        rfactor: Note this setting will only apply if ``template`` is None.
            Determines how fine the extracted map is; higher values
            for finer map (but computing time will increase slightly).
            The default is 1.0, which in effect will make a surface approximentaly
            twice as fine as the average resolution estimated from the 3D grid.
        index_position: Default is "center" which means that the index is taken
            from the Z center of the cell. If "top" is given, the index is taken from
            the top of the cell, and if "base" is given, the index is taken from the
            base of the cell. This is only valid for index properties "i" and "j".

    Note::
        The keyword ``mode`` is deprecated and will be removed in XTGeo version 5,
        use keyword ``property`` instead. If both are given, ``property`` will be used.

    Note::
        For ``property`` "depth", "i" and "j", all cells in a layer will be used
        (including inactive 3D cells), while for a GridProperty, only active cells
        will be used. Hence the extent of the resulting surfaces may differ.

    Note::
        For ``property`` "raw", the return is a list of 2D arrays, where the first
        array is the i-index (int), the second is the j-index (int), the third is the
        top depth (float64), the fourth is the bottom depth (float64), and the
        fifth is a mask (boolean) for inactive cells. For the index arrays, -1
        indicates that the cell is outside any grid cell (projected from above; i.e.
        could also be within a fault). For the depth arrays, the value is NaN
        for inactive cells. The inactive mask is True for inactive cells. The index
        arrays and mask is derived from the Z midpoints of the 3D cells. The "raw"
        option is useful for further processing in Python, e.g. when a combination
        of properties is needed.

    .. versionadded:: 2.1
    .. versionchanged:: 4.2 Changed ``mode`` to ``property`` to add support for
                            a GridProperty. The ``where`` arg. can now be an integer.
                            Added option ``activeonly``.
    .. versionchanged:: 4.3 Added option ``raw`` to get data for further processing.
                            and add ``index_position`` for "i" and "j" properties.
    """
    mode = kwargs.get("mode")
    if mode is not None:
        warnings.warn(
            "The 'mode' argument is deprecated, use 'property' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        property = mode

    if isinstance(property, str) and property == "raw":
        args = _regsurf_grid3d.from_grid3d(
            grid, template, where, property, rfactor, index_position="center"
        )
        return args["values"]

    return RegularSurface._read_grid3d(
        grid, template, where, property, rfactor, index_position
    )


def _data_reader_factory(file_format: FileFormat):
    if file_format == FileFormat.IRAP_BINARY:
        return _regsurf_import.import_irap_binary
    if file_format == FileFormat.IRAP_ASCII:
        return _regsurf_import.import_irap_ascii
    if file_format == FileFormat.IJXYZ:
        return _regsurf_import.import_ijxyz
    if file_format == FileFormat.PETROMOD:
        return _regsurf_import.import_petromod
    if file_format == FileFormat.ZMAP_ASCII:
        return _regsurf_import.import_zmap_ascii
    if file_format == FileFormat.XTG:
        return _regsurf_import.import_xtg
    if file_format == FileFormat.HDF:
        return _regsurf_import.import_hdf5_regsurf

    extensions = FileFormat.extensions_string(
        [
            FileFormat.IRAP_BINARY,
            FileFormat.IRAP_ASCII,
            FileFormat.IJXYZ,
            FileFormat.PETROMOD,
            FileFormat.ZMAP_ASCII,
            FileFormat.XTG,
            FileFormat.HDF,
        ]
    )
    raise InvalidFileFormatError(
        f"File format {file_format} is invalid for type RegularSurface. "
        f"Supported formats are {extensions}."
    )


class RegularSurface:
    """Class for a regular surface in the XTGeo framework.

    The values can as default be accessed by the user as a 2D masked numpy
    (ncol, nrow) float64 array, but also other representations or views are
    possible (e.g. as 1D ordinary numpy).

    """

    def __init__(
        self,
        ncol: int,
        nrow: int,
        xinc: float,
        yinc: float,
        xori: Optional[float] = 0.0,
        yori: Optional[float] = 0.0,
        yflip: Optional[int] = 1,
        rotation: Optional[float] = 0.0,
        values: Optional[Union[List[float], float]] = None,
        ilines: Optional[List[float]] = None,
        xlines: Optional[List[float]] = None,
        masked: Optional[bool] = True,
        name: Optional[str] = "unknown",
        filesrc: Optional[str] = None,
        fformat: Optional[str] = None,
        undef: Optional[float] = UNDEF,
        dtype: Union[Type[np.float64], Type[np.float32]] = np.float64,
    ):
        """Instantiating a RegularSurface::

            vals = np.zeros(30 * 50)
            surf = xtgeo.RegularSurface(
                ncol=30, nrow=50, xori=1234.5, yori=4321.0, xinc=30.0, yinc=50.0,
                rotation=30.0, values=vals, yflip=1,
            )

        Args:
            ncol: Integer for number of X direction columns.
            nrow: Integer for number of Y direction rows.
            xori: X (East) origon coordinate.
            yori: Y (North) origin coordinate.
            xinc: X increment.
            yinc: Y increment.
            yflip: If 1, the map grid is left-handed (assuming depth downwards),
                otherwise -1 means that Y axis is flipped (right-handed).
            rotation: rotation in degrees, anticlock from X axis between 0, 360.
            values: A scalar (for constant values) or a "array-like" input that has
                ncol * nrow elements. As result, a 2D (masked) numpy array of shape
                (ncol, nrow), C order will be made.
            masked: Indicating if numpy array shall be masked or not. Default is True.
            name: A given name for the surface, default is file name root or
                'unknown' if constructed from scratch.

        Examples:
            The instance can be made by specification::

                >>> surface = RegularSurface(
                ... ncol=20,
                ... nrow=10,
                ... xori=2000.0,
                ... yori=2000.0,
                ... rotation=0.0,
                ... xinc=25.0,
                ... yinc=25.0,
                ... values=np.zeros((20,10))
                ... )


        """
        logger.info("Start __init__ method for RegularSurface object %s", id(self))
        self._ncol = ncol
        self._nrow = nrow
        self._xori = xori
        self._yori = yori
        self._xinc = xinc
        self._yinc = yinc
        self._rotation = rotation
        self._yflip = yflip
        self._name = name

        self._undef = undef
        self._undef_limit = UNDEF_LIMIT

        self._filesrc = filesrc  # Name of original input file or stream, if any

        self._fformat = fformat  # current fileformat, useful for load()
        self._metadata = MetaDataRegularSurface()

        self._values = None

        if values is None:
            values = np.ma.zeros((self._ncol, self._nrow))
            self._isloaded = False
        else:
            self._isloaded = True
        self._ensure_correct_values(values, force_dtype=dtype)

        if ilines is None:
            self._ilines = np.array(range(1, self._ncol + 1), dtype=np.int32)
            self._xlines = np.array(range(1, self._nrow + 1), dtype=np.int32)
        else:
            self._ilines = ilines
            self._xlines = xlines

        self._masked = masked  # TODO: check usecase
        self._metadata.required = self

    @classmethod
    def _read_zmap_ascii(cls, mfile, values):
        mfile = FileWrapper(mfile)
        args = _data_reader_factory(FileFormat.ZMAP_ASCII)(mfile, values=values)
        return cls(**args)

    @classmethod
    def _read_ijxyz(cls, mfile: FileWrapper, template: RegularSurface | Cube | None):
        mfile = FileWrapper(mfile)
        args = _data_reader_factory(FileFormat.IJXYZ)(mfile, template=template)
        return cls(**args)

    def __repr__(self):
        """Magic method __repr__."""
        return (
            f"{self.__class__.__name__} (xori={self._xori!r}, yori={self._yori!r}, "
            f"xinc={self._xinc!r}, yinc={self._yinc!r}, ncol={self._ncol!r}, "
            f"nrow={self._nrow!r}, rotation={self._rotation!r}, yflip={self._yflip!r}, "
            f"masked={self._masked!r}, filesrc={self._filesrc!r}, "
            f"name={self._name!r}, ilines={self.ilines.shape!r}, "
            f"xlines={self.xlines.shape!r}, values={self.values.shape!r}) "
            f"ID={id(self)}."
        )

    def __str__(self):
        """Magic method __str__ for user friendly print."""
        return self.describe(flush=False)

    def __getitem__(self, index):
        """Magic method."""
        col, row = index
        return self._values[col, row]

    def __add__(self, other):
        """Magic method."""
        news = self.copy()
        _regsurf_oper.operations_two(news, other, oper="add")

        return news

    def __iadd__(self, other):
        """Magic method."""
        _regsurf_oper.operations_two(self, other, oper="iadd")
        return self

    def __sub__(self, other):
        """Magic method."""
        news = self.copy()
        _regsurf_oper.operations_two(news, other, oper="sub")

        return news

    def __isub__(self, other):
        """Magic method."""
        _regsurf_oper.operations_two(self, other, oper="isub")
        return self

    def __mul__(self, other):
        """Magic method."""
        news = self.copy()
        _regsurf_oper.operations_two(news, other, oper="mul")

        return news

    def __imul__(self, other):
        """Magic method."""
        _regsurf_oper.operations_two(self, other, oper="imul")
        return self

    def __truediv__(self, other):
        """Magic method."""
        news = self.copy()
        _regsurf_oper.operations_two(news, other, oper="div")

        return news

    def __idiv__(self, other):
        """Magic method."""
        _regsurf_oper.operations_two(self, other, oper="idiv")
        return self

    # comparison operators, return boolean arrays

    def __lt__(self, other):
        """Magic method."""
        return _regsurf_oper.operations_two(self, other, oper="lt")

    def __gt__(self, other):
        """Magic method."""
        return _regsurf_oper.operations_two(self, other, oper="gt")

    def __le__(self, other):
        """Magic method."""
        return _regsurf_oper.operations_two(self, other, oper="le")

    def __ge__(self, other):
        """Magic method."""
        return _regsurf_oper.operations_two(self, other, oper="ge")

    def __eq__(self, other):
        """Magic method."""
        return _regsurf_oper.operations_two(self, other, oper="eq")

    def __ne__(self, other):
        """Magic method."""
        return _regsurf_oper.operations_two(self, other, oper="ne")

    # ==================================================================================
    # Class and special methods
    # ==================================================================================

    @classmethod
    def methods(cls):
        """Returns the names of the methods in the class.

        >>> print(RegularSurface.methods())
        METHODS for RegularSurface():
        ======================
        __init__
        __repr__
        ...

        """
        mets = [x for x, y in cls.__dict__.items() if isinstance(y, FunctionType)]

        txt = "METHODS for RegularSurface():\n======================\n"
        for met in mets:
            txt += str(met) + "\n"

        return txt

    # ==================================================================================
    # Properties
    # ==================================================================================

    @property
    def metadata(self):
        """Return metadata object instance of type MetaDataRegularSurface."""
        return self._metadata

    @metadata.setter
    def metadata(self, obj):
        # The current metadata object can be replaced. A bit dangerous so further
        # check must be done to validate. TODO.
        if not isinstance(obj, MetaDataRegularSurface):
            raise ValueError("Input obj not an instance of MetaDataRegularSurface")

        self._metadata = obj  # checking is currently missing! TODO

    @property
    def ncol(self):
        """The NCOL (NX or N-Idir) number, as property (read only)."""
        return self._ncol

    @property
    def nrow(self):
        """The NROW (NY or N-Jdir) number, as property (read only)."""
        return self._nrow

    @property
    def dimensions(self):
        """2-tuple: The surface dimensions as a tuple of 2 integers (read only)."""
        return (self._ncol, self._nrow)

    @property
    def nactive(self):
        """Number of active map nodes (read only)."""
        if self._isloaded:
            return self._values.count()
        return None

    @property
    def rotation(self):
        """The rotation, anticlock from X axis, in degrees [0..360>."""
        return self._rotation

    @rotation.setter
    def rotation(self, rota):
        if 0 <= rota < 360:
            self._rotation = rota
        else:
            raise ValueError("Rotation must be in interval [0, 360>")

    @property
    def xinc(self):
        """The X increment (or I dir increment)."""
        return self._xinc

    @property
    def yinc(self):
        """The Y increment (or I dir increment)."""
        return self._yinc

    @property
    def yflip(self):
        """The Y flip (handedness) indicator 1, or -1 (read only).

        The value 1 (default) means a left-handed system if depth values are
        positive downwards. Assume -1 is rare, but may happen when
        surface is derived from seismic cube.
        """
        return self._yflip

    @property
    def xori(self):
        """The X coordinate origin of the map."""
        return self._xori

    @xori.setter
    def xori(self, xnew):
        self._xori = xnew

    @property
    def yori(self):
        """The Y coordinate origin of the map."""
        return self._yori

    @yori.setter
    def yori(self, ynew):
        self._yori = ynew

    @property
    def ilines(self):
        """The inlines numbering vector (read only)."""
        return self._ilines

    @ilines.setter
    def ilines(self, values):
        if isinstance(values, np.ndarray) and values.shape[0] == self._ncol:
            self._ilines = values

    @property
    def xlines(self):
        """The xlines numbering vector (read only)."""
        return self._xlines

    @xlines.setter
    def xlines(self, values):
        if isinstance(values, np.ndarray) and values.shape[0] == self._nrow:
            self._xlines = values

    @property
    def xmin(self):
        """The minimim X coordinate (read only)."""
        corners = self.get_map_xycorners()

        xmin = VERYLARGEPOSITIVE
        for corner in corners:
            if corner[0] < xmin:
                xmin = corner[0]
        return xmin

    @property
    def xmax(self):
        """The maximum X coordinate (read only)."""
        corners = self.get_map_xycorners()

        xmax = VERYLARGENEGATIVE
        for corner in corners:
            if corner[0] > xmax:
                xmax = corner[0]
        return xmax

    @property
    def ymin(self):
        """The minimim Y coordinate (read only)."""
        corners = self.get_map_xycorners()

        ymin = VERYLARGEPOSITIVE
        for corner in corners:
            if corner[1] < ymin:
                ymin = corner[1]
        return ymin

    @property
    def ymax(self):
        """The maximum Y xoordinate (read only)."""
        corners = self.get_map_xycorners()

        ymax = VERYLARGENEGATIVE
        for corner in corners:
            if corner[1] > ymax:
                ymax = corner[1]
        return ymax

    @property
    def dtype(self) -> np.dtype:
        """Getting the dtype of the values (np.array); float64 or float32"""
        # this is not stored as state varible, but derived from the actual values
        try:
            infer_dtype = self._values.dtype
        except AttributeError:
            infer_dtype = np.float64
        return infer_dtype

    @dtype.setter
    def dtype(self, wanted_dtype: Union[Type[np.float64], Type[np.float32]]):
        """Setting the dtype of the values (np.array); float64 or float32"""
        try:
            self._values = self._values.astype(wanted_dtype)
        except TypeError as msg:
            warnings.warn(f"Cannot change dtype: {msg}. Will keep current", UserWarning)
            return

    @property
    def values(self):
        """The map values, as 2D masked numpy (float64/32), shape (ncol, nrow).

        When setting values as a scalar, the current mask will be preserved.

        When setting values, list-like input (lists, tuples) is also accepted, as
        long as the length is correct and the entries are number-like.

        In order to specify undefined values, you can specify the ``undef`` attribute
        in the list, or use ``float("nan")``.

        Example::

            # list like input where nrow=3 and ncol=5 (15 entries)
            newvalues = list(range(15))
            newvalues[2] = srf.undef
            srf.values = newvalues  # here, entry 2 will be undefined
        """
        return self._values

    @values.setter
    def values(self, values):
        self._ensure_correct_values(values)

    @property
    def values1d(self):
        """(Read only) Map values, as 1D numpy masked, normally a numpy view(?).

        Example::

            map = xtgeo.surface_from_file('myfile.gri')
            map.values1d
        """
        return self.get_values1d(asmasked=True)

    @property
    def npvalues1d(self):
        """(Read only) Map values, as 1D numpy (not masked), undef as np.nan.

        In most cases this will be a copy of the values.

        Example::

            >>> import xtgeo
            >>> map = xtgeo.surface_from_file(surface_dir + '/topreek_rota.gri')
            >>> values = map.npvalues1d
            >>> mean = np.nanmean(values)
            >>> values[values <= 0] = np.nan
            >>> print(values)
            [nan nan ... nan]
        """
        return self.get_values1d(asmasked=False, fill_value=np.nan)

    @property
    def name(self):
        """A free form name for the surface, to be used in display etc."""
        return self._name

    @name.setter
    def name(self, newname):
        if isinstance(newname, str):
            self._name = newname

    @property
    def undef(self):
        """Returns the undef map value (read only)."""
        return self._undef

    @property
    def undef_limit(self):
        """Returns the undef_limit map value (read only)."""
        return self._undef_limit

    @property
    def filesrc(self):
        """Gives the name of the file source (if any)."""
        return self._filesrc

    @filesrc.setter
    def filesrc(self, name):
        self._filesrc = name  # checking is currently missing

    # ==================================================================================
    # Describe, import and export
    # ==================================================================================

    def generate_hash(self, hashmethod="md5"):
        """Return a unique hash ID for current instance.

        See :meth:`~xtgeo.common.sys.generic_hash()` for documentation.

        .. versionadded:: 2.14
        """
        required = (
            "ncol",
            "nrow",
            "xori",
            "yori",
            "xinc",
            "yinc",
            "yflip",
            "rotation",
        )
        gid = ""
        for req in required:
            gid += f"{getattr(self, '_' + req)}"
        # Ignore the mask
        gid += self._values.data.tobytes().hex()

        return generic_hash(gid, hashmethod=hashmethod)

    def describe(self, flush=True):
        """Describe an instance by printing to stdout."""
        #
        dsc = XTGDescription()
        dsc.title("Description of RegularSurface instance")
        dsc.txt("Object ID", id(self))
        dsc.txt("File source", self._filesrc)
        dsc.txt("Shape: NCOL, NROW", self.ncol, self.nrow)
        dsc.txt("Active cells vs total", self.nactive, self.nrow * self.ncol)
        dsc.txt("Origins XORI, YORI", self.xori, self.yori)
        dsc.txt("Increments XINC YINC", self.xinc, self.yinc)
        dsc.txt("Rotation (anti-clock from X)", self.rotation)
        dsc.txt("YFLIP flag", self._yflip)
        np.set_printoptions(threshold=16)
        dsc.txt("Inlines vector", self._ilines)
        dsc.txt("Xlines vector", self._xlines)
        np.set_printoptions(threshold=1000)
        if self._isloaded:
            dsc.txt("Values", self._values.reshape(-1), self._values.dtype)
            dsc.txt(
                "Values: mean, stdev, minimum, maximum",
                self.values.mean(),
                self.values.std(),
                self.values.min(),
                self.values.max(),
            )
            msize = float(self.values.size * 8) / (1024 * 1024 * 1024)
            dsc.txt("Minimum memory usage of array (GB)", msize)
        else:
            dsc.txt("Values:", "Not loaded")

        if flush:
            dsc.flush()
            return None

        return dsc.astext()

    @classmethod
    def _read_file(
        cls,
        mfile: Union[str, pathlib.Path, io.BytesIO],
        fformat: Optional[str] = None,
        load_values: bool = True,
        **kwargs,
    ):
        """Import surface (regular map) from file.

        Note that the ``fformat=None`` or ``guess`` option will guess format by
        looking at the file or stream signature or file extension.
        For the signature, the first bytes are scanned for 'patterns'. If that
        does not work (and input is not a memory stream), it will try to use
        file extension where e.g. "gri" will assume irap_binary and "fgr"
        assume Irap Ascii. If file extension is missing, Irap binary is assumed.

        The ``ijxyz`` format is the typical seismic format, on the form
        (ILINE, XLINE, X, Y, VALUE) as a table of points. Map values are
        estimated from the given values, or by using an existing map or
        cube as template, and match by ILINE/XLINE numbering.

        BytesIO input is supported for Irap binary, Irap Ascii, ZMAP ascii.

        Args:
            mfile: File-like or memory stream instance.
            fformat: File format, None/guess/irap_binary/irap_ascii/ijxyz
                is currently supported. If None or guess, the file 'signature' is
                used to guess format first, then file extension.
            load_values: If True (default), then full array is read, if False
                only metadata will be read. Valid for Irap binary only. This allows
                lazy loading in e.g. ensembles.
            kwargs: some readers allow additonal options

        Keyword Args:
            template: Only valid if ``ijxyz`` format, where an
                existing Cube or RegularSurface instance is applied to
                get correct topology.
            engine: Default is "cxtgeo" which use a C backend. Optionally a pure
                python "python" reader will be used, which in general is slower
                but may be safer when reading memory streams and/or threading.
                Engine is relevant for Irap binary, Irap ascii and zmap.

        Returns:
            Object instance.

        Example::

           >>> surf = RegularSurface._read_file(surface_dir + "/topreek_rota.gri")

        .. versionadded:: 2.14

        """
        mfile = FileWrapper(mfile)
        mfile.check_file(raiseerror=ValueError)
        fmt = mfile.fileformat(fformat)

        new_kwargs = _data_reader_factory(fmt)(mfile, values=load_values, **kwargs)
        new_kwargs["filesrc"] = mfile.file
        new_kwargs["fformat"] = fmt
        new_kwargs["dtype"] = kwargs.get("dtype", np.float64)
        return cls(**new_kwargs)

    def load_values(self):
        """Import surface values in cases where metadata only is loaded.

        Currently, only Irap binary format is supported.

        Example::

            surfs = []
            for i in range(1000):
                surfs.append(xtgeo.surface_from_file(f"myfile{i}.gri", values=False))

            # load values in number 88:
            surfs[88].load_values()

        .. versionadded:: 2.1
        """

        if not self._isloaded:
            if self.filesrc is None:
                raise ValueError(
                    "Can only load values into object initialised from file"
                )

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                mfile = FileWrapper(self.filesrc)
                kwargs = _data_reader_factory(self._fformat)(mfile, values=True)
                self.values = kwargs.get("values", self._values)

            self._isloaded = True

    def to_file(
        self,
        mfile: Union[str, pathlib.Path, io.BytesIO],
        fformat: Optional[str] = "irap_binary",
        pmd_dataunits: Optional[Tuple[int, int]] = (15, 10),
        engine: Optional[str] = "cxtgeo",
        error_if_near_empty: bool = False,
    ):
        """Export a surface (map) to file.

        Note, for zmap_ascii and storm_binary an unrotation will be done
        automatically. The sampling will be somewhat finer than the
        original map in order to prevent aliasing. See :func:`unrotate`.

        Args:
            mfile: Name of file,
                Path instance or IOBytestream instance. An alias can be e.g.
                "%md5sum%" or "%fmu-v1%" with string or Path() input.
            fformat: File format, irap_binary/irap_ascii/zmap_ascii/
                storm_binary/ijxyz/petromod/xtg*. Default is irap_binary.
            pmd_dataunits: A tuple of length 2 for petromod format,
                spesifying metadata for units (DataUnitDistance, DataUnitZ).
            engine: Default is "cxtgeo" which use a C backend. Optionally a pure
                python "python" reader will be used, which in general is slower
                but may be safer when reading memory streams and/or threading. Engine
                is relevant for Irap binary, Irap ascii and zmap. This is mainly a
                developer setting.
            error_if_near_empty: Default is False. If True, raise a RuntimeError if
                number of map nodes is less than 4. Otherwise, if False and number of
                nodes are less than 4, a UserWarning will be given.

        Returns:
            ofile (pathlib.Path): The actual file instance, or None if io.Bytestream

        Examples::

            >>> # read and write to ordinary file
            >>> surf = xtgeo.surface_from_file(
            ...     surface_dir + '/topreek_rota.fgr',
            ...     fformat = 'irap_ascii'
            ... )
            >>> surf.values = surf.values + 300
            >>> filename = surf.to_file(
            ...     outdir + '/topreek_rota300.fgr',
            ...     fformat = 'irap_ascii'
            ... )

            >>> # writing to io.BytesIO:
            >>> stream = io.BytesIO()
            >>> surf.to_file(stream, fformat="irap_binary")

            >>> # read from memory stream:
            >>> _ = stream.seek(0)
            >>> newsurf = xtgeo.surface_from_file(stream, fformat = 'irap_binary')

        .. versionchanged:: 2.5 Added support for BytesIO
        .. versionchanged:: 2.13 Improved support for BytesIO
        .. versionchanged:: 2.14 Support for alias file name and return value
        .. versionchanged:: 3.8 Add key ``error_if_near_empty``
        """
        logger.info("Export RegularSurface to file or memstream...")
        if self.nactive is None or self.nactive < 4:
            msg = (
                f"Number of maps nodes are {self.nactive}. Exporting regular "
                "surfaces with fewer than 4 nodes will not provide any "
                "usable result. The map may also be not loaded if nodes are None."
            )

            if error_if_near_empty:
                raise RuntimeError(msg)
            warnings.warn(msg, UserWarning)

        mfile = FileWrapper(mfile, mode="wb", obj=self)
        mfile.check_folder(raiseerror=OSError)

        if mfile.memstream:
            engine = "python"

        if fformat in FileFormat.IRAP_ASCII.value:
            _regsurf_export.export_irap_ascii(self, mfile, engine=engine)

        elif fformat in FileFormat.IRAP_BINARY.value:
            _regsurf_export.export_irap_binary(self, mfile, engine=engine)

        elif fformat in FileFormat.ZMAP_ASCII.value:
            _regsurf_export.export_zmap_ascii(self, mfile, engine=engine)

        elif fformat in FileFormat.STORM.value:
            _regsurf_export.export_storm_binary(self, mfile)

        elif fformat in FileFormat.PETROMOD.value:
            _regsurf_export.export_petromod_binary(self, mfile, pmd_dataunits)

        elif fformat in FileFormat.IJXYZ.value:
            _regsurf_export.export_ijxyz_ascii(self, mfile)

        elif fformat == "xtgregsurf":
            _regsurf_export.export_xtgregsurf(self, mfile)

        else:
            extensions = FileFormat.extensions_string(
                [
                    FileFormat.IRAP_BINARY,
                    FileFormat.IRAP_ASCII,
                    FileFormat.IJXYZ,
                    FileFormat.PETROMOD,
                    FileFormat.ZMAP_ASCII,
                    FileFormat.XTG,
                    FileFormat.HDF,
                ]
            )
            raise InvalidFileFormatError(
                f"File format {fformat} is invalid for type RegularSurface. "
                f"Supported formats are {extensions}."
            )

        logger.info("Export RegularSurface to file or memstream... done")

        if mfile.memstream:
            return None
        return mfile.file

    def to_hdf(
        self,
        mfile: Union[str, pathlib.Path, io.BytesIO],
        compression: Optional[str] = "lzf",
    ) -> pathlib.Path:
        """Export a surface (map) with metadata to a HDF5 file.

        Warning:
            This implementation is currently experimental and only recommended
            for testing.

        The file extension shall be '.hdf'

        Args:
            mfile: Name of file, Path instance or BytesIO instance. An alias can
                be e.g. ``$md5sum.hdf``,  ``$fmu-v1.hdf`` with string or Path() input.
            compression: Compression method, None, lzf (default), blosc

        Returns:
            pathlib.Path: The actual file instance, or None if io.Bytestream

        Example:
            >>> import xtgeo
            >>> surf = xtgeo.surface_from_file(surface_dir + '/topreek_rota.gri')
            >>> filepath = surf.to_hdf(outdir + "/topreek_rota.hdf")

        .. versionadded:: 2.14

        """
        # developing, in prep and experimental!
        mfile = FileWrapper(mfile, mode="wb", obj=self)

        mfile.check_folder(raiseerror=OSError)

        _regsurf_export.export_hdf5_regsurf(self, mfile, compression=compression)
        return mfile.file

    @classmethod
    def _read_roxar(
        cls,
        project,
        name,
        category,
        stype="horizons",
        realisation=0,
        dtype=np.float64,
    ):  # pragma: no cover
        """Load a surface from a Roxar RMS project.

        The import from the RMS project can be done either within the project
        or outside the project.

        Note that a shortform to::

          import xtgeo
          mysurf = xtgeo.surface_from_roxar(project, 'name', 'category')

        Note also that horizon/zone name and category must exists in advance,
        otherwise an Exception will be raised.

        Args:
            project (str or special): Name of project (as folder) if
                outside RMS, og just use the magic project word if within RMS.
            name (str): Name of surface/map
            category (str): For horizons/zones or clipboard/general2d_data: for
                example 'DS_extracted'
            stype (str): RMS folder type, 'horizons' (default), 'zones' or 'clipboard'
            realisation (int): Realisation number, default is 0
            dtype: For supporting conversion to 32 bit float for the numpy; default
                is 64 bit

        """
        kwargs = _regsurf_roxapi.import_horizon_roxapi(
            project, name, category, stype, realisation
        )
        kwargs["dtype"] = dtype  # eventual dtype change will be done in __init__

        return cls(**kwargs)

    def to_roxar(
        self, project, name, category, stype="horizons", realisation=0
    ):  # pragma: no cover
        """Store (export) a regular surface to a Roxar RMS project.

        The export to the RMS project can be done either within the project
        or outside the project. The storing is done to the Horizons or the
        Zones folder in RMS.

        Note:
            The horizon or zone name and category must exists in advance,
            otherwise an Exception will be raised.

            When project is file path (direct access, outside RMS) then
            ``to_roxar()`` will implicitly do a project save. Otherwise, the project
            will not be saved until the user do an explicit project save action.

        Args:
            project (str or special): Name of project (as folder) if
                outside RMS, og just use the magic project word if within RMS.
            name (str): Name of surface/map
            category (str): Required for horizons/zones: e.g. 'DS_extracted'. For
                clipboard/general2d_data is reperesent the folder(s), where "" or None
                means no folder, while e.g. "myfolder/subfolder" means that folders
                myfolder/subfolder will be created if not already present. For
                stype = 'trends', the category will not be applied
            stype (str): RMS folder type, 'horizons' (default), 'zones', 'clipboard'
                'general2d_data', 'trends'
            realisation (int): Realisation number, default is 0

        Raises:
            ValueError: If name or category does not exist in the project

        Example:
            Here the from_roxar method is used to initiate the object
            directly::

              import xtgeo
              topupperreek = xtgeo.surface_from_roxar(project, 'TopUpperReek',
                                                    'DS_extracted')
              topupperreek.values += 200

              # export to file:
              topupperreek.to_file('topupperreek.gri')

              # store in project
              topupperreek.to_roxar(project, 'TopUpperReek', 'DS_something')

        Note::

            When dealing with surfaces to and from ``stype="trends"``, the surface must
            exist in advance, i.e. the Roxar API do not allow creating new surfaces.
            Actually trends are read only, but a workaround using ``load()`` in Roxar
            API makes it possible to overwrite existing surface trends. In addition,
            ``realisation`` is not applied in trends.


        .. versionadded:: 2.1 clipboard support
        .. versionadded:: 2.19 general2d_data and trends support

        """
        _regsurf_roxapi.export_horizon_roxapi(
            self, project, name, category, stype, realisation
        )

    @classmethod
    def _read_cube(cls, cube, zlevel):
        """Make a constant surface from a Cube, at a given time/depth level.

        The surface instance will have exactly the same origins and increments
        as the cube.

        Args:
            cube (Cube): XTGeo Cube instance
            zlevel (float): Depth or Time (or whatever) value of the surface

        Returns:
            Object instance updated

        Example:
            Here the from_roxar method is used to initiate the object
            directly::

            >>> mycube = xtgeo.cube_from_file(cube_dir + "/ib_test_cube2.segy")
            >>> mymap = RegularSurface._read_cube(mycube, 2700)

        """
        props = [
            "ncol",
            "nrow",
            "xori",
            "yori",
            "xinc",
            "yinc",
            "rotation",
            "ilines",
            "xlines",
            "yflip",
        ]

        input_dict = {key: deepcopy(getattr(cube, key)) for key in props}

        input_dict["values"] = ma.array(
            np.full((input_dict["ncol"], input_dict["nrow"]), zlevel, dtype=np.float64)
        )
        return cls(**input_dict)

    @classmethod
    def _read_grid3d(
        cls,
        grid: Grid,
        template: RegularSurface | str | None = None,
        where: str | int = "top",
        property: str | GridProperty = "depth",
        rfactor: int = 1,
        index_position: str = "center",
    ):
        """Private class method to extract a surface from a 3D grid."""
        args = _regsurf_grid3d.from_grid3d(
            grid, template, where, property, rfactor, index_position
        )
        return cls(**args)

    def copy(self):
        """Deep copy of a RegularSurface object to another instance.

        Example::

            >>> mymap = xtgeo.surface_from_file(surface_dir + '/topreek_rota.gri')
            >>> mymapcopy = mymap.copy()

        """

        xsurf = RegularSurface(
            ncol=self.ncol,
            nrow=self.nrow,
            xinc=self.xinc,
            yinc=self.yinc,
            xori=self.xori,
            yori=self.yori,
            rotation=self.rotation,
            yflip=self.yflip,
        )

        xsurf._values = self._values.copy()
        xsurf._isloaded = self._isloaded

        xsurf.ilines = self._ilines.copy()
        xsurf.xlines = self._xlines.copy()
        xsurf.filesrc = self._filesrc
        xsurf.metadata.required = xsurf

        return xsurf

    def get_values1d(
        self, order="C", asmasked=False, fill_value=UNDEF, activeonly=False
    ):
        """Get a 1D numpy or masked array of the map values.

        Args:
            order (str): Flatteting is in C (default) or F order
            asmasked (bool): If true, return as MaskedArray, other as standard
                numpy ndarray with undef as np.nan or fill_value
            fill_value (str): Relevent only if asmasked is False, this
                will be the value of undef entries
            activeonly (bool): If True, only active cells. Keys 'asmasked' and
                'fill_value' are not revelant.

        Returns:
            A numpy 1D array or MaskedArray

        """
        val = self.values.copy()

        if order == "F":
            val = ma.array(val.data, mask=val.mask, order="F")

        val = val.ravel(order=order)

        if activeonly:
            val = val[~val.mask]

        if not asmasked and not activeonly:
            val = ma.filled(val, fill_value=fill_value)

        return val

    def set_values1d(self, val, order="C"):
        """Update the values attribute based on a 1D input, multiple options.

        If values are np.nan or values are > UNDEF_LIMIT, they will be
        masked.

        Args:
            val (list-like): Set values as a 1D array
            order (str): Input is C (default) or F order
        """
        if order == "F":
            val = np.copy(val, order="C")

        val = val.reshape((self.ncol, self.nrow))

        if not isinstance(val, ma.MaskedArray):
            val = ma.array(val)

        val = ma.masked_greater(val, self.undef_limit)
        val = ma.masked_invalid(val)

        self.values = val

    def get_rotation(self):
        """Returns the surface roation, in degrees, from X, anti-clock."""
        return self._rotation

    def get_nx(self):
        """Same as ncol (nx) (for backward compatibility)."""
        return self._ncol

    def get_ny(self):
        """Same as nrow (ny) (for backward compatibility)."""
        return self._nrow

    def get_xori(self):
        """Same as property xori (for backward compatibility)."""
        return self._xori

    def get_yori(self):
        """Same as property yori (for backward compatibility)."""
        return self._yori

    def get_xinc(self):
        """Same as property xinc (for backward compatibility)."""
        return self._xinc

    def get_yinc(self):
        """Same as property yinc (for backward compatibility)."""
        return self._yinc

    def similarity_index(self, other):
        """Report the degree of similarity between two maps, by comparing mean.

        The method computes the average per surface, and the similarity
        is difference in mean divided on mean of self. I.e. values close
        to 0.0 mean small difference.

        Args:
            other (surface object): The other surface to compare with

        """
        ovalues = other.values
        svalues = self.values

        diff = math.pow(svalues.mean() - ovalues.mean(), 2)
        diff = math.sqrt(diff)

        try:
            diff = diff / svalues.mean()
        except ZeroDivisionError:
            diff = -999

        return diff

    def compare_topology(self, other, strict=True):
        """Check that two object has the same topology, i.e. map definitions.

        Map definitions such as origin, dimensions, number of defined cells...

        Args:
            other (surface object): The other surface to compare with
            strict (bool): If false, the masks are not compared

        Returns:
            True of same topology, False if not
        """
        tstatus = True

        # consider refactor to getattr() instead!
        chklist = {
            "_ncol",
            "_nrow",
            "_xori",
            "_yori",
            "_xinc",
            "_yinc",
            "_rotation",
        }
        for skey, sval in self.__dict__.items():
            if skey in chklist:
                for okey, oval in other.__dict__.items():
                    if skey == okey and sval != oval:
                        logger.info("CMP %s: %s vs %s", skey, sval, oval)
                        tstatus = False

        if not tstatus:
            return False

        # check that masks are equal
        mas1 = ma.getmaskarray(self.values)
        mas2 = ma.getmaskarray(other.values)
        if (
            strict
            and isinstance(mas1, np.ndarray)
            and isinstance(mas2, np.ndarray)
            and not np.array_equal(mas1, mas2)
        ):
            logger.warning("Masks differ, not consistent with 'strict'")
            return False
        return True

    def swapaxes(self):
        """Swap (flip) the axes columns vs rows, keep origin but reverse yflip."""
        _regsurf_utils.swapaxes(self)

    def make_lefthanded(self) -> None:
        """Makes the surface lefthanded in case yflip is -1. This will change origin.

        Lefhanded regular maps are common in subsurface data, where I is to east, J is
        to north and Z axis is positive down for depth and time data.

        The instance is changed in-place.

        .. versionadded:: 4.2
        """
        _regsurf_utils.make_lefthanded(self)

    def get_map_xycorners(self):
        """Get the X and Y coordinates of the map corners.

        Returns a tuple on the form
        ((x0, y0), (x1, y1), (x2, y2), (x3, y3)) where
        (if unrotated and normal flip) 0 is the lower left
        corner, 1 is the right, 2 is the upper left, 3 is the upper right.
        """
        rot1 = self._rotation * math.pi / 180
        rot2 = rot1 + (math.pi / 2.0)

        xc0 = self._xori
        yc0 = self._yori

        xc1 = self._xori + (self.ncol - 1) * math.cos(rot1) * self._xinc
        yc1 = self._yori + (self.ncol - 1) * math.sin(rot1) * self._xinc

        xc2 = self._xori + (self.nrow - 1) * math.cos(rot2) * self._yinc * self._yflip
        yc2 = self._yori + (self.nrow - 1) * math.sin(rot2) * self._yinc * self._yflip

        xc3 = xc2 + (self.ncol - 1) * math.cos(rot1) * self._xinc
        yc3 = yc2 + (self.ncol - 1) * math.sin(rot1) * self._xinc

        return ((xc0, yc0), (xc1, yc1), (xc2, yc2), (xc3, yc3))

    def get_value_from_xy(self, point=(0.0, 0.0), sampling="bilinear"):
        """Return the map value given a X Y point.

        Args:
            point (float tuple): Position of X and Y coordinate
            sampling (str): Sampling method, either "bilinear" for bilinear
                interpolation, or "nearest" for nearest node sampling (e.g. facies maps)

        Returns:
            The map value (interpolated). None if XY is outside defined map

        Example::
            mvalue = map.get_value_from_xy(point=(539291.12, 6788228.2))


        .. versionchanged:: 2.14 Added keyword option `sampling`
        """
        return _regsurf_oper.get_value_from_xy(self, point=point, sampling=sampling)

    def get_xy_value_from_ij(self, iloc, jloc, zvalues=None):
        """Returns x, y, z(value) from a single i j location.

        Args:
            iloc (int): I (col) location (base is 1)
            jloc (int): J (row) location (base is 1)
            zvalues (ndarray). If this is used in a loop it is wise
                to precompute the numpy surface once in the caller,
                and submit the numpy array (use surf.get_values1d()).

        Returns:
            The x, y, z values at location iloc, jloc
        """
        xval, yval, value = _regsurf_oper.get_xy_value_from_ij(
            self, iloc, jloc, zvalues=zvalues
        )

        return xval, yval, value

    def get_ij_values(self, zero_based=False, asmasked=False, order="C"):
        """Return I J numpy 2D arrays, optionally as masked arrays.

        Args:
            zero_based (bool): If False, first number is 1, not 0
            asmasked (bool): If True, UNDEF map nodes are skipped
            order (str): 'C' (default) or 'F' order (row vs column major)
        """
        return _regsurf_oper.get_ij_values(
            self, zero_based=zero_based, asmasked=asmasked, order=order
        )

    def get_ij_values1d(self, zero_based=False, activeonly=True, order="C"):
        """Return I J numpy as 1D arrays.

        Args:
            zero_based (bool): If False, first number is 1, not 0
            activeonly (bool): If True, UNDEF map nodes are skipped
            order (str): 'C' (default) or 'F' order (row vs column major)
        """
        return _regsurf_oper.get_ij_values1d(
            self, zero_based=zero_based, activeonly=activeonly, order=order
        )

    def get_xy_values(self, order="C", asmasked=True):
        """Return coordinates for X and Y as numpy (masked) 2D arrays.

        Args:
            order (str): 'C' (default) or 'F' order (row major vs column major)
            asmasked (bool): If True , inactive nodes are masked.
        """
        xvals, yvals = _regsurf_oper.get_xy_values(self, order=order, asmasked=asmasked)

        return xvals, yvals

    def get_xy_values1d(self, order="C", activeonly=True):
        """Return coordinates for X and Y as numpy 1D arrays.

        Args:
            order (str): 'C' (default) or 'F' order (row major vs column major)
            activeonly (bool): Only active cells are returned.
        """
        xvals, yvals = _regsurf_oper.get_xy_values1d(
            self, order=order, activeonly=activeonly
        )

        return xvals, yvals

    def get_xyz_values(self):
        """Return coordinates for X Y and Z (values) as numpy (masked) 2D arrays."""
        xcoord, ycoord = self.get_xy_values(asmasked=True)

        values = self.values.copy()

        return xcoord, ycoord, values

    def get_xyz_values1d(self, order="C", activeonly=True, fill_value=np.nan):
        """Return coordinates for X Y and Z (values) as numpy 1D arrays.

        Args:
            order (str): 'C' (default) or 'F' order (row major vs column major)
            activeonly (bool): Only active cells are returned.
            fill_value (float): If activeonly is False, value of inactive nodes
        """
        xcoord, ycoord = self.get_xy_values1d(order=order, activeonly=activeonly)

        values = self.get_values1d(
            order=order, asmasked=False, fill_value=fill_value, activeonly=activeonly
        )

        return xcoord, ycoord, values

    def get_dataframe(
        self, ijcolumns=False, ij=False, order="C", activeonly=True, fill_value=np.nan
    ):
        """Return a Pandas dataframe object, with columns X_UTME, Y_UTMN, VALUES.

        Args:
            ijcolumns (bool): If True, and IX and JY indices will be
               added as dataframe columns. Redundant, use "ij" instead.
            ij (bool): Same as ijcolumns. If True, and IX and JY indices will be
               added as dataframe columns. Preferred syntax
            order (str): 'C' (default) for C order (row fastest), or 'F'
               for Fortran order (column fastest)
            activeonly (bool): If True, only active nodes are listed. If
                False, the values will have fill_value default None = NaN
                as values
            fill_value (float): Value of inactive nodes if activeonly is False

        Example::

            >>> import xtgeo
            >>> surf = xtgeo.surface_from_file(surface_dir + '/topreek_rota.gri')
            >>> dfr = surf.get_dataframe()
            >>> dfr.to_csv('somecsv.csv')

        Returns:
            A Pandas dataframe object.
        """
        xcoord, ycoord, values = self.get_xyz_values1d(
            order=order, activeonly=activeonly, fill_value=fill_value
        )

        entry = {}

        if ijcolumns or ij:
            ixn, jyn = self.get_ij_values1d(order=order, activeonly=activeonly)
            entry["IX"] = ixn
            entry["JY"] = jyn

        entry.update([("X_UTME", xcoord), ("Y_UTMN", ycoord), ("VALUES", values)])

        return pd.DataFrame(entry)

    def dataframe(self, **kwargs):
        """Deprecated; see method get_dataframe()."""
        warnings.warn(
            "The dataframe() is deprecated and will be removed in xtgeo "
            "version 5. Use get_dataframe() instead",
            PendingDeprecationWarning,
        )

        return self.get_dataframe(**kwargs)

    def get_xy_value_lists(self, lformat="webportal", xyfmt=None, valuefmt=None):
        """Returns two lists for coordinates (x, y) and values.

        For lformat = 'webportal' (default):

        The lists are returned as xylist and valuelist, where xylist
        is on the format:

            [(x1, y1), (x2, y2) ...] (a list of x, y tuples)

        and valuelist is one the format

            [v1, v2, ...]

        Inactive cells will be ignored.

        Args:
            lformat (string): List return format ('webportal' is default,
                other options later)
            xyfmt (string): The formatter for xy numbers, e.g. '12.2f'
                (default None). Note no checks on valid input.
            valuefmt (string): The formatter for values e.g. '8.4f'
                (default None). Note no checks on valid input.

        Returns:
            xylist, valuelist

        Example:

            >>> import xtgeo
            >>> surf = xtgeo.surface_from_file(surface_dir + '/topreek_rota.gri')
            >>> xylist, valuelist = surf.get_xy_value_lists(valuefmt='6.2f')
        """
        xylist = []
        valuelist = []

        zvalues = self.get_values1d()

        if lformat != "webportal":
            raise ValueError("Unsupported lformat")

        for jnum in range(self.nrow):
            for inum in range(self.ncol):
                xcv, ycv, vcv = self.get_xy_value_from_ij(
                    inum + 1, jnum + 1, zvalues=zvalues
                )

                if vcv is not None:
                    if xyfmt is not None:
                        xcv = float(f"{xcv:{xyfmt}}")
                        ycv = float(f"{ycv:{xyfmt}}")
                    if valuefmt is not None:
                        vcv = float(f"{vcv:{valuefmt}}")
                    valuelist.append(vcv)
                    xylist.append((xcv, ycv))

        return xylist, valuelist

    # ==================================================================================
    # Crop, interpolation, smooth or fill of values (possibly many methods here)
    # ==================================================================================

    def autocrop(self):
        """Automatic cropping of the surface to minimize undefined areas.

        This method is simply removing undefined "white areas". The
        instance will be updated with new values for xori, yori, ncol, etc. Rotation
        will never change

        Returns:
            RegularSurface instance is updated in-place

        .. versionadded:: 2.12
        """
        _regsurf_utils.autocrop(self)

    def fill(self, fill_value=None):
        """Fast infilling of undefined values.

        Note that minimum and maximum values will not change.

        Algorithm if `fill_value` is not set is based on a nearest node extrapolation.
        Technically, ``scipy.ndimage.distance_transform_edt`` is applied. If fill_value
        is set by a scalar, that (constant) value be be applied

        Args:
            fill_value (float): If defined, fills all undefined cells with that value.

        Returns:
            RegularSurface instance is updated in-place

        .. versionadded:: 2.1
        .. versionchanged:: 2.6 Added option key `fill_value`
        """
        _regsurf_gridding.surf_fill(self, fill_value=fill_value)

    def smooth(
        self,
        method: Literal["median", "gaussian"] = "median",
        iterations: int = 1,
        width: float = 1,
    ) -> None:
        """Various smoothing methods for surfaces.

        Args:
            method: Smoothing method (median or gaussian)
            iterations: Number of iterations
            width:
                - If method is 'median' range of influence is in nodes.
                - If method is 'gaussian' range of influence is standard
                  deviation of the Gaussian kernel.

        .. versionadded:: 2.1
        """

        if method == "median":
            _regsurf_gridding._smooth(
                self,
                window_function=functools.partial(
                    scipy.ndimage.median_filter, size=int(width)
                ),
                iterations=iterations,
            )
        elif method == "gaussian":
            _regsurf_gridding._smooth(
                self,
                window_function=functools.partial(
                    scipy.ndimage.gaussian_filter, sigma=width
                ),
                iterations=iterations,
            )
        else:
            raise ValueError("Unsupported method for smoothing")

    # ==================================================================================
    # Operation on map values (list to be extended)
    # ==================================================================================

    def operation(self, opname, value):
        """Do operation on map values.

        Do operations on the current map values. Valid operations are:

        * 'elilt' or 'eliminatelessthan': Eliminate less than <value>

        * 'elile' or 'eliminatelessequal': Eliminate less or equal than <value>

        Args:
            opname (str): Name of operation. See list above.
            value (*): A scalar number (float) or a tuple of two floats,
                dependent on operation opname.

        Examples::

            surf.operation('elilt', 200)  # set all values < 200 as undef
        """
        if opname in ("elilt", "eliminatelessthan"):
            self._values = ma.masked_less(self._values, value)
        elif opname in ("elile", "eliminatelessequal"):
            self._values = ma.masked_less_equal(self._values, value)
        else:
            raise ValueError("Invalid operation name")

    # ==================================================================================
    # Operations restricted to inside/outside polygons
    # ==================================================================================

    def operation_polygons(self, poly, value, opname="add", inside=True, _version=2):
        """A generic function for map operations inside or outside polygon(s).

        Args:
            poly (Polygons): A XTGeo Polygons instance
            value(float or RegularSurface): Value to add, subtract etc
            opname (str): Name of operation... 'add', 'sub', etc
            inside (bool): If True do operation inside polygons; else outside.
            _version (int): Algorithm version, 2 will be much faster when many points
                on polygons (this key will be removed in later versions and shall not
                be applied)
        """
        if _version == 2:
            _regsurf_oper.operation_polygons_v2(
                self, poly, value, opname=opname, inside=inside
            )
        else:
            _regsurf_oper.operation_polygons(
                self, poly, value, opname=opname, inside=inside
            )

    # shortforms
    def add_inside(self, poly, value):
        """Add a value (scalar or other map) inside polygons."""
        self.operation_polygons(poly, value, opname="add", inside=True)

    def add_outside(self, poly, value):
        """Add a value (scalar or other map) outside polygons."""
        self.operation_polygons(poly, value, opname="add", inside=False)

    def sub_inside(self, poly, value):
        """Subtract a value (scalar or other map) inside polygons."""
        self.operation_polygons(poly, value, opname="sub", inside=True)

    def sub_outside(self, poly, value):
        """Subtract a value (scalar or other map) outside polygons."""
        self.operation_polygons(poly, value, opname="sub", inside=False)

    def mul_inside(self, poly, value):
        """Multiply a value (scalar or other map) inside polygons."""
        self.operation_polygons(poly, value, opname="mul", inside=True)

    def mul_outside(self, poly, value):
        """Multiply a value (scalar or other map) outside polygons."""
        self.operation_polygons(poly, value, opname="mul", inside=False)

    def div_inside(self, poly, value):
        """Divide a value (scalar or other map) inside polygons."""
        self.operation_polygons(poly, value, opname="div", inside=True)

    def div_outside(self, poly, value):
        """Divide a value (scalar or other map) outside polygons."""
        self.operation_polygons(poly, value, opname="div", inside=False)

    def set_inside(self, poly, value):
        """Set a value (scalar or other map) inside polygons."""
        self.operation_polygons(poly, value, opname="set", inside=True)

    def set_outside(self, poly, value):
        """Set a value (scalar or other map) outside polygons."""
        self.operation_polygons(poly, value, opname="set", inside=False)

    def eli_inside(self, poly):
        """Eliminate current map values inside polygons."""
        self.operation_polygons(poly, 0, opname="eli", inside=True)

    def eli_outside(self, poly):
        """Eliminate current map values outside polygons."""
        self.operation_polygons(poly, 0, opname="eli", inside=False)

    # ==================================================================================
    # Operation with secondary map
    # ==================================================================================

    def add(self, other):
        """Add another map to current map."""
        _regsurf_oper.operations_two(self, other, oper="add")

    def subtract(self, other):
        """Subtract another map from current map."""
        _regsurf_oper.operations_two(self, other, oper="sub")

    def multiply(self, other):
        """Multiply another map and current map."""
        _regsurf_oper.operations_two(self, other, oper="mul")

    def divide(self, other):
        """Divide current map with another map."""
        _regsurf_oper.operations_two(self, other, oper="div")

    # ==================================================================================
    # Interacion with points
    # ==================================================================================

    def gridding(self, points, method="linear", coarsen=1):
        """Grid a surface from points.

        Args:
            points(Points): XTGeo Points instance.
            method (str): Gridding method option: linear / cubic / nearest
            coarsen (int): Coarsen factor, to speed up gridding, but will
                give poorer result.

        Example::

            >>> import xtgeo
            >>> mypoints = xtgeo.Points(points_dir + '/pointset2.poi')
            >>> mysurf = xtgeo.surface_from_file(surface_dir + '/topreek_rota.gri')

            >>> # update the surface by gridding the points
            >>> mysurf.gridding(mypoints)

        Raises:
            RuntimeError: If not possible to grid for some reason
            ValueError: If invalid input

        """
        if not isinstance(points, Points):
            raise ValueError("Argument not a Points instance")

        logger.info("Do gridding...")

        _regsurf_gridding.points_gridding(self, points, coarsen=coarsen, method=method)

    # ==================================================================================
    # Interacion with other surface
    # ==================================================================================

    def resample(self, other, mask=True, sampling="bilinear"):
        """Resample an instance surface values from another surface instance.

        Note that there may be some 'loss' of nodes at the edges of the
        updated map, as only the 'inside' nodes in the updated map
        versus the input map are applied.

        The interpolation algorithm in resample is bilinear interpolation. The
        topolopogy of the surface (map definitions, rotation, ...) will not change,
        only the map values. Areas with undefined nodes in ``other`` will become
        undefined in the instance if mask is True; othewise they will be kept as is.

        Args:
            other (RegularSurface): Surface to resample from.
            mask (bool): If True (default) nodes outside will be made undefined,
                if False then values will be kept as original
            sampling (str): Either 'bilinear' interpolation (default) or, 'nearest' for
                nearest node. The latter can be useful for resampling discrete maps.

        Example::

            # map with 230x210 columns, rotation 20
            surf1 = xtgeo.surface_from_file("some1.gri")
            # map with 270x190 columns, rotation 0
            surf2 = xtgeo.surface_from_file("some2.gri")
            # will sample (interpolate) surf2's values to surf1
            surf1.resample(surf2)

        Returns:
            Instance's surface values will be updated in-place.


        .. versionchanged:: 2.9
           Added ``mask`` keyword, default is True for backward compatibility.

        .. versionchanged:: 2.21
           Added ``sampling`` keyword option.

        """
        if not isinstance(other, RegularSurface):
            raise ValueError("Argument not a RegularSurface instance")

        logger.info("Do resampling...")

        _regsurf_oper.resample(self, other, mask=mask, sampling=sampling)

    # ==================================================================================
    # Change a surface more fundamentally
    # ==================================================================================

    def unrotate(self, factor=2):
        r"""Unrotete a map instance, and this will also change nrow, ncol, xinc, etc.

        The default sampling (factor=2) makes a finer grid in order to
        avoid artifacts, and this default can be used in most cases.

        If an even finer grid is wanted, increase the factor. Theoretically the
        new increment for factor=N is between :math:`\\frac{1}{N}` and
        :math:`\\frac{1}{N}\\sqrt{2}` of the original increment,
        dependent on the rotation of the original surface.

        If the current instance already is unrotated, nothing is done.

        Args:
            factor (int): Refinement factor (>= 1)

        """
        if abs(self.rotation) < 0.00001:
            logger.info("Surface has no rotation, nothing is done")
            return

        if factor < 1:
            raise ValueError("Unrotate refinement factor cannot be be less than 1")

        if not isinstance(factor, int):
            raise ValueError("Refinementfactor must an integer")

        scopy = self
        if scopy._yflip < 0:
            scopy = self.copy()
            scopy.swapaxes()

        xlen = scopy.xmax - scopy.xmin
        ylen = scopy.ymax - scopy.ymin
        ncol = scopy.ncol * factor
        nrow = scopy.nrow * factor
        xinc = xlen / (ncol - 1)  # node based, not cell center based
        yinc = ylen / (nrow - 1)
        vals = ma.zeros((ncol, nrow), order="C")

        nonrot = RegularSurface(
            xori=scopy.xmin,
            yori=scopy.ymin,
            xinc=xinc,
            yinc=yinc,
            ncol=ncol,
            nrow=nrow,
            values=vals,
            yflip=1,
        )
        nonrot.resample(scopy)

        self._values = nonrot.values
        self._nrow = nonrot.nrow
        self._ncol = nonrot.ncol
        self._rotation = nonrot.rotation
        self._xori = nonrot.xori
        self._yori = nonrot.yori
        self._xinc = nonrot.xinc
        self._yinc = nonrot.yinc
        self._yflip = nonrot.yflip
        self._ilines = nonrot.ilines
        self._xlines = nonrot.xlines

    def refine(self, factor):
        """Refine a surface with a factor.

        Range for factor is 2 to 10.

        Note that there may be some 'loss' of nodes at the edges of the
        updated map, as only the 'inside' nodes in the updated map
        versus the input map are applied.

        Args:
            factor (int): Refinement factor
        """
        logger.info("Do refining...")

        if not isinstance(factor, int):
            raise ValueError("Argument not a, Integer")

        if factor < 2 or factor >= 10:
            raise ValueError("Argument exceeds range 2 .. 10")

        xlen = self._xinc * (self._ncol - 1)
        ylen = self._yinc * (self._nrow - 1)

        proxy = self.copy()
        self._ncol = proxy.ncol * factor
        self._nrow = proxy.nrow * factor
        self._xinc = xlen / (self._ncol - 1)
        self._yinc = ylen / (self._nrow - 1)

        self._values = ma.zeros((self._ncol, self._nrow))

        self._ilines = np.array(range(1, self._ncol + 1), dtype=np.int32)
        self._xlines = np.array(range(1, self._nrow + 1), dtype=np.int32)

        self.resample(proxy)

        del proxy
        logger.info("Do refining... done")

    def coarsen(self, factor):
        """Coarsen a surface with a factor.

        Range for coarsening is 2 to 10, where e.g. 2 meaning half the number of
        columns and rows.

        Note that there may be some 'loss' of nodes at the edges of the
        updated map, as only the 'inside' nodes in the updated map
        versus the input map are applied.

        Args:
            factor (int): Coarsen factor (2 .. 10)

        Raises:
            ValueError: Coarsen is too large, giving too few nodes in result
        """
        logger.info("Do coarsening...")
        if not isinstance(factor, int):
            raise ValueError("Argument not a, Integer")

        if factor < 2 or factor >= 10:
            raise ValueError("Argument exceeds range 2 .. 10")

        proxy = self.copy()
        xlen = self._xinc * (self._ncol - 1)
        ylen = self._yinc * (self._nrow - 1)

        ncol = int(round(proxy._ncol / factor))
        nrow = int(round(proxy._nrow / factor))

        if ncol < 4 or nrow < 4:
            raise ValueError(
                "Coarsen is too large, giving ncol or nrow less than 4 nodes"
            )

        self._ncol = ncol
        self._nrow = nrow

        self._xinc = xlen / (self._ncol - 1)
        self._yinc = ylen / (self._nrow - 1)

        self._values = ma.zeros((self._ncol, self._nrow))

        self._ilines = np.array(range(1, self._ncol + 1), dtype=np.int32)
        self._xlines = np.array(range(1, self._nrow + 1), dtype=np.int32)

        self.resample(proxy)

        del proxy
        logger.info("Do coarsening... done")

    # ==================================================================================
    # Interacion with a grid3d
    # ==================================================================================

    def slice_grid3d(self, grid, prop, zsurf=None, sbuffer=1):
        """Slice the grid property and update the instance surface to sampled values.

        Args:
            grid (Grid): Instance of a Grid.
            prop (GridProperty): Instance of a GridProperty, belongs to grid
            zsurf (surface object): Instance of map, which is used a slicer.
                If None, then the surface instance itself is used a slice
                criteria. Note that zsurf must have same map defs as the
                surface instance.
            sbuffer (int): Default is 1; if "holes" after sampling
                extend this to e.g. 3
        Example::

            >>> import xtgeo
            >>> grd = xtgeo.grid_from_file(reek_dir + '/REEK.EGRID')
            >>> prop = xtgeo.gridproperty_from_file(
            ...     reek_dir + '/REEK.UNRST',
            ...     name='PRESSURE',
            ...     date="first",
            ...     grid=grd,
            ... )
            >>> surf = xtgeo.surface_from_file(surface_dir + '/topreek_rota.gri')
            >>> # update surf to sample the 3D grid property:
            >>> surf.slice_grid3d(grd, prop)

        Raises:
            Exception if maps have different definitions (topology)
        """
        # TODO: Remove this when circular dependency untangled
        from xtgeo.grid3d.grid import Grid

        if not isinstance(grid, Grid):
            raise ValueError("First argument must be a grid instance")

        ier = _regsurf_grid3d.slice_grid3d(
            self, grid, prop, zsurf=zsurf, sbuffer=sbuffer
        )

        if ier != 0:
            raise RuntimeError(
                "Wrong status from routine; something went wrong. Contact the author"
            )

    # ==================================================================================
    # Interacion with a cube
    # ==================================================================================

    def slice_cube(
        self,
        cube,
        zsurf=None,
        sampling="nearest",
        mask=True,
        snapxy=False,
        deadtraces=True,
        algorithm=2,
    ):
        """Slice the cube and update the instance surface to sampled cube values.

        Args:
            cube (object): Instance of a Cube()
            zsurf (surface object): Instance of a depth (or time) map, which
                is the depth or time map (or...) that is used a slicer.
                If None, then the surface instance itself is used a slice
                criteria. Note that zsurf must have same map defs as the
                surface instance.
            sampling (str): 'nearest' for nearest node (default), or
                'trilinear' for trilinear interpolation.
            mask (bool): If True (default), then the map values outside
                the cube will be undef. Otherwise, map will be kept as is.
            snapxy (bool): If True (optional), then the map values will get
                values at nearest Cube XY location. Only relevant to use if
                surface is derived from seismic coordinates (e.g. Auto4D).
            deadtraces (bool): If True (default) then dead cube traces
                (given as value 2 in SEGY trace headers), are treated as
                undefined, and map will become undefined at dead trace location.
            algorithm (int): 1 for legacy method, 2 (default from 2.9) for
                new method available in xtgeo from version 2.9

        Example::

            >>> import xtgeo
            >>> cube = xtgeo.cube_from_file(cube_dir + "/ib_test_cube2.segy")
            >>> surf = xtgeo.surface_from_file(surface_dir + '/topreek_rota.gri')
            >>> # update surf to sample cube values:
            >>> surf.slice_cube(cube)

        Raises:
            Exception if maps have different definitions (topology)
            RuntimeWarning if number of sampled nodes is less than 10%

        .. versionchanged:: 2.9 Added ``algorithm`` keyword, default is 2
        """
        scube = surface_from_cube(cube, 0)
        ier = _regsurf_cube.slice_cube(
            self,
            cube,
            scube,
            zsurf=zsurf,
            sampling=sampling,
            mask=mask,
            snapxy=snapxy,
            deadtraces=deadtraces,
            algorithm=algorithm,
        )

        if ier == -4:
            xtg.warnuser("Number of sampled surface nodes < 10 percent of Cube nodes")
            print("Number of sampled surface nodes < 10 percent of Cube nodes")
        elif ier == -5:
            xtg.warn("No nodes sampled: map is 100 percent outside of cube?")

    def slice_cube_window(
        self,
        cube: Cube,
        zsurf: Optional[RegularSurface] = None,
        other: Optional[RegularSurface] = None,
        other_position: str = "below",
        sampling: Literal["nearest", "cube", "trilinear"] = "nearest",
        mask: bool = True,
        zrange: Optional[float] = None,
        ndiv: Optional[int] = None,
        attribute: Union[List[ValidAttrs], ValidAttrs] = "max",
        maskthreshold: float = 0.1,
        snapxy: bool = False,
        showprogress: bool = False,
        deadtraces: bool = True,
        algorithm: Literal[1, 2, 3] = 2,
    ) -> Optional[Dict[RegularSurface]]:
        """Slice the cube within a vertical window and get the statistical attrubutes.

        The statistical attributes can be min, max etc. Attributes are:

        * 'max' for maximum

        * 'min' for minimum

        * 'rms' for root mean square

        * 'mean' for expected value

        * 'var' for variance (population var; https://en.wikipedia.org/wiki/Variance)

        * 'maxpos' for maximum of positive values

        * 'maxneg' for negative maximum of negative values

        * 'maxabs' for maximum of absolute values

        * 'sumpos' for sum of positive values using cube sampling resolution

        * 'sumneg' for sum of negative values using cube sampling resolution

        * 'meanabs' for mean of absolute values

        * 'meanpos' for mean of positive values

        * 'meanneg' for mean of negative values

        Note that 'all' can be used to select all attributes that are currently
        available.

        Args:
            cube: Instance of a Cube() here
            zsurf: Instance of a depth (or time) map, which
                is the depth or time map (or...) that is used a slicer.
                If None, then the surface instance itself is used a slice
                criteria. Note that zsurf must have same map defs as the
                surface instance.
            other: Instance of other surface if window is
                between surfaces instead of a static window. The zrange
                input is then not applied.
            sampling: 'nearest'/'trilinear'/'cube' for nearest node (default),
                 or 'trilinear' for trilinear interpolation. The 'cube' option is
                 only available with algorithm = 2 and will overrule ndiv and sample
                 at the cube's Z increment resolution.
            mask: If True (default), then the map values outside
                the cube will be undef, otherwise map will be kept as-is
            zrange: The one-sided "radius" range of the window, e.g. 10
                (10 is default) units (e.g. meters if in depth mode).
                The full window is +- zrange (i.e. diameter).
                If other surface is present, zrange is computed based on those
                two surfaces instead.
            ndiv: Number of intervals for sampling within zrange. None
                means 'auto' sampling, using 0.5 of cube Z increment as basis. If
                algorithm = 2/3 and sampling is 'cube', the cube Z increment
                will be used.
            attribute: The requested attribute(s), e.g.
                'max' value. May also be a list of attributes, e.g.
                ['min', 'rms', 'max']. By such, a dict of surface objects is
                returned. Note 'all' will make a list of all possible attributes.
            maskthreshold (float): Only if two surface; if isochore is less
                than given value, the result will be masked.
            snapxy: If True (optional), then the map values will get
                values at nearest Cube XY location, and the resulting surfaces layout
                map will be defined by the seismic layout. Quite relevant to use if
                surface is derived from seismic coordinates (e.g. Auto4D), but can be
                useful in other cases also, as long as one notes that map definition
                may change from input.
            showprogress: If True, then a progress is printed to stdout.
            deadtraces: If True (default) then dead cube traces
                (given as value 2 in SEGY trace headers), are treated as
                undefined, and map will be undefined at dead trace location.
            algorithm: 1 for legacy method, 2 (default) for new faster
                and more precise method available from xtgeo version 2.9, and
                algorithm 3 as new implementation from Sept. 2023 (v3.4)

        Example::

            >>> import xtgeo
            >>> cube = xtgeo.cube_from_file(cube_dir + "/ib_test_cube2.segy")
            >>> surf = xtgeo.surface_from_file(surface_dir + '/topreek_rota.gri')
            >>> # update surf to sample cube values in a total range of 30 m:
            >>> surf.slice_cube_window(cube, attribute='min', zrange=15.0)

            >>> # Here a list is given instead:
            >>> alst = ['min', 'max', 'rms']

            >>> myattrs = surf.slice_cube_window(cube, attribute=alst, zrange=15.0)
            >>> for attr in myattrs.keys():
            ...     _ = myattrs[attr].to_file(
            ...         outdir + '/myfile_' + attr + '.gri'
            ...     )

        Raises:
            Exception if maps have different definitions (topology)
            ValueError if attribute is invalid.

        Returns:
            If `attribute` is a string, then the instance is updated and
            None is returned. If `attribute` is a list, then a dictionary
            of surface objects is returned.

        Note:
            This method is now deprecated and will be removed in xtgeo version 5.
            Please replace with :meth:`Cube().compute_attributes_in_window()` as soon
            as possible.


        .. versionchanged:: 2.9 Added ``algorithm`` keyword, default is now 2,
                            while 1 is the legacy version

        .. versionchanged:: 3.4 Added ``algorithm`` 3 which is more robust and
                            hence recommended!

        .. versionchanged:: 4.1 Flagged as deprecated.

        """

        warnings.warn(
            "This method is deprecated and will be removed in xtgeo version 5. "
            "It is strongly recommended to use `Cube().compute_attributes_in_window()` "
            "instead!",
            FutureWarning,
        )

        if other is None and zrange is None:
            zrange = 10

        if algorithm != 3:
            warnings.warn(
                "Other algorithms than no. 3 is not recommended, and will be "
                "removed in near future.",
                DeprecationWarning,
            )

        scube = surface_from_cube(cube, 0)
        return _regsurf_cube_window.slice_cube_window(
            self,
            cube,
            scube,
            zsurf=zsurf,
            other=other,
            other_position=other_position,
            sampling=sampling,
            mask=mask,
            zrange=zrange,
            ndiv=ndiv,
            attribute=attribute,
            maskthreshold=maskthreshold,
            snapxy=snapxy,
            showprogress=showprogress,
            deadtraces=deadtraces,
            algorithm=algorithm,
        )

    # ==================================================================================
    # Special methods
    # ==================================================================================

    def get_boundary_polygons(
        self,
        alpha_factor: Optional[float] = 1.0,
        convex: Optional[bool] = False,
        simplify: Optional[bool] = True,
    ):
        """Extract boundary polygons from the surface.

        A regular surface may often contain areas of undefined (masked) entries which
        makes the surface appear 'ragged' and/or 'patchy'.

        This method extracts boundaries around the surface patches, and the
        precision depends on the keyword settings. As default, the ``alpha_factor``
        of 1 makes a precise boundary, while a larger alpha_factor makes more rough
        polygons.

        .. image:: images/regsurf_boundary_polygons.png
           :width: 600
           :align: center

        |

        Args:
            alpha_factor: An alpha multiplier, where lowest allowed value is 1.0.
                A higher number will produce smoother and less accurate polygons. Not
                applied if convex is set to True.
            convex: The default is False, which means that a "concave hull" algorithm
                is used. If convex is True, the alpha factor is overridden to a large
                number, producing a 'convex' shape boundary instead.
            simplify: If True, a simplification is done in order to reduce the number
                of points in the polygons, where tolerance is 0.1. Another
                alternative to True is to input a Dict on the form
                ``{"tolerance": 2.0, "preserve_topology": True}``, cf. the
                :func:`Polygons.simplify()` method. For details on e.g. tolerance, see
                Shapely's simplify() method.

        Returns:
            A XTGeo Polygons instance

        Example::

            surf = xtgeo.surface_from_file("mytop.gri")
            # eliminate all values below a depth, e.g. a fluid contact
            surf.values = np.ma.masked_greater(surf.values, 2100.0)
            boundary = surf.get_boundary_polygons()
            # the boundary may contain several smaller polygons; keep only the
            # largest (first) polygon which is number 0:
            boundary.filter_byid([0])  # polygon is updated in-place

        See also:
            The :func:`Polygons.boundary_from_points()` class method.

        .. versionadded:: 3.1
        """
        return _regsurf_boundary.create_boundary(self, alpha_factor, convex, simplify)

    def get_fence(
        self, xyfence: np.ndarray, sampling: Optional[str] = "bilinear"
    ) -> np.ma.MaskedArray:
        """Sample the surface along X and Y positions (numpy arrays) and get Z.

        .. versionchanged:: 2.14 Added keyword option `sampling`

        Returns a masked numpy 2D array similar as input, but with updated
        Z values, which are masked if undefined.

        Args:
            xyfence: A 2D numpy array with shape (N, 3) where columns
                are (X, Y, Z). The Z will be updated to the map.
            sampling: Use "bilinear" (default) for interpolation or "nearest" for
                snapping to nearest node.

        """
        return _regsurf_oper.get_fence(self, xyfence, sampling=sampling)

    def get_randomline(
        self,
        fencespec: Union[np.ndarray, object],
        hincrement: Optional[Union[bool, float]] = None,
        atleast: Optional[int] = 5,
        nextend: Optional[int] = 2,
        sampling: Optional[str] = "bilinear",
    ) -> np.ndarray:
        """Extract a line along a fencespec.

        .. versionadded:: 2.1
        .. versionchanged:: 2.14 Added keyword option `sampling`

        Here, horizontal axis is "length" and vertical axis is sampled depth, and
        this is used for fence plots.

        The input fencespec is either a 2D numpy where each row is X, Y, Z, HLEN,
        where X, Y are UTM coordinates, Z is depth/time, and HLEN is a
        length along the fence, or a Polygons instance.

        If input fencspec is a numpy 2D, it is important that the HLEN array
        has a constant increment and ideally a sampling that is less than the
        map resolution. If a Polygons() instance, this is automated if hincrement is
        None, and ignored if hincrement is False.

        Returns a ndarray with shape (:, 2).

        Args:
            fencespec:
                2D numpy with X, Y, Z, HLEN as rows or a xtgeo Polygons() object.
            hincrement: Resampling horizontally. This applies only
                if the fencespec is a Polygons() instance. If None (default),
                the distance will be deduced automatically. If False, then it assumes
                the Polygons can be used as-is.
            atleast: Minimum number of horizontal samples (only if
                fencespec is a Polygons instance and hincrement != False)
            nextend: Extend with nextend * hincrement in both ends (only if
                fencespec is a Polygons instance and hincrement != False)
            sampling: Use "bilinear" (default) for interpolation or "nearest" for
                snapping to nearest node.


        Example::

            fence = xtgeo.polygons_from_file("somefile.pol")
            fspec = fence.get_fence(distance=20, nextend=5, asnumpy=True)
            surf = xtgeo.surface_from_file("somefile.gri")

            arr = surf.get_randomline(fspec)

            distance = arr[:, 0]
            zval = arr[:, 1]
            # matplotlib...
            plt.plot(distance, zval)

        .. seealso::
           Class :class:`~xtgeo.xyz.polygons.Polygons`
              The method :meth:`~xtgeo.xyz.polygons.Polygons.get_fence()` which can be
              used to pregenerate `fencespec`
        """
        return _regsurf_oper.get_randomline(
            self,
            fencespec,
            hincrement=hincrement,
            atleast=atleast,
            nextend=nextend,
            sampling=sampling,
        )

    def hc_thickness_from_3dprops(
        self,
        xprop=None,
        yprop=None,
        hcpfzprop=None,
        zoneprop=None,
        zone_minmax=None,
        dzprop=None,
        zone_avg=False,
        coarsen=1,
        mask_outside=False,
    ):
        """Make a thickness weighted HC thickness map.

        Make a HC thickness map based on numpy arrays of properties
        from a 3D grid. The numpy arrays here shall be ndarray,
        not masked numpies (MaskedArray).

        Note that the input hcpfzprop is hydrocarbon fraction multiplied
        with thickness, which can be achieved by e.g.:
        cpfz = dz*poro*ntg*shc or by hcpfz = dz*hcpv/vbulk

        Args:
            xprop (ndarray): 3D numpy array of X coordinates
            yprop (ndarray): 3D numpy array of Y coordinates
            hcpfzprop (ndarray): 3D numpy array of HC fraction multiplied
                with DZ per cell.
            zoneprop (ndarray): 3D numpy array indicating zonation
                property, where 1 is the lowest (0 values can be used to
                exclude parts of the grid)
            dzprop (ndarray): 3D numpy array holding DZ thickness. Will
                be applied in weighting if zone_avg is active.
            zone_minmax (tuple): (optional) 2 element list indicating start
                and stop zonation (both start and end spec are included)
            zone_avg (bool): A zone averaging is done prior to map gridding.
                This may speed up the process a lot, but result will be less
                precise. Default is False.
            coarsen (int): Select every N'th X Y point in the gridding. Will
                speed up process, but less precise result. Default=1
            mask_outside (bool): Will mask the result map undef where sum of DZ
                is zero. Default is False as it costs some extra CPU.

        Returns:
            True if operation went OK (but check result!), False if not
        """
        for inum, myprop in enumerate([xprop, yprop, hcpfzprop, zoneprop]):
            if isinstance(myprop, ma.MaskedArray):
                raise ValueError(
                    f"Property input {inum} with avg {myprop.mean()} to {__name__} "
                    "is a masked array, not a plain numpy ndarray"
                )

        status = _regsurf_gridding.avgsum_from_3dprops_gridding(
            self,
            summing=True,
            xprop=xprop,
            yprop=yprop,
            mprop=hcpfzprop,
            dzprop=dzprop,
            zoneprop=zoneprop,
            zone_minmax=zone_minmax,
            zone_avg=zone_avg,
            coarsen=coarsen,
            mask_outside=mask_outside,
        )

        if status is False:
            raise RuntimeError("Failure from hc thickness calculation")

    def avg_from_3dprop(
        self,
        xprop=None,
        yprop=None,
        mprop=None,
        dzprop=None,
        truncate_le=None,
        zoneprop=None,
        zone_minmax=None,
        coarsen=1,
        zone_avg=False,
    ):
        """Average map (DZ weighted) based on numpy arrays of properties from a 3D grid.

        The 3D arrays mush be ordinary numpies of size (nx,ny,nz). Undef
        entries must be given weights 0 by using DZ=0

        Args:
            xprop: 3D numpy of all X coordinates (also inactive cells)
            yprop: 3D numpy of all Y coordinates (also inactive cells)
            mprop: 3D numpy of requested property (e.g. porosity) all
            dzprop: 3D numpy of dz values (for weighting)
                NB zero for undef cells
            truncate_le (float): Optional. Truncate value (mask) if
                value is less
            zoneprop: 3D numpy to a zone property
            zone_minmax: a tuple with from-to zones to combine
                (e.g. (1,3))

        Returns:
            Nothing explicit, but updates the surface object.
        """
        for inum, myprop in enumerate([xprop, yprop, mprop, dzprop, zoneprop]):
            if isinstance(myprop, ma.MaskedArray):
                raise ValueError(
                    f"Property input {inum} with avg {myprop.mean()} to {__name__} "
                    "is a masked array, not a plain numpy ndarray"
                )

        _regsurf_gridding.avgsum_from_3dprops_gridding(
            self,
            summing=False,
            xprop=xprop,
            yprop=yprop,
            mprop=mprop,
            dzprop=dzprop,
            truncate_le=truncate_le,
            zoneprop=zoneprop,
            zone_minmax=zone_minmax,
            coarsen=coarsen,
            zone_avg=zone_avg,
        )

    def quickplot(
        self,
        filename=None,
        title="QuickPlot for Surfaces",
        subtitle=None,
        infotext=None,
        minmax=(None, None),
        xlabelrotation=None,
        colormap="rainbow",
        faults=None,
        logarithmic=False,
    ):
        """Fast surface plot of maps using matplotlib.

        Args:
            filename (str): Name of plot file; None will plot to screen.
            title (str): Title of plot
            subtitle (str): Subtitle of plot
            infotext (str): Additonal info on plot.
            minmax (tuple): Tuple of min and max values to be plotted. Note
                that values outside range will be set equal to range limits
            xlabelrotation (float): Rotation in degrees of X labels.
            colormap (str): Name of matplotlib or RMS file or XTGeo
                colormap. Default is matplotlib's 'rainbow'
            faults (dict): If fault plot is wanted, a dictionary on the
                form => {'faults': XTGeo Polygons object, 'color': 'k'}
            logarithmic (bool): If True, a logarithmic contouring color scale
                will be used.

        """
        # This is using the more versatile Map class in XTGeo. Most kwargs
        # is just passed as is. Prefer using Map() directly in apps?

        ncount = self.values.count()
        if ncount < 5:
            xtg.warn(f"None or too few map nodes for plotting. Skip output {filename}!")
            return

        import xtgeoviz.plot

        mymap = xtgeoviz.plot.Map()

        logger.info("Infotext is <%s>", infotext)
        mymap.canvas(title=title, subtitle=subtitle, infotext=infotext)

        minvalue = minmax[0]
        maxvalue = minmax[1]

        mymap.colormap = colormap

        mymap.plot_surface(
            self,
            minvalue=minvalue,
            maxvalue=maxvalue,
            xlabelrotation=xlabelrotation,
            logarithmic=logarithmic,
        )
        if faults:
            poly = faults.pop("faults")
            mymap.plot_faults(poly, **faults)

        if filename is None:
            mymap.show()
        else:
            mymap.savefig(filename)

    def distance_from_point(self, point=(0, 0), azimuth=0.0):
        """Make map values as horizontal distance from a point with azimuth direction.

        Args:
            point (tuple): Point to measure from
            azimuth (float): Angle from North (clockwise) in degrees

        """
        _regsurf_oper.distance_from_point(self, point=point, azimuth=azimuth)

    def translate_coordinates(self, translate=(0, 0, 0)):
        """Translate a map in X Y VALUE space.

        Args:
            translate (tuple): Translate (shift) distance in X Y Z

        Example::

            >>> import xtgeo
            >>> mysurf = xtgeo.surface_from_file(surface_dir + '/topreek_rota.gri')
            >>> print(mysurf.xori, mysurf.yori)
            468895.125 5932889.5
            >>> mysurf.translate_coordinates((300,500,0))
            >>> print(mysurf.xori, mysurf.yori)
            469195.125 5933389.5

        """
        xshift, yshift, zshift = translate

        # just shift the xori and yori
        self.xori = self.xori + xshift
        self.yori = self.yori + yshift

        # note the Z coordinates are perhaps not depth
        # numpy operation:
        self.values = self.values + zshift

    # ==================================================================================
    # Private
    # ==================================================================================

    def _ensure_correct_values(
        self,
        values,
        force_dtype=None,
    ):
        """Ensures that values is a 2D masked numpy (ncol, nrow), C order.

        This is an improved but private version over ensure_correct_values

        Args:
            values (array-like or scalar): Values to process.
            force_dtype (numpy dtype or None): If not None, try to derive dtype from
                current values

        Return:
            Nothing, self._values will be updated inplace

        """
        apply_dtype = force_dtype if force_dtype else self.dtype

        currentmask = None
        if isinstance(self.values, ma.MaskedArray):
            currentmask = ma.getmaskarray(self.values)

        if isinstance(values, ma.MaskedArray):
            newmask = ma.getmaskarray(values)
            vals = values.astype(np.float64)
            vals = ma.masked_greater(vals, self.undef_limit)
            vals = ma.masked_invalid(vals)
            if (
                currentmask is not None
                and np.array_equal(currentmask, newmask)
                and self.values.shape == values.shape
                and values.flags.c_contiguous is True
            ):
                self._values *= 0
                self._values += vals
            else:
                vals = vals.reshape((self._ncol, self._nrow))
                if not vals.flags.c_contiguous:
                    mask = ma.getmaskarray(values)
                    mask = np.asanyarray(mask, order="C")
                    vals = np.asanyarray(vals, order="C")
                    vals = ma.array(vals, mask=mask, order="C")
                self._values = vals

        elif isinstance(values, numbers.Number):
            if currentmask is not None:
                vals = np.ones(self.dimensions, dtype=apply_dtype) * values
                vals = np.ma.array(vals, mask=currentmask)

                # there maybe cases where values scalar input is some kind of UNDEF
                # which will change the mask
                vals = ma.masked_greater(vals, self.undef_limit, copy=False)
                vals = ma.masked_invalid(vals, copy=False)
                self._values *= 0
                self._values += vals
            else:
                vals = ma.zeros((self.ncol, self.nrow), order="C", dtype=apply_dtype)
                self._values = vals + float(values)

        elif isinstance(values, (list, tuple, np.ndarray)):  # ie values ~ list-like
            vals = ma.array(values, order="C", dtype=apply_dtype)
            vals = ma.masked_greater(vals, self.undef_limit, copy=True)
            vals = ma.masked_invalid(vals, copy=True)

            if vals.shape != (self.ncol, self.nrow):
                try:
                    vals = ma.reshape(vals, (self.ncol, self.nrow), order="C")
                except ValueError as emsg:
                    raise ValueError(f"Cannot reshape array: {values}") from emsg

            self._values = vals

        else:
            raise ValueError(f"Input values are in invalid format: {values}")

        if self._values.mask is ma.nomask:
            self._values = ma.array(self._values, mask=ma.getmaskarray(self._values))

        # ensure dtype; avoid allocation and ID change if possible by setting copy=False
        self._values = self._values.astype(apply_dtype, copy=False)
