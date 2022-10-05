# -*- coding: utf-8 -*-
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

# pylint: disable=too-many-public-methods

import functools
import io
import math
import numbers
import pathlib
import warnings
from collections import OrderedDict
from copy import deepcopy
from types import FunctionType
from typing import List, Optional, Tuple, Union

import deprecation
import numpy as np
import numpy.ma as ma
import pandas as pd
import xtgeo
import xtgeo.common.sys as xtgeosys
from xtgeo.common.constants import VERYLARGENEGATIVE, VERYLARGEPOSITIVE

from . import (
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

xtg = xtgeo.common.XTGeoDialog()
logger = xtg.functionlogger(__name__)


# ======================================================================================
# METHODS as wrappers to class init + import


def surface_from_file(mfile, fformat=None, template=None, values=True, engine="cxtgeo"):
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

    Example::

        >>> import xtgeo
        >>> mysurf = xtgeo.surface_from_file(surface_dir + '/topreek_rota.gri')

    .. versionchanged:: 2.1
      Key "values" for Irap binary maps added

    .. versionchanged:: 2.13 Key "engine" added
    """

    return RegularSurface._read_file(
        mfile, fformat=fformat, load_values=values, engine=engine, template=template
    )


def surface_from_roxar(project, name, category, stype="horizons", realisation=0):
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
        project, name, category, stype=stype, realisation=realisation
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


def surface_from_grid3d(grid, template=None, where="top", mode="depth", rfactor=1):
    """This makes 3 instances of a RegularSurface directly from a Grid() instance.

    Args:
        grid (Grid): XTGeo Grid instance
        template(RegularSurface): Optional to use an existing surface as
            template for geometry
        where (str): "top", "base" or use the syntax "2_top" where 2
            is layer no. 2 and _top indicates top of cell, while "_base"
            indicates base of cell
        mode (str): "depth", "i" or "j"
        rfactor (float): Determines how fine the extracted map is; higher values
            for finer map (but computing time will increase). Will only apply if
            template is None.

    .. versionadded:: 2.1
    """
    return RegularSurface._read_grid3d(
        grid, template=template, where=where, mode=mode, rfactor=rfactor
    )


def _data_reader_factory(file_format):
    if file_format == "irap_binary":
        return _regsurf_import.import_irap_binary
    if file_format == "irap_ascii":
        return _regsurf_import.import_irap_ascii
    if file_format == "ijxyz":
        return _regsurf_import.import_ijxyz
    if file_format == "petromod":
        return _regsurf_import.import_petromod
    if file_format == "zmap_ascii":
        return _regsurf_import.import_zmap_ascii
    if file_format == "xtg":
        return _regsurf_import.import_xtg
    if file_format == "hdf":
        return _regsurf_import.import_hdf5_regsurf
    raise ValueError(f"Unknown file format {file_format}")


def allow_deprecated_init(func):
    # This decorator is here to maintain backwards compatibility in the construction
    # of RegularSurface and should be deleted once the deprecation period has expired,
    # the construction will then follow the new pattern.
    @functools.wraps(func)
    def wrapper(cls, *args, **kwargs):
        # Checking if we are doing an initialization
        # from file and raise a deprecation warning if
        # we are.
        if "sfile" in kwargs or len(args) == 1:
            warnings.warn(
                "Initializing directly from file name is deprecated and will be "
                "removed in xtgeo version 4.0. Use: "
                "mysurf = xtgeo.surface_from_file('some_name.gri') instead",
                DeprecationWarning,
            )
            sfile = kwargs.get("sfile", args[0])
            fformat = kwargs.get("fformat", None)
            values = kwargs.get("values", None)
            if isinstance(values, bool) and values is False:
                load_values = False
            else:
                load_values = True
            mfile = xtgeosys._XTGeoFile(sfile)
            if fformat is None or fformat == "guess":
                fformat = mfile.detect_fformat()
            else:
                fformat = mfile.generic_format_by_proposal(fformat)  # default
            kwargs = _data_reader_factory(fformat)(mfile, values=load_values)
            kwargs["filesrc"] = mfile.file
            kwargs["fformat"] = fformat
            return func(cls, **kwargs)

        if "nx" in kwargs:
            warnings.warn(
                (
                    "nx is deprecated and will be removed "
                    "in xtgeo version 3.0. Use ncol instead"
                ),
                DeprecationWarning,
            )
            kwargs["ncol"] = kwargs["nx"]
            kwargs.pop("nx")
        if "ny" in kwargs:
            warnings.warn(
                (
                    "ny is deprecated and will be removed "
                    "in xtgeo version 3.0. Use nrow instead"
                ),
                DeprecationWarning,
            )
            kwargs["nrow"] = kwargs["ny"]
            kwargs.pop("ny")
        return func(cls, *args, **kwargs)

    return wrapper


def allow_deprecated_default_init(func):
    # This decorator is here to maintain backwards compatibility in the construction
    # of RegularSurface and should be deleted once the deprecation period has expired,
    # the construction will then follow the new pattern.
    @functools.wraps(func)
    def wrapper(cls, *args, **kwargs):
        # This is (mostly) for cases where we are doing an empty
        # initialization, so we need to inject default values
        # for the required args. The excessive checking is in
        # corner cases where we provide some positional arguments
        # as keyword arguments.
        _deprecation_msg = (
            "{} is a required argument, will no "
            "longer be defaulted in xtgeo version 4.0"
        )
        if len(args) != 4:
            if "ncol" not in kwargs and len(args) != 1:
                warnings.warn(_deprecation_msg.format("ncol"), DeprecationWarning)
                kwargs["ncol"] = 5
            if "nrow" not in kwargs and len(args) != 2:
                warnings.warn(_deprecation_msg.format("nrow"), DeprecationWarning)
                kwargs["nrow"] = 3
            if "xinc" not in kwargs and len(args) != 3:
                warnings.warn(_deprecation_msg.format("xinc"), DeprecationWarning)
                kwargs["xinc"] = 25.0
            if "yinc" not in kwargs:
                warnings.warn(_deprecation_msg.format("yinc"), DeprecationWarning)
                kwargs["yinc"] = 25.0
            default = (
                kwargs.get("ncol", 5) == 5
                and kwargs.get("nrow", 3) == 3
                and kwargs.get("xori", 0.0) == kwargs.get("yori", 0.0) == 0.0
                and kwargs.get("xinc", 25.0) == kwargs.get("yinc", 25.0) == 25.0
            )
            values = kwargs.get("values", None)
            if values is None and default:
                default_values = [
                    [1, 6, 11],
                    [2, 7, 12],
                    [3, 8, 1e33],
                    [4, 9, 14],
                    [5, 10, 15],
                ]
                warnings.warn(
                    f"Default values {default_values} for RegularSurface is "
                    f"deprecated and will be set to an array of zero if not explicitly "
                    f"given in version 3",
                    DeprecationWarning,
                )
                # make default surface (mostly for unit testing)
                kwargs["values"] = np.array(
                    default_values,
                    dtype=np.float64,
                    order="C",
                )
        return func(cls, *args, **kwargs)

    return wrapper


class RegularSurface:
    """Class for a regular surface in the XTGeo framework.

    The values can as default be accessed by the user as a 2D masked numpy
    (ncol, nrow) float64 array, but also other representations or views are
    possible (e.g. as 1D ordinary numpy).

    """

    @allow_deprecated_init
    @allow_deprecated_default_init
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
        undef: Optional[float] = xtgeo.UNDEF,
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
        self._undef_limit = xtgeo.UNDEF_LIMIT

        self._filesrc = filesrc  # Name of original input file or stream, if any

        self._fformat = fformat  # current fileformat, useful for load()
        self._metadata = xtgeo.MetaDataRegularSurface()

        self._values = None
        if values is None:
            values = np.ma.zeros((self._ncol, self._nrow))
            self._isloaded = False
        else:
            self._isloaded = True
        self.values = values

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
        mfile = xtgeosys._XTGeoFile(mfile)
        args = _data_reader_factory("zmap_ascii")(mfile, values=values)
        return cls(**args)

    def __repr__(self):
        """Magic method __repr__."""
        myrp = (
            "{0.__class__.__name__} (xori={0._xori!r}, yori={0._yori!r}, "
            "xinc={0._xinc!r}, yinc={0._yinc!r}, ncol={0._ncol!r}, "
            "nrow={0._nrow!r}, rotation={0._rotation!r}, "
            "yflip={0._yflip!r}, masked={0._masked!r}, "
            "filesrc={0._filesrc!r}, name={0._name!r}, "
            "ilines={0.ilines.shape!r}, xlines={0.xlines.shape!r}, "
            "values={0.values.shape!r}) ID={1}.".format(self, id(self))
        )
        return myrp

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

    @deprecation.deprecated(
        deprecated_in="1.16",
        removed_in="3.0",
        current_version=xtgeo.version,
        details="method 'ensure_correct_values' is obsolete and will removed soon",
    )
    def ensure_correct_values(self, ncol, nrow, values):
        """Ensures that values is a 2D masked numpy (ncol, nrol), C order.

        This function is deprecated

        Args:
            ncol (int): Number of columns.
            nrow (int): Number of rows.
            values (array or scalar): Values to process.

        Return:
            values (MaskedArray): Array on correct format.

        Example::

            >>> mysurf = xtgeo.surface_from_file(surface_dir + '/topreek_rota.gri')
            >>> # a 1D numpy array in C order by default
            >>> vals = np.ones((mysurf.ncol*mysurf.nrow))

            >>> # secure that the values are masked, in correct format and shape:
            >>> mysurf.values = mysurf.ensure_correct_values(
            ...     mysurf.ncol,
            ...     mysurf.nrow,
            ...     vals
            ... )
        """

        if not self._isloaded:
            return None

        currentmask = None
        if self._values is not None:
            if isinstance(self._values, ma.MaskedArray):
                currentmask = ma.getmaskarray(self._values)

        if isinstance(values, numbers.Number):
            vals = ma.zeros((ncol, nrow), order="C", dtype=np.float64)
            vals = ma.array(vals, mask=currentmask)
            values = vals + float(values)

        if not isinstance(values, ma.MaskedArray):
            values = ma.array(values, order="C")

        if values.shape != (ncol, nrow):
            try:
                values = ma.reshape(values, (ncol, nrow), order="C")
            except ValueError as emsg:
                xtg.error("Cannot reshape array: {}".format(emsg))
                raise

        # replace any undef or nan with mask
        values = ma.masked_greater(values, self.undef_limit)
        values = ma.masked_invalid(values)

        if not values.flags.c_contiguous:
            mask = ma.getmaskarray(values)
            mask = np.asanyarray(mask, order="C")
            values = np.asanyarray(values, order="C")
            values = ma.array(values, mask=mask, order="C")

        return values

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
        if not isinstance(obj, xtgeo.MetaDataRegularSurface):
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
    @deprecation.deprecated(
        deprecated_in="1.6",
        removed_in="3.0",
        current_version=xtgeo.version,
        details="nx is deprecated; use ncol instead,",
    )
    def nx(self):  # pylint: disable=C0103
        """The NX (or N-Idir) number, as property (deprecated, use ncol)."""
        return self._ncol

    @property
    @deprecation.deprecated(
        deprecated_in="1.6",
        removed_in="3.0",
        current_version=xtgeo.version,
        details="ny is deprecated; use ncol instead,",
    )
    def ny(self):  # pylint: disable=C0103
        """The NY (or N-Jdir) number, as property (deprecated, use nrow)."""
        return self._nrow

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
    def values(self):
        """The map values, as 2D masked numpy (float64), shape (ncol, nrow).

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

        hash_ = xtgeosys.generic_hash(gid, hashmethod=hashmethod)
        return hash_

    def describe(self, flush=True):
        """Describe an instance by printing to stdout."""
        #
        dsc = xtgeo.common.XTGDescription()
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

    @deprecation.deprecated(
        deprecated_in="2.15",
        removed_in="4.0",
        current_version=xtgeo.version,
        details="Use xtgeo.surface_from_file() instead",
    )
    def from_file(
        self,
        mfile: Union[str, pathlib.Path, io.BytesIO],
        fformat: Optional[str] = None,
        values: Optional[bool] = True,
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
            values: If True (default), then full array is read, if False
                only metadata will be read. Valid for Irap binary only. This allows
                lazy loading in e.g. ensembles.
            kwargs: some readers allow additonal options:
            template: Only valid if ``ijxyz`` format, where an
                existing Cube or RegularSurface instance is applied to
                get correct topology.
            engine: Default is "cxtgeo" which use a C backend. Optionally a pure
                python "python" reader will be used, which in general is slower
                but may be safer when reading memory streams and/or threading.
                Engine is relevant for Irap binary, Irap ascii and zmap.

        Returns:
            Object instance.

        Example:
            Here the from_file method is used to initiate the object
            directly::

            >>> surf = RegularSurface().from_file(surface_dir + "/topreek_rota.gri")

        .. versionchanged:: 2.1
          Key "values" for Irap binary maps added

        .. versionchanged:: 2.2
          Input io.BytesIO instance instead of file is now possible

        .. versionchanged:: 2.13
            ZMAP + import is added, and io.BytesIO input is extended to more formats
        """
        logger.info("Import RegularSurface from file or memstream...")

        mfile = xtgeosys._XTGeoFile(mfile)

        if fformat is None or fformat == "guess":
            fformat = mfile.detect_fformat()
        else:
            fformat = mfile.generic_format_by_proposal(fformat)  # default

        kwargs = _data_reader_factory(fformat)(mfile, values=values, **kwargs)
        if values:
            self._isloaded = True
        self._reset(**kwargs)

    def _reset(self, **kwargs):
        self._ncol = kwargs["ncol"]
        self._nrow = kwargs["nrow"]
        self._xinc = kwargs["xinc"]
        self._yinc = kwargs["yinc"]
        self._xori = kwargs.get("xori", self._xori)
        self._yori = kwargs.get("yori", self._yori)
        self._yflip = kwargs.get("yflip", self._yflip)
        self._rotation = kwargs.get("rotation", self._rotation)
        self._ilines = kwargs.get("ilines", self._ilines)
        self._xlines = kwargs.get("xlines", self._xlines)
        self.values = kwargs.get("values", self._values)

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
        mfile = xtgeosys._XTGeoFile(mfile)
        mfile.check_file(raiseerror=ValueError)
        if fformat is None or fformat == "guess":
            fformat = mfile.detect_fformat()
        else:
            fformat = mfile.generic_format_by_proposal(fformat)  # default
        kwargs = _data_reader_factory(fformat)(mfile, values=load_values, **kwargs)
        kwargs["filesrc"] = mfile.file
        kwargs["fformat"] = fformat
        return cls(**kwargs)

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

                mfile = xtgeosys._XTGeoFile(self.filesrc)
                kwargs = _data_reader_factory(self._fformat)(mfile, values=True)
                self.values = kwargs.get("values", self._values)

            self._isloaded = True

    def to_file(
        self,
        mfile: Union[str, pathlib.Path, io.BytesIO],
        fformat: Optional[str] = "irap_binary",
        pmd_dataunits: Optional[Tuple[int, int]] = (15, 10),
        engine: Optional[str] = "cxtgeo",
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
        """
        logger.info("Export RegularSurface to file or memstream...")
        mfile = xtgeosys._XTGeoFile(mfile, mode="wb", obj=self)

        if not mfile.memstream:
            mfile.check_folder(raiseerror=OSError)
        else:
            engine = "python"

        if fformat in ("irap_ascii", "irapascii", "irap_txt", "irapasc"):
            _regsurf_export.export_irap_ascii(self, mfile, engine=engine)

        elif fformat in ("irap_binary", "irapbinary", "irapbin", "irap", "gri"):
            _regsurf_export.export_irap_binary(self, mfile, engine=engine)

        elif "zmap" in fformat:
            _regsurf_export.export_zmap_ascii(self, mfile, engine=engine)

        elif fformat == "storm_binary":
            _regsurf_export.export_storm_binary(self, mfile)

        elif fformat == "petromod":
            _regsurf_export.export_petromod_binary(self, mfile, pmd_dataunits)

        elif fformat == "ijxyz":
            _regsurf_export.export_ijxyz_ascii(self, mfile)

        # developing, in prep and experimental!
        elif fformat == "xtgregsurf":
            _regsurf_export.export_xtgregsurf(self, mfile)

        else:
            raise ValueError(f"Invalid file format: {fformat}")

        logger.info("Export RegularSurface to file or memstream... done")

        if mfile.memstream:
            return None
        return mfile.file

    @deprecation.deprecated(
        deprecated_in="2.15",
        removed_in="4.0",
        current_version=xtgeo.version,
        details="Use xtgeo.surface_from_hdf() instead",
    )
    def from_hdf(
        self,
        mfile: Union[str, pathlib.Path, io.BytesIO],
        values: Optional[bool] = True,
    ):
        """Import/load a surface (map) with metadata from a HDF5 file.

        Warning:
            This implementation is currently experimental and only recommended
            for testing.

        The file extension shall be '.hdf'.

        Args:
            mfile: File name or Path object or memory stream
            values: If False, only metadatadata are read

        Returns:
            RegularSurface() instance

        Example:
            >>> import xtgeo
            >>> surf = xtgeo.surface_from_file(surface_dir + '/topreek_rota.gri')
            >>> filepath = surf.to_hdf(outdir + "/topreek_rota.hdf")
            >>> mysurf = xtgeo.RegularSurface().from_hdf(filepath)

        .. versionadded:: 2.14
        """
        # developing, in prep and experimental!
        mfile = xtgeosys._XTGeoFile(mfile, mode="rb", obj=self)

        kwargs = _regsurf_import.import_hdf5_regsurf(mfile, values=values)

        self._reset(**kwargs)

        _self: self.__class__ = self
        return _self  # to make obj = xtgeo.RegularSurface().from_hdf(stream) work

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
        mfile = xtgeosys._XTGeoFile(mfile, mode="wb", obj=self)

        if not mfile.memstream:
            mfile.check_folder(raiseerror=OSError)

        _regsurf_export.export_hdf5_regsurf(self, mfile, compression=compression)
        return mfile.file

    @classmethod
    def _read_roxar(
        cls, project, name, category, stype="horizons", realisation=0
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

        """
        kwargs = _regsurf_roxapi.import_horizon_roxapi(
            project, name, category, stype, realisation
        )

        return cls(**kwargs)

    @deprecation.deprecated(
        deprecated_in="2.15",
        removed_in="4.0",
        current_version=xtgeo.version,
        details="Use xtgeo.surface_from_roxar() instead",
    )
    def from_roxar(
        self, project, name, category, stype="horizons", realisation=0
    ):  # pragma: no cover
        """Load a surface from a Roxar RMS project.

        The import from the RMS project can be done either within the project
        or outside the project.

        Note that a shortform to::

          import xtgeo
          mysurf = xtgeo.surface.RegularSurface()
          mysurf.from_roxar(project, 'TopAare', 'DepthSurface')

        is::

          import xtgeo
          mysurf = xtgeo.surface_from_roxar(project, 'TopAare', 'DepthSurface')

        Note also that horizon/zone name and category must exists in advance,
        otherwise an Exception will be raised.

        Args:
            project (str or special): Name of project (as folder) if
                outside RMS, og just use the magic project word if within RMS.
            name (str): Name of surface/map
            category (str): For horizons/zones or clipboard/general2d_data: for
                example 'DS_extracted'
            stype (str): RMS folder type, 'horizons' (default), 'zones', 'clipboard'
                or 'general2d_data'
            realisation (int): Realisation number, default is 0

        Returns:
            Object instance updated

        Raises:
            ValueError: Various types of invalid inputs.

        Example:
            Here the from_roxar method is used to initiate the object
            directly::

              mymap = RegularSurface()
              mymap.from_roxar(project, 'TopAare', 'DepthSurface')


        """
        kwargs = _regsurf_roxapi.import_horizon_roxapi(
            project, name, category, stype, realisation
        )

        self.metadata.required = self
        self._reset(**kwargs)

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

    @deprecation.deprecated(
        deprecated_in="2.15",
        removed_in="4.0",
        current_version=xtgeo.version,
        details="Use xtgeo.surface.surface_from_cube() instead",
    )
    def from_cube(self, cube, zlevel):
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

            >>> import xtgeo
            >>> mycube = xtgeo.cube_from_file(cube_dir + "/ib_test_cube2.segy")
            >>> mymap = xtgeo.RegularSurface()
            >>> mymap.from_cube(mycube, 2700)

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
        self._reset(**input_dict)

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
    def _read_grid3d(cls, grid, template=None, where="top", mode="depth", rfactor=1):
        """Extract a surface from a 3D grid.

        Args:
            grid (Grid): XTGeo Grid instance
            template (RegularSurface): Using an existing surface as template
            where (str): "top", "base" or use the syntax "2_top" where 2
                is layer no. 2 and _top indicates top of cell, while "_base"
                indicates base of cell
            mode (str): "depth", "i" or "j"
            rfactor (float): Determines how fine the extracted map is; higher values
                for finer map (but computing time will increase). Will only apply if
                template is None.

        Returns:
            Object instance

        Example::


            >>> import xtgeo
            >>> mygrid = xtgeo.grid_from_file(reek_dir + "/REEK.EGRID")
            >>> # make surface from top (default)
            >>> mymap = RegularSurface._read_grid3d(mygrid)

        .. versionadded:: 2.14

        """
        args, _, _ = _regsurf_grid3d.from_grid3d(
            grid, template=template, where=where, mode=mode, rfactor=rfactor
        )
        return cls(**args)

    @deprecation.deprecated(
        deprecated_in="2.15",
        removed_in="4.0",
        current_version=xtgeo.version,
        details="Use xtgeo.surface_from_grid3d() instead",
    )
    def from_grid3d(self, grid, template=None, where="top", mode="depth", rfactor=1):
        # It would perhaps to be natural to have this as a Grid() method also?
        """Extract a surface from a 3D grid.

        Args:
            grid (Grid): XTGeo Grid instance
            template(RegularSurface): Optional to use an existing surface as
                template for geometry
            where (str): "top", "base" or use the syntax "2_top" where 2
                is layer no. 2 and _top indicates top of cell, while "_base"
                indicates base of cell
            mode (str): "depth", "i" or "j"
            rfactor (float): Determines how fine the extracted map is; higher values
                for finer map (but computing time will increase). Will only apply if
                template is None.

        Returns:
            Object instance is updated in-place
            When mode="depth", two RegularSurface: icols and jrows are also returned.

        Example::

            >>> import xtgeo
            >>> mymap = RegularSurface()
            >>> mygrid = xtgeo.grid_from_file(reek_dir + "/REEK.EGRID")
            >>> # return two additonal maps
            >>> ic, jr = mymap.from_grid3d(mygrid)

        .. versionadded:: 2.1

        """
        args, ivalues, jvalues = _regsurf_grid3d.from_grid3d(
            grid, template=template, where=where, mode=mode, rfactor=rfactor
        )
        self._reset(**args)
        if ivalues is not None and jvalues is not None:
            ivals = self.copy()
            args["values"] = ivalues
            ivals._reset(**args)
            jvals = self.copy()
            args["values"] = jvalues
            jvals._reset(**args)
            return ivals, jvals
        else:
            return None

    def copy(self):
        """Deep copy of a RegularSurface object to another instance.

        Example::

            >>> mymap = xtgeo.surface_from_file(surface_dir + '/topreek_rota.gri')
            >>> mymapcopy = mymap.copy()

        """
        # pylint: disable=protected-access

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

        xsurf.ilines = self._ilines.copy()
        xsurf.xlines = self._xlines.copy()
        xsurf.filesrc = self._filesrc
        xsurf.metadata.required = xsurf

        return xsurf

    def get_values1d(
        self, order="C", asmasked=False, fill_value=xtgeo.UNDEF, activeonly=False
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
            val = ma.filled(val, fill_value=np.nan)
            val = np.array(val, order="F")
            val = ma.masked_invalid(val)

        val = val.ravel(order="K")

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

    @deprecation.deprecated(
        deprecated_in="2.0",
        removed_in="3.0",
        current_version=xtgeo.version,
        details=(
            "The get_zval() method is deprecated, use values.ravel() "
            "or get_values1d() instead"
        ),
    )
    def get_zval(self):
        """Get an an 1D, numpy array, F order of the map values (not masked).

        Note that undef values are very large numbers (see undef property).
        Also, this will reorder a 2D values array to column fastest, i.e.
        get stuff into Fortran order.

        This routine exists for historical reasons and prefer using
        property 'values', or alternatively get_values1d()
        instead (with order='F').
        """
        return self.get_values1d(order="F", asmasked=False, fill_value=self.undef)

    @deprecation.deprecated(
        deprecated_in="2.0",
        removed_in="3.0",
        current_version=xtgeo.version,
        details=(
            "The set_zval() method is deprecated, use values "
            "or set_values1d() instead"
        ),
    )
    def set_zval(self, vals):
        """Set a 1D (unmasked) numpy array (kept for historical reasons).

        The numpy array must be in Fortran order (i columns (ncol) fastest).

        This routine exists for historical reasons and prefer 'values or
        set_values1d instead (with option order='F').
        """
        self.set_values1d(vals, order="F")

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
        chklist = set(
            ["_ncol", "_nrow", "_xori", "_yori", "_xinc", "_yinc", "_rotation"]
        )
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
        zcoord = _regsurf_oper.get_value_from_xy(self, point=point, sampling=sampling)

        return zcoord

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

        entry = OrderedDict()

        if ijcolumns or ij:
            ixn, jyn = self.get_ij_values1d(order=order, activeonly=activeonly)
            entry["IX"] = ixn
            entry["JY"] = jyn

        entry.update([("X_UTME", xcoord), ("Y_UTMN", ycoord), ("VALUES", values)])

        dataframe = pd.DataFrame(entry)
        return dataframe

    dataframe = get_dataframe  # for compatibility backwards

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
                        xcv = float("{:{width}}".format(xcv, width=xyfmt))
                        ycv = float("{:{width}}".format(ycv, width=xyfmt))
                    if valuefmt is not None:
                        vcv = float("{:{width}}".format(vcv, width=valuefmt))
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

    def smooth(self, method="median", iterations=1, width=1):
        """Various smoothing methods for surfaces.

        Args:
            method: Smoothing method (median)
            iterations: Number of iterations
            width: Range of influence (in nodes)

        .. versionadded:: 2.1
        """
        if method == "median":
            _regsurf_gridding.smooth_median(self, iterations=iterations, width=width)
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

    def operation_polygons(self, poly, value, opname="add", inside=True):
        """A generic function for map operations inside or outside polygon(s).

        Args:
            poly (Polygons): A XTGeo Polygons instance
            value(float or RegularSurface): Value to add, subtract etc
            opname (str): Name of operation... 'add', 'sub', etc
            inside (bool): If True do operation inside polygons; else outside.
        """
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
        if not isinstance(points, xtgeo.xyz.Points):
            raise ValueError("Argument not a Points instance")

        logger.info("Do gridding...")

        _regsurf_gridding.points_gridding(self, points, coarsen=coarsen, method=method)

    # ==================================================================================
    # Interacion with other surface
    # ==================================================================================

    def resample(self, other, mask=True):
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

        """
        if not isinstance(other, RegularSurface):
            raise ValueError("Argument not a RegularSurface " "instance")

        logger.info("Do resampling...")

        _regsurf_oper.resample(self, other, mask=mask)

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
            raise ValueError("Unrotate refinement factor cannot be be " "less than 1")

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
                "Coarsen is too large, giving ncol or nrow less " "than 4 nodes"
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
        if not isinstance(grid, xtgeo.grid3d.Grid):
            raise ValueError("First argument must be a grid instance")

        ier = _regsurf_grid3d.slice_grid3d(
            self, grid, prop, zsurf=zsurf, sbuffer=sbuffer
        )

        if ier != 0:
            raise RuntimeError(
                "Wrong status from routine; something went " "wrong. Contact the author"
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
        ier = _regsurf_cube.slice_cube(
            self,
            cube,
            zsurf=zsurf,
            sampling=sampling,
            mask=mask,
            snapxy=snapxy,
            deadtraces=deadtraces,
            algorithm=algorithm,
        )

        if ier == -4:
            xtg.warnuser(
                "Number of sampled surface nodes < 10 percent of " "Cube nodes"
            )
            print("Number of sampled surface nodes < 10 percent of Cube nodes")
        elif ier == -5:
            xtg.warn("No nodes sampled: map is 100 percent outside of cube?")

    def slice_cube_window(
        self,
        cube,
        zsurf=None,
        other=None,
        other_position="below",
        sampling="nearest",
        mask=True,
        zrange=None,
        ndiv=None,
        attribute="max",
        maskthreshold=0.1,
        snapxy=False,
        showprogress=False,
        deadtraces=True,
        algorithm=2,
    ):
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
            cube (Cube): Instance of a Cube()
            zsurf (RegularSurface): Instance of a depth (or time) map, which
                is the depth or time map (or...) that is used a slicer.
                If None, then the surface instance itself is used a slice
                criteria. Note that zsurf must have same map defs as the
                surface instance.
            other (RegularSurface): Instance of other surface if window is
                between surfaces instead of a static window. The zrange
                input is then not applied.
            sampling (str): 'nearest'/'trilinear'/'cube' for nearest node (default),
                 or 'trilinear' for trilinear interpolation. The 'cube' option is
                 only available with algorithm = 2 and will overrule ndiv and sample
                 at the cube's Z increment resolution.
            mask (bool): If True (default), then the map values outside
                the cube will be undef, otherwise map will be kept as-is
            zrange (float): The one-sided "radius" range of the window, e.g. 10
                (10 is default) units (e.g. meters if in depth mode).
                The full window is +- zrange (i.e. diameter).
                If other surface is present, zrange is computed based on that.
            ndiv (int): Number of intervals for sampling within zrange. None
                means 'auto' sampling, using 0.5 of cube Z increment as basis. If
                algorithm = 2 and sampling is 'cube', the cube Z increment
                will be used.
            attribute (str or list): The requested attribute(s), e.g.
                'max' value. May also be a list of attributes, e.g.
                ['min', 'rms', 'max']. By such, a dict of surface objects is
                returned. Note 'all' will make a list of possible attributes
            maskthreshold (float): Only if two surface; if isochore is less
                than given value, the result will be masked.
            snapxy (bool): If True (optional), then the map values will get
                values at nearest Cube XY location. Only relevant to use if
                surface is derived from seismic coordinates (e.g. Auto4D).
            showprogress (bool): If True, then a progress is printed to stdout.
            deadtraces (bool): If True (default) then dead cube traces
                (given as value 2 in SEGY trace headers), are treated as
                undefined, nad map will be undefined at dead trace location.
            algorithm (int): 1 for legacy method, 2 (default) for new faster
                and more precise method available from xtgeo version 2.9.

        Example::

            >>> import xtgeo
            >>> cube = xtgeo.Cube(cube_dir + "/ib_test_cube2.segy")
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

        .. versionchanged:: 2.9 Added ``algorithm`` keyword, default is now 2,
                            while 1 is the legacy version
        """
        if other is None and zrange is None:
            zrange = 10

        asurfs = _regsurf_cube_window.slice_cube_window(
            self,
            cube,
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

        return asurfs

    # ==================================================================================
    # Special methods
    # ==================================================================================

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
        xyfence = _regsurf_oper.get_fence(self, xyfence, sampling=sampling)

        return xyfence

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

            fence = xtgeo.Polygons("somefile.pol")
            fspec = fence.get_fence(distance=20, nextend=5, asnumpy=True)
            surf = xtgeo.RegularSurface("somefile.gri")

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
        xyfence = _regsurf_oper.get_randomline(
            self,
            fencespec,
            hincrement=hincrement,
            atleast=atleast,
            nextend=nextend,
            sampling=sampling,
        )

        return xyfence

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
                    "Property input {} with avg {} to {} is a "
                    "masked array, not a plain numpy ndarray".format(
                        inum, myprop.mean(), __name__
                    )
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
                    "Property input {} with avg {} to {} is a "
                    "masked array, not a plain numpy ndarray".format(
                        inum, myprop.mean(), __name__
                    )
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
        colortable=None,
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
            colortable (str): Deprecated, for backward compatibility! used
                colormap instead.
            faults (dict): If fault plot is wanted, a dictionary on the
                form => {'faults': XTGeo Polygons object, 'color': 'k'}
            logarithmic (bool): If True, a logarithmic contouring color scale
                will be used.

        """
        # This is using the more versatile Map class in XTGeo. Most kwargs
        # is just passed as is. Prefer using Map() directly in apps?

        ncount = self.values.count()
        if ncount < 5:
            xtg.warn(
                "None or too few map nodes for plotting. Skip "
                "output {}!".format(filename)
            )
            return

        mymap = xtgeo.plot.Map()

        logger.info("Infotext is <%s>", infotext)
        mymap.canvas(title=title, subtitle=subtitle, infotext=infotext)

        minvalue = minmax[0]
        maxvalue = minmax[1]

        if colortable is not None:
            xtg.warndeprecated(
                "The colortable parameter is deprecated,"
                "and will be removed in version 4.0. Use colormap instead."
            )
            colormap = colortable

        mymap.colormap = colormap

        mymap.plot_surface(
            self,
            minvalue=minvalue,
            maxvalue=maxvalue,
            xlabelrotation=xlabelrotation,
            logarithmic=logarithmic,
        )
        if faults:
            mymap.plot_faults(faults["faults"])

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
        self, values
    ):  # pylint: disable=too-many-branches, too-many-statements
        """Ensures that values is a 2D masked numpy (ncol, nrow), C order.

        This is an improved but private version over ensure_correct_values

        Args:
            values (array-like or scalar): Values to process.

        Return:
            Nothing, self._values will be updated inplace

        """
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
                vals = np.ones(self.dimensions, dtype=np.float64) * values
                vals = np.ma.array(vals, mask=currentmask)

                # there maybe cases where values scalar input is some kind of UNDEF
                # which will change the mask
                vals = ma.masked_greater(vals, self.undef_limit, copy=False)
                vals = ma.masked_invalid(vals, copy=False)
                self._values *= 0
                self._values += vals
            else:
                vals = ma.zeros((self.ncol, self.nrow), order="C", dtype=np.float64)
                self._values = vals + float(values)

        elif isinstance(values, (list, tuple, np.ndarray)):  # ie values ~ list-like
            vals = ma.array(values, order="C", dtype=np.float64)
            vals = ma.masked_greater(vals, self.undef_limit, copy=True)
            vals = ma.masked_invalid(vals, copy=True)

            if vals.shape != (self.ncol, self.nrow):
                try:
                    vals = ma.reshape(vals, (self.ncol, self.nrow), order="C")
                except ValueError as emsg:
                    raise ValueError(f"Cannot reshape array: {values}") from emsg

            self._values = vals

        elif isinstance(values, (list, tuple)):  # ie values ~ list-like
            vals = ma.array(values, order="C", dtype=np.float64)
            vals = ma.masked_greater(vals, self.undef_limit, copy=True)
            vals = ma.masked_invalid(vals, copy=True)

            if vals.shape != (self.ncol, self.nrow):
                try:
                    vals = ma.reshape(vals, (self.ncol, self.nrow), order="C")
                except ValueError as emsg:
                    raise ValueError(f"Cannot reshape array: {values}") from emsg

            self._values = vals

        else:
            raise ValueError("Input values are in invalid format: {}".format(values))

        if self._values.mask is ma.nomask:
            self._values = ma.array(self._values, mask=ma.getmaskarray(self._values))
