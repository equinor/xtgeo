# -*- coding: utf-8 -*-
"""
Module/class for regular surfaces with XTGeo. Regular surfaces have a
constant distance between nodes (xinc, yinc), and this simplifies
computations a lot. A regular surface is defined by an origin (xori, yori)
in UTM, a number of columns (along X axis, if no rotation), a number of
rows (along Y axis if no rotation), and increment (distance between nodes).

The map itself is an array of values.

Rotation is allowed and is measured in degrees, anticlock from X axis.

Note that an instance of a regular surface can be made directly with::

 import xtgeo
 mysurf = xtgeo.surface_from_file('some_name.gri')

or::

 mysurf = xtgeo.surface_from_roxar('some_rms_project', 'TopX', 'DepthSurface')

"""
# -----------------------------------------------------------------------------
# Comment on 'asmasked' vs 'activeonly:
# 'asmasked'=True will return a np.ma array, with some fill_value if
# if asmasked = False
#
# while 'activeonly' will filter
# out maked entries, or use np.nan if 'activeonly' is False
#
# For functions with mask=... ,the should be replaced with asmasked=...
# -----------------------------------------------------------------------------

# pylint: disable=too-many-public-methods

from __future__ import print_function, absolute_import
from __future__ import division

import os
import os.path

from copy import deepcopy
import math
from types import FunctionType
import warnings

from collections import OrderedDict

import numpy as np
import numpy.ma as ma

import pandas as pd

import xtgeo
from xtgeo.common.constants import UNDEF, UNDEF_LIMIT
from xtgeo.common.constants import VERYLARGENEGATIVE, VERYLARGEPOSITIVE

from . import _regsurf_import
from . import _regsurf_export
from . import _regsurf_cube
from . import _regsurf_grid3d
from . import _regsurf_roxapi
from . import _regsurf_gridding
from . import _regsurf_oper
from . import _regsurf_utils

xtg = xtgeo.common.XTGeoDialog()
logger = xtg.functionlogger(__name__)

# =============================================================================
# METHODS as wrappers to class init + import


def surface_from_file(mfile, fformat=None, template=None):
    """Make an instance of a RegularSurface directly from file import.

    Args:
        mfile (str): Name of file
        fformat (str): See :meth:`RegularSurface.from_file`
        template (Cube or RegularSurface): See :meth:`RegularSurface.from_file`

    Example::

        import xtgeo
        mysurf = xtgeo.surface_from_file('some_name.gri')
    """

    obj = RegularSurface()

    obj.from_file(mfile, fformat=fformat, template=template)

    return obj


def surface_from_roxar(project, name, category, stype="horizons", realisation=0):
    """This makes an instance of a RegularSurface directly from roxar input.

    For arguments, see :meth:`RegularSurface.from_roxar`.

    Example::

        # inside RMS:
        import xtgeo
        mysurf = xtgeo.surface_from_roxar(project, 'TopEtive', 'DepthSurface')

    """

    obj = RegularSurface()

    obj.from_roxar(project, name, category, stype=stype, realisation=realisation)

    return obj


def surface_from_cube(cube, value):
    """This makes an instance of a RegularSurface directly from a cube instance
    with a constant value.

    The surface geometry will be exactly the same as for the Cube.

    Args:
        cube(xtgeo.cube.Cube): A Cube instance
        value (float): A constant value for the surface

    Example::

        mycube = xtgeo.cube_from_file('somefile.segy')
        mysurf = xtgeo.surface_from_cube(mycube, 1200)

    """

    obj = RegularSurface()

    obj.from_cube(cube, value)

    return obj


# =============================================================================
# RegularSurface class:


class RegularSurface(object):
    """Class for a regular surface in the XTGeo framework.

    The regular surface instance is usually initiated by
    import from file, but can also be made from scratch.
    The values can as default be accessed by the user as a 2D masked numpy
    (ncol, nrow) float64 array, but also other representations or views are
    possible (e.g. as 1D ordinary numpy).

    For construction:

    Args:
        ncol: Integer for number of X direction columns
        nrow: Integer for number of Y direction rows
        xori: X (East) origon coordinate
        yori: Y (North) origin coordinate
        xinc: X increment
        yinc: Y increment
        rotation: rotation in degrees, anticlock from X axis between 0, 360
        values: 2D (masked) numpy array of shape (ncol, nrow), C order
        name: A given name for the surface, default is file name root or
            'unkown' if constructed from scratch.

    Examples:

        The instance can be made either from file or by a spesification::

            >>> x1 = RegularSurface('somefilename')  # assume Irap binary
            >>> x2 = RegularSurface('somefilename', fformat='irap_ascii')
            >>> x3 = RegularSurface().from_file('somefilename',
                                                 fformat='irap_binary')
            >>> x4 = RegularSurface()
            >>> x4.from_file('somefilename', fformat='irap_binary')
            >>> x5 = RegularSurface(ncol=20, nrow=10, xori=2000.0, yori=2000.0,
                                    rotation=0.0, xinc=25.0, yinc=25.0,
                                    values=np.zeros((20,10)))

        Initiate a class and import::

          from xtgeo.surface import RegularSurface
          x = RegularSurface()
          x.from_file('some.irap', fformat='irap_binary')

    """

    def __init__(self, *args, **kwargs):  # pylint: disable=too-many-statements
        """Initiate a RegularSurface instance."""

        clsname = "{}.{}".format(type(self).__module__, type(self).__name__)
        logger.info(clsname)

        self._undef = UNDEF
        self._undef_limit = UNDEF_LIMIT
        self._masked = False
        self._filesrc = None  # Name of original input file, if any
        self._name = "unknown"

        # These are useful when import/export of surfaces with seismic origin
        self._ilines = None
        self._xlines = None

        # assume so far (1: meaning columns along X, to East, rows along Y,
        # North. If -1: rows along negative Y

        self._yflip = 1

        self._values = None

        if args:
            # make instance from file import
            mfile = args[0]
            fformat = kwargs.get("fformat", None)
            template = kwargs.get("template", None)
            self.from_file(mfile, fformat=fformat, template=template)

        else:
            # make instance by kw spesification
            self._xori = kwargs.get("xori", 0.0)
            self._yori = kwargs.get("yori", 0.0)
            if "nx" in kwargs:
                self._ncol = kwargs.get("nx", 5)  # backward compatibility
                self._nrow = kwargs.get("ny", 3)  # backward compatibility
            else:
                self._ncol = kwargs.get("ncol", 5)
                self._nrow = kwargs.get("nrow", 3)
            self._xinc = kwargs.get("xinc", 25.0)
            self._yinc = kwargs.get("yinc", 25.0)
            self._rotation = kwargs.get("rotation", 0.0)

            values = kwargs.get("values", None)

            if values is None:
                values = np.array(
                    [[1, 6, 11], [2, 7, 12], [3, 8, 1e33], [4, 9, 14], [5, 10, 15]],
                    dtype=np.float64,
                    order="C",
                )
                # make it masked
                values = ma.masked_greater(values, UNDEF_LIMIT)
                self._masked = True
                self._values = values
            else:
                self._values = self.ensure_correct_values(self.ncol, self.nrow, values)
            self._yflip = kwargs.get("yflip", 1)
            self._masked = kwargs.get("masked", True)
            self._filesrc = kwargs.get("filesrc", None)
            self._name = kwargs.get("name", "unknown")

        if self._values is not None:
            logger.debug("Shape of value: and values")
            logger.debug("\n%s", self._values.shape)
            logger.debug("\n%s", self._values)

            self._ilines = np.array(range(1, self._ncol + 1), dtype=np.int32)
            self._xlines = np.array(range(1, self._nrow + 1), dtype=np.int32)

            if self._values.mask is ma.nomask:
                logger.info("Ensure that mask is a full array")
                self._values = ma.array(
                    self._values, mask=ma.getmaskarray(self._values)
                )

        logger.debug("Ran __init__ method for RegularSurface object")

    # =========================================================================

    def __repr__(self):
        # should be able to newobject = eval(repr(thisobject))
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
        # user friendly print
        return self.describe(flush=False)

    def __add__(self, other):

        news = self.copy()
        news.add(other)

        return news

    def __sub__(self, other):

        news = self.copy()
        news.subtract(other)

        return news

    def __mul__(self, other):

        news = self.copy()
        news.multiply(other)

        return news

    # =========================================================================
    # Class and static methods
    # =========================================================================

    @classmethod
    def methods(cls):
        """Returns the names of the methods in the class.

        >>> print(RegularSurface.methods())
        """
        return [x for x, y in cls.__dict__.items() if isinstance(y, FunctionType)]

    def ensure_correct_values(self, ncol, nrow, values):
        """Ensures that values is a 2D masked numpy (ncol, nrol), C order.

        Args:
            ncol (int): Number of columns.
            nrow (int): Number of rows.
            values (array or scalar): Values to process.

        Return:
            values (MaskedArray): Array on correct format.

        Example::

            vals = np.ones((nx*ny))  # a 1D numpy array in C order by default

            # secure that the values are masked, in correct format and shape:
            mymap.values = mymap.ensure_correct_values(nc, nr, vals)
        """

        currentmask = None
        if self._values is not None:
            if isinstance(self._values, ma.MaskedArray):
                currentmask = ma.getmaskarray(self._values)

        if np.isscalar(values):
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
        values = ma.masked_greater(values, UNDEF_LIMIT)
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
    def ncol(self):
        """The NCOL (NX or N-Idir) number, as property (read only)."""
        return self._ncol

    @property
    def nrow(self):
        """The NROW (NY or N-Jdir) number, as property (read only)."""
        return self._nrow

    @property
    def nx(self):  # pylint: disable=C0103
        """The NX (or N-Idir) number, as property (deprecated, use ncol)."""
        warnings.warn("Deprecated; use ncol instead", DeprecationWarning)
        return self._ncol

    @property
    def ny(self):  # pylint: disable=C0103
        """The NY (or N-Jdir) number, as property (deprecated, use nrow)."""
        warnings.warn("Deprecated; use nrow instead", DeprecationWarning)
        return self._nrow

    @property
    def nactive(self):
        """Number of active map nodes (read only)."""
        return self._values.count()

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
        """

        if not self._values.flags.c_contiguous:
            logger.warning("Not C order in numpy")

        return self._values

    @values.setter
    def values(self, values):
        logger.debug("Enter method...")

        values = self.ensure_correct_values(self.ncol, self.nrow, values)

        self._values = values

        logger.debug("Values shape: %s", self._values.shape)
        logger.debug("Flags: %s", self._values.flags)

    @property
    def values1d(self):
        """(Read only) Map values, as 1D numpy masked, normally a numpy
        view(?).

        Example::

            map = RegularSurface('myfile.gri')
            values = map.values1d
        """
        return self.get_values1d(asmasked=True)

    @property
    def npvalues1d(self):
        """(Read only) Map values, as 1D numpy (not masked), undef as np.nan.

        In most cases this will be a copy of the values.

        Example::

            map = RegularSurface('myfile.gri')
            values = map.npvalues1d
            mean = np.nanmean(values)
            values[values <= 0] = np.nan
        """
        return self.get_values1d(asmasked=False, fill_value=np.nan)

    @property
    def name(self):
        """A free form name (str) e.g. for display, for the surface."""
        return self._name

    @name.setter
    def name(self, newname):
        if isinstance(newname, str):
            self._name = newname

    @property
    def undef(self):
        """Returns or set the undef value, to be used e.g. when in the
        get_zval method."""
        return self._undef

    @undef.setter
    def undef(self, undef):
        self._undef = undef

    @property
    def filesrc(self):
        """Gives the name of the file source (if any)"""
        return self._filesrc

    @filesrc.setter
    def filesrc(self, name):
        self._filesrc = name  # checking is currently missing

    @property
    def undef_limit(self):
        """Returns or set the undef_limit value, to be used when in the
        get_zval method."""
        return self._undef_limit

    @undef_limit.setter
    def undef_limit(self, undef_limit):
        self._undef_limit = undef_limit

    # =============================================================================
    # Describe, import and export
    # =============================================================================
    def describe(self, flush=True):
        """Describe an instance by printing to stdout"""

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
        dsc.txt("Values", self._values.reshape(-1), self._values.dtype)
        np.set_printoptions(threshold=1000)
        dsc.txt(
            "Values: mean, stdev, minimum, maximum",
            self.values.mean(),
            self.values.std(),
            self.values.min(),
            self.values.max(),
        )
        msize = float(self.values.size * 8) / (1024 * 1024 * 1024)
        dsc.txt("Minimum memory usage of array (GB)", msize)

        if flush:
            dsc.flush()
            return None

        return dsc.astext()

    def from_file(self, mfile, fformat=None, template=None):
        """Import surface (regular map) from file.

        Note that the fformat=None option will guess bye looking at the file
        extension, where e.g. "gri" will assume irap_binary and "fgr"
        assume Irap Ascii.

        The ijxyz format is the typical seismic format, on the form
        (ILINE, XLINE, X, Y, VALUE) as a table of points. Map values are
        estimated from the given values, or by using an existing map or
        cube as template, and match by ILINE/XLINE numbering.

        Args:
            mfile (str): Name of file
            fformat (str): File format, None/guess/irap_binary/irap_ascii/ijxyz
                is currently supported.
            template (object): Only valid if ``ijxyz`` format, where an
                existing Cube or RegularSurface instance is applied to
                get correct topology.

        Returns:
            Object instance.

        Example:
            Here the from_file method is used to initiate the object
            directly::

            >>> mymapobject = RegularSurface().from_file('myfile.x')

        """

        self._values = None

        if not os.path.isfile(mfile):
            msg = "Does file exist? {}".format(mfile)
            logger.critical(msg)
            raise IOError(msg)

        froot, fext = os.path.splitext(mfile)
        if fformat is None or fformat == "guess":
            if not fext:
                msg = (
                    'Stop: fformat is "guess" but file '
                    "extension is missing for {}".format(froot)
                )
                raise ValueError(msg)

            fformat = fext.lower().replace(".", "")

        if fformat in ("irap_binary", "gri", "bin", "irapbin"):
            _regsurf_import.import_irap_binary(self, mfile)
        elif fformat in ("irap_ascii", "fgr", "asc", "irapasc"):
            _regsurf_import.import_irap_ascii(self, mfile)
        elif fformat == "ijxyz":
            if template:
                _regsurf_import.import_ijxyz_ascii_tmpl(self, mfile, template)
            else:
                _regsurf_import.import_ijxyz_ascii(self, mfile)

        else:
            raise ValueError("Invalid file format: {}".format(fformat))

        self._name = os.path.basename(froot)
        self.ensure_correct_values(self.ncol, self.nrow, self._values)
        return self

    def to_file(self, mfile, fformat="irap_binary"):
        """Export a surface (map) to file.

        Note, for zmap_ascii and storm_binary an unrotation will be done
        automatically. The sampling will be somewhat finer than the
        original map in order to prevent aliasing. See :func:`unrotate`.

        Args:
            mfile (str): Name of file
            fformat (str): File format, irap_binary/irap_ascii/zmap_ascii/
                storm_binary/ijxyz. Default is irap_binary.

        Example::

            >>> x = RegularSurface()
            >>> x.from_file('myfile.x', fformat = 'irap_ascii')
            >>> x.values = x.values + 300
            >>> x.to_file('myfile2.x', fformat = 'irap_ascii')

        """

        logger.debug("Enter method...")
        logger.info("Export to file...")
        if fformat == "irap_ascii":
            _regsurf_export.export_irap_ascii(self, mfile)
        elif fformat == "irap_binary":
            _regsurf_export.export_irap_binary(self, mfile)
        elif fformat == "zmap_ascii":
            _regsurf_export.export_zmap_ascii(self, mfile)
        elif fformat == "storm_binary":
            _regsurf_export.export_storm_binary(self, mfile)
        elif fformat == "ijxyz":
            _regsurf_export.export_ijxyz_ascii(self, mfile)
        else:
            logger.critical("Invalid file format")

    def from_roxar(self, project, name, category, stype="horizons", realisation=0):
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
            category (str): For horizons/zones only: for example 'DS_extracted'
            stype (str): RMS folder type, 'horizons' (default) or 'zones'
            realisation (int): Realisation number, default is 0

        Returns:
            Object instance updated

        Raises:
            ValueError: Various types of invalid inputs.

        Example:
            Here the from_roxar method is used to initiate the object
            directly::

            >>> mymap = RegularSurface()
            >>> mymap.from_roxar(project, 'TopAare', 'DepthSurface')

        """

        stype = stype.lower()
        valid_stypes = ["horizons", "zones"]

        if stype not in valid_stypes:
            raise ValueError(
                "Invalid stype, only {} stypes is supported.".format(valid_stypes)
            )

        _regsurf_roxapi.import_horizon_roxapi(
            self, project, name, category, stype, realisation
        )

    def to_roxar(self, project, name, category, stype="horizons", realisation=0):
        """Store (export) a regular surface to a Roxar RMS project.

        The export to the RMS project can be done either within the project
        or outside the project. The storing is done to the Horizons or the
        Zones folder in RMS.

        Note that horizon or zone name and category must exists in advance,
        otherwise an Exception will be raised.

        Args:
            project (str or special): Name of project (as folder) if
                outside RMS, og just use the magic project word if within RMS.
            name (str): Name of surface/map
            category (str): For horizons/zones only: e.g. 'DS_extracted'.
            stype (str): RMS folder type, 'horizons' (default) or 'zones'
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

        """

        stype = stype.lower()
        valid_stypes = ["horizons", "zones"]

        if stype in valid_stypes and name is None or category is None:
            logger.error("Need to spesify name and category for " "horizon")
        elif stype not in valid_stypes:
            raise ValueError("Only {} stype is supported per now".format(valid_stypes))

        _regsurf_roxapi.export_horizon_roxapi(
            self, project, name, category, stype, realisation
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

            >>> mymap = RegularSurface()
            >>> mymap.from_roxar(project, 'TopAare', 'DepthSurface')

        """

        props = [
            "_ncol",
            "_nrow",
            "_xori",
            "_yori",
            "_xinc",
            "_yinc",
            "_rotation",
            "_ilines",
            "_xlines",
            "_yflip",
        ]

        for prop in props:
            setattr(self, prop, deepcopy(getattr(cube, prop)))

        self.values = ma.array(
            np.full((self._ncol, self._nrow), zlevel, dtype=np.float64)
        )

        self._filesrc = cube.filesrc + " (derived surface)"

    def copy(self):
        """Copy a xtgeo.surface.RegularSurface object to another instance::

            >>> mymapcopy = mymap.copy()

        """
        # pylint: disable=protected-access
        logger.debug("Copy object instance...")
        logger.debug(self._values)
        logger.debug(self._values.flags)
        logger.debug(id(self._values))

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

        if self._filesrc is not None and "(copy)" not in self._filesrc:
            xsurf.filesrc = self._filesrc + " (copy)"
        elif self._filesrc is not None:
            xsurf.filesrc = self._filesrc

        logger.debug("New array + flags + ID")
        return xsurf

    def get_values1d(
        self, order="C", asmasked=False, fill_value=UNDEF, activeonly=False
    ):
        """Get an an 1D, numpy or masked array of the map values.

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
            val = ma.array(val, order="F")
            val = ma.masked_invalid(val)

        val = val.ravel(order="K")

        if activeonly:
            val = val[~val.mask]

        if not asmasked and not activeonly:
            val = ma.filled(val, fill_value=fill_value)

        return val

    def set_values1d(self, val, order="C"):
        """Update the values attribute based on a 1D input, multiple options.

        If values are np.nan or values are > self.undef_limit, they will be
        masked.

        Args:
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

    def get_zval(self):
        """Get an an 1D, numpy array, F order of the map values (not masked).

        Note that undef values are very large numbers (see undef property).
        Also, this will reorder a 2D values array to column fastest, i.e.
        get stuff into Fortran order.

        This routine exists for historical reasons and prefer using
        property 'values', or alternatively get_values1d()
        instead (with order='F').
        """

        xtg.warndeprecated("Deprecated method (get_zval())")

        zval = self.get_values1d(order="F", asmasked=False, fill_value=self.undef)

        return zval

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
        """ Same as ncol (nx) (for backward compatibility) """
        return self._ncol

    def get_ny(self):
        """ Same as nrow (ny) (for backward compatibility) """
        return self._nrow

    def get_xori(self):
        """ Same as xori (for backward compatibility) """
        return self._xori

    def get_yori(self):
        """ Same as yori (for backward compatibility) """
        return self._yori

    def get_xinc(self):
        """ Same as xinc (for backward compatibility) """
        return self._xinc

    def get_yinc(self):
        """ Same as yinc (for backward compatibility) """
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
        if isinstance(mas1, np.ndarray) and isinstance(mas2, np.ndarray):
            if np.array_equal(mas1, mas2):
                pass
            else:
                logger.info("CMP mask %s %s", mas1, mas2)
                logger.warning("CMP mask %s %s", mas1, mas2)
                if strict:
                    return False

        logger.debug("Surfaces have same topology")
        return True

    def swapaxes(self):
        """Swap the axes columns vs rows, keep origin byt reverse yflip."""

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

        xc2 = self._xori + (self.nrow - 1) * math.cos(rot2) * self._yinc
        yc2 = self._yori + (self.nrow - 1) * math.sin(rot2) * self._yinc

        xc3 = xc2 + (self.ncol - 1) * math.cos(rot1) * self._xinc
        yc3 = yc2 + (self.ncol - 1) * math.sin(rot1) * self._xinc

        return ((xc0, yc0), (xc1, yc1), (xc2, yc2), (xc3, yc3))

    def get_value_from_xy(self, point=(0.0, 0.0)):
        """Return the map value given a X Y point.

        Args:
            point (float tuple): Position of X and Y coordinate
        Returns:
            The map value (interpolated). None if XY is outside defined map

        Example::
            mvalue = map.get_value_from_xy(point=(539291.12, 6788228.2))

        """
        zcoord = _regsurf_oper.get_value_from_xy(self, point=point)

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
            The z value at location iloc, jloc, None if undefined cell.
        """

        xval, yval, value = _regsurf_oper.get_xy_value_from_ij(
            self, iloc, jloc, zvalues=zvalues
        )

        return xval, yval, value

    def get_ij_values(self, zero_based=False, asmasked=False, order="C"):
        """Return I J numpy 2D arrays, optionally as masked arrays.

        Args:
            zero_base (bool): If False, first number is 1, not 0
            asmasked (bool): If True, UNDEF map nodes are skipped
            order (str): 'C' (default) or 'F' order (row vs column major)
        """

        return _regsurf_oper.get_ij_values(
            self, zero_based=zero_based, asmasked=asmasked, order=order
        )

    def get_ij_values1d(self, zero_based=False, activeonly=True, order="C"):
        """Return I J numpy as 1D arrays

        Args:
            zero_base (bool): If False, first number is 1, not 0
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
        """Return coordinates for X Y and Z (values) as numpy (masked)
        2D arrays.

        """

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
        """Return a Pandas dataframe object, with columns X_UTME,
        Y_UTMN, VALUES.

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

            surf = RegularSurface('myfile.gri')
            dfr = surf.dataframe()
            dfr.to_csv('somecsv.csv')

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
        logger.debug(dataframe)
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

            >>> mymap = RegularSurface('somefile.gri')
            >>> xylist, valuelist = mymap.get_xy_value_lists(valuefmt='6.2f')
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

    # =========================================================================
    # Operation on map values (list to be extended)
    # =========================================================================

    def operation(self, opname, value):
        """Do operation on map values.

        Do operations on the current map values. Valid operations are:

        * 'elilt' or 'eliminatelessthan': Eliminate less than <value>

        * 'elile' or 'eliminatelessequal': Eliminate less or equal than <value>

        Args:
            opname (str): Name of operation. See list above.
            values (*): A scalar number (float) or a tuple of two floats,
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

    # =========================================================================
    # Operations restricted to inside/outside polygons
    # =========================================================================

    def operation_polygons(self, poly, value, opname="add", inside=True):
        """A generic function for doing map operations restricted to inside
        or outside polygon(s).

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
        """Add a value (scalar or other map) inside polygons"""
        self.operation_polygons(poly, value, opname="add", inside=True)

    def add_outside(self, poly, value):
        """Add a value (scalar or other map) outside polygons"""
        self.operation_polygons(poly, value, opname="add", inside=False)

    def sub_inside(self, poly, value):
        """Subtract a value (scalar or other map) inside polygons"""
        self.operation_polygons(poly, value, opname="sub", inside=True)

    def sub_outside(self, poly, value):
        """Subtract a value (scalar or other map) outside polygons"""
        self.operation_polygons(poly, value, opname="sub", inside=False)

    def mul_inside(self, poly, value):
        """Multiply a value (scalar or other map) inside polygons"""
        self.operation_polygons(poly, value, opname="mul", inside=True)

    def mul_outside(self, poly, value):
        """Multiply a value (scalar or other map) outside polygons"""
        self.operation_polygons(poly, value, opname="mul", inside=False)

    def div_inside(self, poly, value):
        """Divide a value (scalar or other map) inside polygons"""
        self.operation_polygons(poly, value, opname="div", inside=True)

    def div_outside(self, poly, value):
        """Divide a value (scalar or other map) outside polygons"""
        self.operation_polygons(poly, value, opname="div", inside=False)

    def set_inside(self, poly, value):
        """Set a value (scalar or other map) inside polygons"""
        self.operation_polygons(poly, value, opname="set", inside=True)

    def set_outside(self, poly, value):
        """Set a value (scalar or other map) outside polygons"""
        self.operation_polygons(poly, value, opname="set", inside=False)

    def eli_inside(self, poly):
        """Eliminate current map values inside polygons"""
        self.operation_polygons(poly, 0, opname="eli", inside=True)

    def eli_outside(self, poly):
        """Eliminate current map values outside polygons"""
        self.operation_polygons(poly, 0, opname="eli", inside=False)

    # =========================================================================
    # Operation with secondary map
    # =========================================================================

    def add(self, other):
        """Add another map to current map"""

        _regsurf_oper.operations_two(self, other, oper="add")

    def subtract(self, other):
        """Subtract another map from current map"""

        _regsurf_oper.operations_two(self, other, oper="sub")

    def multiply(self, other):
        """Multiply another map and current map"""

        _regsurf_oper.operations_two(self, other, oper="mul")

    def divide(self, other):
        """Multiply another map and current map"""

        _regsurf_oper.operations_two(self, other, oper="div")

    # =========================================================================
    # Interacion with points
    # =========================================================================

    def gridding(self, points, method="linear", coarsen=1):
        """Grid a surface from points.

        Args:
            points(Points): XTGeo Points instance.
            method (str): Gridding method option: linear / cubic / nearest
            coarsen (int): Coarsen factor, to speed up gridding, but will
                give poorer result.

        Example::

            mypoints = Points('pfile.poi')
            mysurf = RegularSurface('top.gri')

            # update the surface by gridding the points
            mysurf.gridding(mypoints)

        Raises:
            RuntimeError: If not possible to grid for some reason
            ValueError: If invalid input

        """

        if not isinstance(points, xtgeo.xyz.Points):
            raise ValueError("Argument not a Points instance")

        logger.info("Do gridding...")

        _regsurf_gridding.points_gridding(self, points, coarsen=coarsen, method=method)

    # =========================================================================
    # Interacion with other surface
    # =========================================================================

    def resample(self, other):
        """Resample a surface from another surface instance.

        Note that there may be some 'loss' of nodes at the edges of the
        updated map, as only the 'inside' nodes in the updated map
        versus the input map are applied.

        Args:
            other (RegularSurface): Surface to resample from.
        """

        if not isinstance(other, RegularSurface):
            raise ValueError("Argument not a RegularSurface " "instance")

        logger.info("Do resampling...")

        _regsurf_oper.resample(self, other)

    # =========================================================================
    # Change a surface more fundamentally
    # =========================================================================

    def unrotate(self, factor=2):
        """Unrotete a map instance, and this will also change nrow, ncol,
        xinc, etc.

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

        if self._filesrc and "(resampled)" not in self._filesrc:
            self._filesrc = self._filesrc + " (resampled)"

    def refine(self, factor):
        """Refine a surface with a factor, e.g. 2 for double.

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

        if self._filesrc and "(refined)" not in self._filesrc:
            self._filesrc = self._filesrc + " (refined)"

        self.resample(proxy)

        del proxy
        logger.info("Do refining... DONE")

    def coarsen(self, factor):
        """Coarsen a surface with a factor, e.g. 2 meaning half the number of
        columns and rows.

        Range for coarsening is 2 to 10.

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

        if self._filesrc and "(coarsened)" not in self._filesrc:
            self._filesrc = self._filesrc + " (coarsened)"

        self.resample(proxy)

        del proxy
        logger.info("Do coarsening... DONE")

    # =========================================================================
    # Interacion with a grid3d
    # =========================================================================

    def slice_grid3d(self, grid, prop, zsurf=None, sbuffer=1):
        """Slice the grid property and update the instance surface to sampled
        values.

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

            grd = Grid('some.roff')
            prop = GridProperty('someprop.roff', name='PHIT')
            surf = RegularSurface('s.gri')
            # update surf to sample the 3D grid property:
            surf.slice_grid3d(grd, prop)

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

    # =========================================================================
    # Interacion with a cube
    # =========================================================================

    def slice_cube(
        self,
        cube,
        zsurf=None,
        sampling="nearest",
        mask=True,
        snapxy=False,
        deadtraces=True,
    ):
        """Slice the cube and update the instance surface to sampled cube
        values.

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
                the cube will be undef.
            snapxy (bool): If True (optional), then the map values will get
                values at nearest Cube XY location. Only relevant to use if
                surface is derived from seismic coordinates (e.g. Auto4D).
            deadtraces (bool): If True (default) then dead cube traces
                (given as value 2 in SEGY trace headers), are treated as
                undefined, nad map will be undefined at dead trace location.

        Example::

            cube = Cube('some.segy')
            surf = RegularSurface('s.gri')
            # update surf to sample cube values:
            surf.slice_cube(cube)

        Raises:
            Exception if maps have different definitions (topology)
            RuntimeWarning if number of sampled nodes is less than 10%
        """

        ier = _regsurf_cube.slice_cube(
            self,
            cube,
            zsurf=zsurf,
            sampling=sampling,
            mask=mask,
            snapxy=snapxy,
            deadtraces=deadtraces,
        )
        if ier == -4:
            xtg.warnuser(
                "Number of sampled surface nodes < 10 percent of " "Cube nodes"
            )
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
        algorithm=1,
    ):
        """Slice the cube within a vertical window and get the statistical
        attrubute.

        The statistical attribute can be min, max etc. Attributes are:

        * 'max' for maximum

        * 'min' for minimum

        * 'rms' for root mean square

        * 'mean' for expected value

        * 'var' for variance

        * 'maxpos' for maximum of positive values

        * 'maxneg' for maximum of negative values ??

        * 'maxabs' for maximum of absolute values

        * 'sumpos' for sum of positive values

        * 'sumneg' for sum of negative values

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
            sampling (str): 'nearest' for nearest node (default), or
                'trilinear' for trilinear interpolation.
            mask (bool): If True (default), then the map values outside
                the cube will be undef.
            zrange (float): The one-sided "radius" range of the window, e.g. 10
                (10 is default) units (e.g. meters if in depth mode).
                The full window is +- zrange (i.e. diameter).
                If other surface is present, zrange is computed based on that.
            ndiv (int): Number of intervals for sampling within zrange. None
                means 'auto' sampling, using 0.5 of cube Z increment as basis.
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
            algorithm (int): 1 for old method, 2 for new alternative.

        Example::

            cube = Cube('some.segy')
            surf = RegularSurface('s.gri')
            # update surf to sample cube values in a total range of 30 m:
            surf.slice_cube_window(cube, attribute='min', zrange=15.0)

            # Here a list is given instead:
            alst = ['min', 'max', 'rms']

            myattrs = surf.slice_cube_window(cube, attribute=alst, zrange=15.0)
            for attr in myattrs.keys():
                myattrs[attr].to_file('myfile_' + attr + '.gri')

        Raises:
            Exception if maps have different definitions (topology)
            ValueError if attribute is invalid.

        Returns:
            If attribute is a string, then the instance is updated and
            None is returned. If attribute is a list, then a dictionary
            of surface objects is returned.
        """

        if other is None and zrange is None:
            zrange = 10

        asurfs = _regsurf_cube.slice_cube_window(
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

    # =========================================================================
    # Special methods
    # =========================================================================

    def get_fence(self, xyfence):
        """
        Sample the surface along X and Y positions (numpy arrays) and get Z.

        Note the result is a masked numpy (2D) with rows masked

        Args:
            xyfence (np): A 2D numpy array with shape (N, 3) where columns
            are (X, Y, Z). The Z will be updated to the map.
        """

        xyfence = _regsurf_oper.get_fence(self, xyfence)

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
        """
        Make an average map (DZ weighted) based on numpy arrays of
        properties from a 3D grid.

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
        title="QuickPlot",
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
        """Make map values as horizontal distance from a point with azimuth
        direction.

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

            map = RegularSurface()
            map.from_file('some_file')
            map.translate((300,500,0))
            map.to_file('some_file')

        """

        xshift, yshift, zshift = translate

        # just shift the xori and yori
        self.xori = self.xori + xshift
        self.yori = self.yori + yshift

        # note the Z coordinates are perhaps not depth
        # numpy operation:
        self.values = self.values + zshift
