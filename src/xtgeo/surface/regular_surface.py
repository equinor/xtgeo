# -*- coding: utf-8 -*-
"""Module/class for regular surfaces with XTGeo.

Note that an instance of a regular surface can be made directly with::

 import xtgeo
 mysurf = xtgeo.surface_from_file('some_name.gri')

or::

 mysurf = xtgeo.surface_from_roxar('some_rms_project', 'TopX', 'DepthSurface')

"""

from __future__ import print_function, absolute_import

import os
import sys
import math
import logging
import numpy as np
import numpy.ma as ma

import os.path
from types import FunctionType

import cxtgeo.cxtgeo as _cxtgeo
from xtgeo.common import XTGeoDialog
from xtgeo.plot import Map
from xtgeo.xyz import Points
from xtgeo.common.constants import UNDEF, UNDEF_LIMIT
from xtgeo.common.constants import VERYLARGENEGATIVE, VERYLARGEPOSITIVE

from xtgeo.surface import _regsurf_import
from xtgeo.surface import _regsurf_export
from xtgeo.surface import _regsurf_cube
from xtgeo.surface import _regsurf_roxapi
from xtgeo.surface import _regsurf_gridding
from xtgeo.surface import _regsurf_oper


# =============================================================================
# Globals (always chack that these are same as in CLIB/CXTGEO)
# =============================================================================

# =============================================================================
# Class constructor
# Properties:
# _ncol     =  number of cols (X or I, cycling fastest)
# _nrow     =  number of rows (Y or J)
# _xori     =  X origin
# _yori     =  Y origin
# _xinc     =  X increment
# _yinc     =  Y increment
# _rotation =  Rotation in degrees, anti-clock relative to X axis (aka school)
# _values   =  Numpy 2D array of doubles, of shape (ncol,nrow)
# _cvalues  =  Pointer to C array (SWIG).
#
# Note: The _values (2D array) may be C or F contiguous. As longs as it stays
# 2D it does not matter. However, when reshaped from a 1D array, or the
# other way, we need to know, as the file i/o (ie CXTGEO) is F contiguous!
#
# =============================================================================


class RegularSurface(object):
    """Class for a regular surface in the xtgeo framework.

    The regular surface instance is usually initiated by
    import from file, but can also be made from scratch.
    The values can be accessed by the user as a 2D masked numpy float64 array,
    but also other variants are possible (e.g. as 1D ordinary numpy).

    Args:
        ncol: Integer for number of X direction columns
        nrow: Integer for number of Y direction rows
        xori: X (East) origon coordinate
        yori: Y (North) origin coordinate
        xinc: X increment
        yinc: Y increment
        rotation: rotation in degrees, anticlock from X axis between 0, 360
        values: 2D (masked or not)  numpy array of shape (ncol,nrow), F order

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

    def __init__(self, *args, **kwargs):
        """The __init__ (constructor) method."""

        clsname = '{}.{}'.format(type(self).__module__, type(self).__name__)
        self.logger = logging.getLogger(clsname)
        self.logger.addHandler(logging.NullHandler())

        self._xtg = XTGeoDialog()

        self._undef = UNDEF
        self._undef_limit = UNDEF_LIMIT
        self._cvalues = None     # carray swig C pointer of map values
        self._masked = False
        self._filesrc = None     # Name of original input file

        # assume so far:
        self._yflip = 1

        if args:
            # make instance from file import
            mfile = args[0]
            fformat = kwargs.get('fformat', 'guess')
            self.from_file(mfile, fformat=fformat)

        else:
            # make instance by kw spesification
            self._xori = kwargs.get('xori', 0.0)
            self._yori = kwargs.get('yori', 0.0)
            if 'nx' in kwargs:
                self._ncol = kwargs.get('nx', 5)  # backward compatibility
                self._nrow = kwargs.get('ny', 3)  # backward compatibility
            else:
                self._ncol = kwargs.get('ncol', 5)
                self._nrow = kwargs.get('nrow', 3)
            self._xinc = kwargs.get('xinc', 25.0)
            self._yinc = kwargs.get('yinc', 25.0)
            self._rotation = kwargs.get('rotation', 0.0)

            values = kwargs.get('values', None)

            if values is None:
                values = np.array([[1, 6, 11],
                                   [2, 7, 12],
                                   [3, 8, 1e33],
                                   [4, 9, 14],
                                   [5, 10, 15]],
                                  dtype=np.double, order='F')
                # make it masked
                values = ma.masked_greater(values, UNDEF_LIMIT)
                self._masked = True
                self._values = values
            else:
                self._values = self.ensure_correct_values(self.ncol,
                                                          self.nrow, values)

                if self._check_shape_ok(self._values) is False:
                    raise ValueError('Wrong dimension of values')

        # _nsurfaces += 1

        if self._values is not None:
            self.logger.debug('Shape of value: and values')
            self.logger.debug('\n{}'.format(self._values.shape))
            self.logger.debug('\n{}'.format(repr(self._values)))

        self.logger.debug('Ran __init__ method for RegularSurface object')

    def __repr__(self):
        avg = self.values.mean()
        dsc = ('{0.__class__} (ncol={0.ncol!r}, '
               'nrow={0.nrow!r}, original file: {0._filesrc}), '
               'average {1} ID=<{2}>'.format(self, avg, id(self)))
        return dsc

    def __str__(self):
        avg = self.values.mean()
        dsc = ('{0.__class__.__name__} (ncol={0.ncol!r}, '
               'nrow={0.nrow!r}, original file: {0._filesrc}), '
               'average {1:.4f}'.format(self, avg))
        return dsc

    def __del__(self):
        self._delete_cvalues()

    # =========================================================================
    # Class and static methods
    # =========================================================================

    @classmethod
    def methods(cls):
        """Returns the names of the methods in the class.

        >>> print(RegularSurface.methods())
        """
        return [x for x, y in cls.__dict__.items() if type(y) == FunctionType]

    @staticmethod
    def ensure_correct_values(ncol, nrow, values):
        """Ensures that values is a 2D masked numpy (ncol, nrol), F order.

        Example::

            vals = np.ones((nx*ny))  # a 1D numpy array in C order by default

            # secure that the values are masked, in correct format and shape:
            mymap.values = mymap.ensure_correct_values(nc, nr, vals)
        """
        if np.isscalar(values):
            vals = ma.zeros((ncol, nrow), order='F', dtype=np.double)
            values = vals + float(values)

        if not isinstance(values, ma.MaskedArray):
            values = ma.array(values, order='F')

        if not values.shape == (ncol, nrow):
            values = ma.reshape(values, (ncol, nrow), order='F')

        # replace any undef or nan with mask
        values = ma.masked_greater(values, UNDEF_LIMIT)
        values = ma.masked_invalid(values)

        if not values.flags.f_contiguous:
            mask = ma.getmaskarray(values)
            mask = np.asfortranarray(mask)
            values = np.asfortranarray(values)
            values = ma.array(values, mask=mask, order='F')

        return values

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def ncol(self):
        """The NCOL (NX or N-Idir) number, as property."""
        return self._ncol

    @ncol.setter
    def ncol(self, n):
        raise ValueError('Cannot change ncol')

    @property
    def nrow(self):
        """The NROW (NY or N-Jdir) number, as property."""
        return self._nrow

    @nrow.setter
    def nrow(self, n):
        raise ValueError('Cannot change nrow')

    @property
    def nx(self):
        """The NX (or N-Idir) number, as property (deprecated, use ncol)."""
        self.logger.warning('Deprecated; use ncol instead')
        return self._ncol

    @nx.setter
    def nx(self, n):
        self.logger.warning('Cannot change nx')
        raise ValueError('Cannot change nx')

    @property
    def ny(self):
        """The NY (or N-Jdir) number, as property (deprecated, use nrow)."""
        self.logger.warning('Deprecated; use nrow instead')
        return self._nrow

    @ny.setter
    def ny(self, n):
        raise ValueError('Cannot change ny')

    @property
    def rotation(self):
        """The rotation, anticlock from X axis, in degrees [0..360]."""
        return self._rotation

    @rotation.setter
    def rotation(self, rota):
        if rota >= 0 and rota < 360:
            self._rotation = rota
        else:
            raise ValueError

    @property
    def xinc(self):
        """The X increment (or I dir increment)."""
        self.logger.debug('Enter method...')
        return self._xinc

    @property
    def yinc(self):
        """The Y increment (or I dir increment)."""
        return self._yinc

    @property
    def yflip(self):
        """The Y flip indicator"""
        return self._yflip

    @property
    def xori(self):
        """The X coordinate origin of the map (can be modified)."""
        return self._xori

    @xori.setter
    def xori(self, xnew):
        self._xori = xnew

    @property
    def yori(self):
        """The Y coordinate origin of the map (can be modified)."""
        return self._yori

    @yori.setter
    def yori(self, ynew):
        self._yori = ynew

    @property
    def xmin(self):
        """The minimim X coordinate"""
        corners = self.get_map_xycorners()

        xmin = VERYLARGEPOSITIVE
        for c in corners:
            if c[0] < xmin:
                xmin = c[0]
        return xmin

    @property
    def xmax(self):
        """The maximum X coordinate"""
        corners = self.get_map_xycorners()

        xmax = VERYLARGENEGATIVE
        for c in corners:
            if c[0] > xmax:
                xmax = c[0]
        return xmax

    @property
    def ymin(self):
        """The minimim Y coordinate"""
        corners = self.get_map_xycorners()

        ymin = VERYLARGEPOSITIVE
        for c in corners:
            if c[1] < ymin:
                ymin = c[1]
        return ymin

    @property
    def ymax(self):
        """The maximum Y xoordinate"""
        corners = self.get_map_xycorners()

        ymax = VERYLARGENEGATIVE
        for c in corners:
            if c[1] > ymax:
                ymax = c[1]
        return ymax

    @property
    def values(self):
        """The map values, as 2D masked numpy (float64), shape (ncol, nrow)."""
        self._update_values()

        if not self._values.flags.f_contiguous:
            self.logger.warning('Not Fortran order in numpy')

        return self._values

    @values.setter
    def values(self, values):
        self.logger.debug('Enter method...')

        values = self.ensure_correct_values(self.ncol, self.nrow, values)

        self._values = values
        self._cvalues = None

        self.logger.debug('Values shape: {}'.format(self._values.shape))
        self.logger.debug('Flags: {}'.format(self._values.flags.c_contiguous))

    @property
    def values1d(self):
        """The map values, as 1D numpy, not masked, with undef as np.nan.

        Example::

            map = RegularSurface('myfile.gri')
            values = map.values1d
            mean = np.nanmean(values)
            values[values <= 0] = np.nan
            map.values1d = values  # update
        """
        self._update_values()
        val = self._values.flatten(order='F')  # flatten will return a copy
        val = ma.filled(val, UNDEF)
        val[val > UNDEF_LIMIT] = np.nan
        return val

    @values1d.setter
    def values1d(self, ndarray):

        if not isinstance(ndarray, np.ndarray):
            raise ValueError('Provided array is not a Numpy ndarray')

        if ndarray.shape[0] != self.ncol * self.nrow:
            raise ValueError('Provided array has wrong shape')

        val = np.reshape(ndarray, (self._ncol, self._nrow), order='F')

        self._values = ma.masked_invalid(val)  # will return a copy

    @property
    def cvalues(self):
        """The map values, as 1D C pointer i.e. a reference only (F-order)."""
        self._update_cvalues()
        return self._cvalues

    @cvalues.setter
    def cvalues(self, cvalues):
        self.logger.warn('Not possible!')

    @property
    def undef(self):
        """Returns the undef value, to be used when in the get_zval method."""
        return self._undef

    @property
    def undef_limit(self):
        """Returns the undef_limit value, to be used when in the
        get_zval method."""
        return self._undef_limit

# =============================================================================
# Import and export
# =============================================================================

    def from_file(self, mfile, fformat='guess'):
        """Import surface (regular map) from file.

        Note that the 'guess' option will look at the file extesions, where
        "gri" will irap_binary and "fgr" assume Irap Ascii

        Args:
            mfile (str): Name of file
            fformat (str): File format, guess/irap_binary/irap_ascii
                is currently supported.


        Returns:
            Object instance, optionally.

        Example:
            Here the from_file method is used to initiate the object
            directly::

            >>> mymapobject = RegularSurface().from_file('myfile.x')


        """

        self._values = None

        if os.path.isfile(mfile):
            pass
        else:
            self.logger.critical('Not OK file')
            raise os.error

        froot, fext = os.path.splitext(mfile)
        if fformat is None or fformat == 'guess':
            if len(fext) == 0:
                self.logger.critical('File extension missing. STOP')
                raise SystemExit('Stop: fformat is "guess" but file '
                                 'extension is missing')
            else:
                fformat = fext.lower().replace('.', '')

        if fformat in ['irap_binary', 'gri', 'bin', 'irapbin']:
            sdata = _regsurf_import.import_irap_binary(mfile)
        elif fformat in ['irap_ascii', 'fgr', 'asc', 'irapasc']:
            sdata = _regsurf_import.import_irap_ascii(mfile)
        else:
            raise ValueError('Invalid file format: {}'.format(fformat))

        self._ncol = sdata['ncol']
        self._nrow = sdata['nrow']
        self._xori = sdata['xori']
        self._yori = sdata['yori']
        self._xinc = sdata['xinc']
        self._yinc = sdata['yinc']
        self._rotation = sdata['rotation']
        self._cvalues = sdata['cvalues']
        self._values = sdata['values']

        self._filesrc = mfile

        return self

    def to_file(self, mfile, fformat='irap_binary'):
        """Export surface (regular map) to file

        Args:
            mfile (str): Name of file
            fformat (str): File format, irap_binary/irap_ascii

        Example::

            >>> x = RegularSurface()
            >>> x.from_file('myfile.x', fformat = 'irap_ascii')
            >>> x.values = x.values + 300
            >>> x.to_file('myfile2.x', fformat = 'irap_ascii')

        """

        self.logger.debug('Enter method...')
        self.logger.info('Export to file...')
        if (fformat == 'irap_ascii'):
            _regsurf_export.export_irap_ascii(self, mfile)
        elif (fformat == 'irap_binary'):
            _regsurf_export.export_irap_binary(self, mfile)
        else:
            self.logger.critical('Invalid file format')

    def from_roxar(self, project, name, category, stype='horizons',
                   realisation=0):
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
        valid_stypes = ['horizons', 'zones']

        if stype not in valid_stypes:
            raise ValueError('Invalid stype, only {} stypes is supported.'
                             .format(valid_stypes))

        self = _regsurf_roxapi.import_horizon_roxapi(
            self, project, name, category, stype, realisation)

    def to_roxar(self, project, name, category, stype='horizons',
                 realisation=0):
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
        valid_stypes = ['horizons', 'zones']

        if stype in valid_stypes and name is None or category is None:
            self.logger.error('Need to spesify name and category for '
                              'horizon')
        elif stype not in valid_stypes:
            raise ValueError('Only {} stype is supported per now'
                             .format(valid_stypes))

        self = _regsurf_roxapi.export_horizon_roxapi(
            self, project, name, category, stype, realisation)

    def copy(self):
        """Copy a xtgeo.surface.RegularSurface object to another instance::

            >>> mymapcopy = mymap.copy()

        """
        self.logger.debug('Copy object instance...')
        self.logger.debug(self._values)
        self.logger.debug(self._values.flags)
        self.logger.debug(id(self._values))

        xsurf = RegularSurface(ncol=self.ncol, nrow=self.nrow, xinc=self.xinc,
                               yinc=self.yinc, xori=self.xori, yori=self.yori,
                               rotation=self.rotation,
                               values=self.values)

        self.logger.debug('New array + flags + ID')
        self.logger.debug(xsurf._values)
        self.logger.debug(xsurf._values.flags)
        self.logger.debug(id(xsurf._values))
        return xsurf

    def get_zval(self):
        """Get an an 1D, numpy array of the map values (not masked).

        Note that undef values are very large numbers (see undef property).
        Also, this will reorder a 2D values array to column fastest, i.e.
        get stuff into Fortran order.
        """

        self._update_values()

        self.logger.debug('Enter method...')
        self.logger.debug('Shape: {}'.format(self._values.shape))

        if self._check_shape_ok(self._values) is False:
            raise ValueError

        # unmask the self._values numpy array, by filling the masked
        # values with undef value
        self.logger.debug('Fill the masked...')
        x = ma.filled(self._values, self._undef)

        # make it 1D (Fortran order)
        self.logger.debug('1D')

        x = np.reshape(x, -1, order='F')

        self.logger.debug('1D done {}'.format(x.shape))

        return x

    def set_zval(self, x):
        """Set a 1D (unmasked) numpy array.

        The numpy array must be in Fortran order (i columns (ncol) fastest).
        """

        # NOTE: Will convert it to a 2D masked array internally.

        self._update_values()

        # not sure if this is right always?...
        x = np.reshape(x, (self._ncol, self._nrow), order='F')

        # replace Nans, if any
        x = np.where(np.isnan(x), self.undef, x)

        # make it masked
        x = ma.masked_greater(x, self._undef_limit)

        self._values = x

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

    def compare_topology(self, other):
        """Check that two object has the same topology, i.e. map definitions.

        Map definitions such as origin, dimensions, number of defined cells...

        Args:
            other (surface object): The other surface to compare with

        Returns:
            True of same topology, False if not
        """
        if (self.ncol != other.ncol or self.nrow != other.nrow or
                self.xori != other.xori or self.yori != other.yori or
                self.xinc != other.xinc or self.yinc != other.yinc or
                self.rotation != other.rotation):

            return False

        # check that masks are equal
        m1 = ma.getmaskarray(self.values)
        m2 = ma.getmaskarray(other.values)
        if not np.array_equal(m1, m2):
            return False

        self.logger.debug('Surfaces have same topology')
        return True

    def get_map_xycorners(self):
        """Get the X and Y coordinates of the map corners.

        Returns a tuple on the form
        ((x0, y0), (x1, y1), (x2, y2), (x3, y3)) where
        (if unrotated and normal flip) 0 is the lower left
        corner, 1 is the right, 2 is the upper left, 3 is the upper right.
        """

        rot1 = self._rotation * math.pi / 180
        rot2 = rot1 + (math.pi / 2.0)

        x0 = self._xori
        y0 = self._yori

        x1 = self._xori + (self.ncol - 1) * math.cos(rot1) * self._xinc
        y1 = self._yori + (self.ncol - 1) * math.sin(rot1) * self._xinc

        x2 = self._xori + (self.nrow - 1) * math.cos(rot2) * self._yinc
        y2 = self._yori + (self.nrow - 1) * math.sin(rot2) * self._yinc

        x3 = x2 + (self.ncol - 1) * math.cos(rot1) * self._xinc
        y3 = y2 + (self.ncol - 1) * math.sin(rot1) * self._xinc

        return ((x0, y0), (x1, y1), (x2, y2), (x3, y3))

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
        """Returns x, y, z(value) from i j location.

        Args:
            iloc (int): I (col) location (base is 1)
            jloc (int): J (row) location (base is 1)
            zvalues (ndarray). If this is used in a loop it is wise
                to precompute the numpy surface once in the caller,
                and submit the numpy array (use surf.get_zval()).

        Returns:
            The z value at location iloc, jloc, None if undefined cell.
        """

        xval, yval, value = _regsurf_oper.get_xy_value_from_ij(
            self, iloc, jloc, zvalues=zvalues)

        return xval, yval, value

    def get_xy_values(self):
        """Return coordinates for X and Y as numpy 2D masked arrays."""

        xvals, yvals = _regsurf_oper.get_xy_values(self)

        return xvals, yvals

    def get_xyz_values(self):
        """Return coordinates for X Y and Z (values) as numpy 2D masked
        arrays."""

        xc, yc = self.get_xy_values()

        return xc, yc, self.values

    def get_xy_value_lists(self, lformat='webportal', xyfmt=None,
                           valuefmt=None):
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

        zvalues = self.get_zval()

        for j in range(self.nrow):
            for i in range(self.ncol):
                xc, yc, vc = self.get_xy_value_from_ij(i + 1, j + 1,
                                                       zvalues=zvalues)

                if vc is not None:
                    if xyfmt is not None:
                        xc = float('{:{width}}'.format(xc, width=xyfmt))
                        yc = float('{:{width}}'.format(yc, width=xyfmt))
                    if valuefmt is not None:
                        vc = float('{:{width}}'.format(vc, width=valuefmt))
                    valuelist.append(vc)
                    xylist.append((xc, yc))

        return xylist, valuelist

    # =========================================================================
    # Interacion with points
    # =========================================================================

    def gridding(self, points, method='linear', coarsen=1):
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

        if not isinstance(points, Points):
            raise ValueError('Argument not a Points '
                             'instance')

        self.logger.info('Do gridding...')

        _regsurf_gridding.points_gridding(self, points, coarsen=coarsen,
                                          method=method)

    # =========================================================================
    # Interacion with other surface
    # =========================================================================

    def resample(self, other):
        """Resample a surface from another surface instance.

        Note that there may be some 'loss' of nodes at the edges of the
        updated map, as only the 'inside' nodes in the updated map
        versus the input map are applied.

        Args:
            other(RegularSurface): Surface to resample from
        """

        if not isinstance(other, RegularSurface):
            raise ValueError('Argument not a RegularSurface '
                             'instance')

        self.logger.info('Do resampling...')

        _regsurf_oper.resample(self, other)

    # =========================================================================
    # Change a surface more fundamentally
    # =========================================================================

    def unrotate(self):
        """Unrotete a map instance, and this will also change nrow, ncol,
        xinc, etc.

        The default sampling makes a finer gridding in order to avoid
        aliasing.

        """

        xlen = self.xmax - self.xmin
        ylen = self.ymax - self.ymin
        ncol = self.ncol * 2
        nrow = self.nrow * 2
        xinc = xlen / (ncol - 1)
        yinc = ylen / (nrow - 1)
        vals = ma.zeros((ncol, nrow), order='F')

        nonrot = RegularSurface(xori=self.xmin,
                                yori=self.ymin,
                                xinc=xinc, yinc=yinc,
                                ncol=ncol, nrow=nrow,
                                values=vals)
        nonrot.resample(self)

        self._values = nonrot.values
        self._nrow = nonrot.nrow
        self._ncol = nonrot.ncol
        self._rotation = nonrot.rotation
        self._xori = nonrot.xori
        self._yori = nonrot.yori
        self._xinc = nonrot.xinc
        self._yinc = nonrot.yinc
        self._cvalues = None

    # =========================================================================
    # Interacion with a cube
    # =========================================================================

    def slice_cube(self, cube, zsurf=None, sampling='nearest', mask=True):
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

        Example::

            cube = Cube('some.segy')
            surf = RegularSurface('s.gri')
            # update surf to sample cube values:
            surf.slice_cube(cube)

        Raises:
            Exception if maps have different definitions (topology)
            RuntimeWarning if number of sampled nodes is less than 10%
        """

        ier = _regsurf_cube.slice_cube(self, cube, zsurf=zsurf,
                                       sampling=sampling, mask=mask)

        if ier == -4:
            raise RuntimeWarning('Number of sampled nodes < 10\%')
        elif ier == -5:
            raise RuntimeWarning('No nodes sampled: map is outside cube?')

    def slice_cube_window(self, cube, zsurf=None, other=None,
                          other_position='below', sampling='nearest',
                          mask=True, zrange=None, ndiv=None, attribute='max'):
        """Slice the cube within a vertical window and get the statistical
        attrubute.

        The statistical attribute can be min, max etc. Attributes are:

        * 'max' for maximum

        * 'min' for minimum

        * 'rms' for root mean square

        * 'mean' for expected value

        * 'var' for variance

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
            zrange (float): The one-sided range of the window, e.g. 10
                (10 is default) units (e.g. meters if in depth mode).
                The full window is +- zrange. If other surface is present,
                zrange is computed based on that.
            ndiv (int): Number of intervals for sampling within zrange. None
                means 'auto' sampling, using 0.5 of cube Z increment as basis.
            attribute (str): The requested attribute, e.g. 'max' value

        Example::

            cube = Cube('some.segy')
            surf = RegularSurface('s.gri')
            # update surf to sample cube values:
            surf.slice_cube_window(cube, attribute='min', zrange=15.0)

        Raises:
            Exception if maps have different definitions (topology)
            ValueError if attribute is invalid.
        """

        if other is None and zrange is None:
            zrange = 10

        _regsurf_cube.slice_cube_window(self, cube, zsurf=zsurf, other=other,
                                        other_position=other_position,
                                        sampling=sampling, mask=mask,
                                        zrange=zrange, ndiv=ndiv,
                                        attribute=attribute)

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

    def hc_thickness_from_3dprops(self, xprop=None, yprop=None,
                                  hcpfzprop=None, zoneprop=None,
                                  zone_minmax=None, dzprop=None,
                                  zone_avg=False, coarsen=1,
                                  mask_outside=False):
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
        subname = sys._getframe().f_code.co_name

        for i, myprop in enumerate([xprop, yprop, hcpfzprop, zoneprop]):
            if isinstance(myprop, ma.MaskedArray):
                raise ValueError('Property input {} with avg {} to {} is a '
                                 'masked array, not a plain numpy ndarray'
                                 .format(i, myprop.mean(), subname))

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
            mask_outside=mask_outside)

        if status is False:
            raise RuntimeError('Failure from hc thickness calculation')

    def avg_from_3dprop(self, xprop=None, yprop=None,
                        mprop=None, dzprop=None,
                        truncate_le=None, zoneprop=None, zone_minmax=None,
                        coarsen=1, zone_avg=False):
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

        subname = sys._getframe().f_code.co_name

        for i, myprop in enumerate([xprop, yprop, mprop, dzprop, zoneprop]):
            if isinstance(myprop, ma.MaskedArray):
                raise ValueError('Property input {} with avg {} to {} is a '
                                 'masked array, not a plain numpy ndarray'
                                 .format(i, myprop.mean(), subname))

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
            zone_avg=zone_avg)

    def quickplot(self, filename='/tmp/default.png', title='QuickPlot',
                  infotext=None, minmax=(None, None), xlabelrotation=None,
                  colortable='rainbow', faults=None, logarithmic=False):
        """Fast surface plot of maps using matplotlib.

        Args:
            filename (str): Name of plot file.
            title (str): Title of plot
            infotext (str): Additonal info on plot.
            minmax (tuple): Tuple of min and max values to be plotted. Note
                that values outside range will be set equal to range limits
            xlabelrotation (float): Rotation in degrees of X labels.
            colortable (str): Name of matplotlib or RMS file or XTGeo
                colortable. Default is matplotlib's 'rainbow'
            faults (dict): If fault plot is wanted, a dictionary on the
                form => {'faults': XTGeo Polygons object, 'color': 'k'}
            logarithmic (bool): If True, a logarithmic contouring color scale
                will be used.

        """

        # This is using the more versatile Map class in XTGeo. Most kwargs
        # is just passed as is. Prefer using Map() directly in apps?

        mymap = Map()

        self.logger.info('Infotext is <{}>'.format(infotext))
        mymap.canvas(title=title, infotext=infotext)

        minvalue = minmax[0]
        maxvalue = minmax[1]

        mymap.set_colortable(colortable)

        mymap.plot_surface(self, minvalue=minvalue,
                           maxvalue=maxvalue, xlabelrotation=xlabelrotation,
                           colortable=colortable, logarithmic=logarithmic)
        if faults:
            mymap.plot_faults(faults['faults'])

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

    # =========================================================================
    # Helper methods, for internal usage
    # -------------------------------------------------------------------------
    # copy self (update) values from SWIG carray to numpy, 1D array

    def _update_values(self):
        nnum = self._ncol * self._nrow

        if self._cvalues is None and self._values is not None:
            return

        elif self._cvalues is None and self._values is None:
            self.logger.critical('_cvalues and _values is None in '
                                 '_update_values. STOP')
            sys.exit(9)

        xvv = _cxtgeo.swig_carr_to_numpy_1d(nnum, self._cvalues)

        xvv = np.reshape(xvv, (self._ncol, self._nrow), order='F')

        # make it masked
        xvv = ma.masked_greater(xvv, self._undef_limit)

        self._values = xvv

        self._delete_cvalues()

    # copy (update) values from numpy to SWIG, 1D array

    def _update_cvalues(self):
        self.logger.debug('Enter update cvalues method...')
        nnum = self._ncol * self._nrow

        if self._values is None and self._cvalues is not None:
            self.logger.debug('CVALUES unchanged')
            return

        elif self._cvalues is None and self._values is None:
            self.logger.critical('_cvalues and _values is None in '
                                 '_update_cvalues. STOP')
            sys.exit(9)

        elif self._cvalues is not None and self._values is None:
            self.logger.critical('_cvalues and _values are both present in '
                                 '_update_cvalues. STOP')
            sys.exit(9)

        # make a 1D F order numpy array, and update C array
        xvv = ma.filled(self._values, self._undef)
        xvv = np.reshape(xvv, -1, order='F')

        self._cvalues = _cxtgeo.new_doublearray(nnum)

        _cxtgeo.swig_numpy_to_carr_1d(xvv, self._cvalues)
        self.logger.debug('Enter method... DONE')

        self._values = None

    def _delete_cvalues(self):
        self.logger.debug('Enter delete cvalues values method...')

        if self._cvalues is not None:
            _cxtgeo.delete_doublearray(self._cvalues)

        self._cvalues = None
        self.logger.debug('Enter method... DONE')

    # check if values shape is OK (return True or False)

    def _check_shape_ok(self, values):

        if not values.flags['F_CONTIGUOUS']:
            self.logger.error('Wrong order; shall be Fortran (Flags: {}'
                              .format(values.flags))
            return False

        (ncol, nrow) = values.shape
        if ncol != self._ncol or nrow != self._nrow:
            self.logger.error('Wrong shape: Dimens of values {} {} vs {} {}'
                              .format(ncol, nrow, self._ncol, self._nrow))
            return False
        return True

    def _convert_np_carr_double(self, np_array, nlen):
        """Convert numpy 1D array to C array, assuming double type"""
        carr = _cxtgeo.new_doublearray(nlen)

        _cxtgeo.swig_numpy_to_carr_1d(np_array, carr)

        return carr

    def _convert_carr_double_np(self, carray, nlen=None):
        """Convert a C array to numpy, assuming double type."""
        if nlen is None:
            nlen = len(self._df.index)

        nparray = _cxtgeo.swig_carr_to_numpy_1d(nlen, carray)

        return nparray

# =============================================================================
# METHODS as wrappers to class init + import


def surface_from_file(mfile, fformat='guess'):
    """This makes an instance of a RegularSurface directly from import."""

    obj = RegularSurface()

    obj.from_file(mfile, fformat=fformat)

    return obj


def surface_from_roxar(project, name, category, stype='horizons',
                       realisation=0):
    """This makes an instance of a RegularSurface directly from roxar input."""

    obj = RegularSurface()

    obj.from_roxar(project, name, category, stype=stype,
                   realisation=realisation)

    return obj
