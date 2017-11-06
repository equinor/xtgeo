# -*- coding: utf-8 -*-
"""Module/class for regular surfaces with XTGeo."""

from __future__ import print_function

import os
import sys
import math
import logging
import numpy as np
import numpy.ma as ma
import scipy.interpolate
import os.path
from types import FunctionType


import cxtgeo.cxtgeo as _cxtgeo
from xtgeo.common import XTGeoDialog
from xtgeo.plot import Map

from xtgeo.surface import _regsurf_import
from xtgeo.surface import _regsurf_cube

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

    Attributes:
        ncol: Integer for number of X direction columns
        nrow: Integer for number of Y direction rows
        xori: X (East) origon coordinate
        yori: Y (North) origin coordinate
        xinc: X increment
        yinc: Y increment
        rotation: rotation in degrees, anticlock from X axis between 0, 360
        values: 2D (masked or not)  numpy array of shape (ncol,nrow), F order

    Example:
        Initiate a class and import::

          from xtgeo.surface import RegularSurface
          x = RegularSurface()
          x.from_file('some.irap', fformat='irap_binary')

    """

    def __init__(self, *args, **kwargs):
        """The __init__ (constructor) method.

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

        Args:
            xori (float): Origin of grid X (East) coordinate
            yori (float): Origin of grid Y (North) coordinate
            xinc (float): Increment in X
            yinc (float): Increment in Y
            ncol (int): Number of columns, X
            nrow (int): Number of rows, Y
            rotation (float): Rotation angle (deg.), from X axis, anti-clock
            values (ndarray): 2D numpy (maked or not) of shape (ncol,nrow))

        """

        clsname = '{}.{}'.format(type(self).__module__, type(self).__name__)
        self.logger = logging.getLogger(clsname)
        self.logger.addHandler(logging.NullHandler())

        self._xtg = XTGeoDialog()

        self._undef = _cxtgeo.UNDEF
        self._undef_limit = _cxtgeo.UNDEF_LIMIT
        self._cvalues = None     # carray swig C pointer of map values
        self._masked = False

        # assume so far:
        self._yflip = 1

        if args:
            # make instance from file import
            mfile = args[0]
            fformat = kwargs.get('fformat', 'irap_binary')
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
                self._values = np.array([[1, 6, 11],
                                         [2, 7, 12],
                                         [3, 8, 1e33],
                                         [4, 9, 14],
                                         [5, 10, 15]],
                                        dtype=np.double, order='F')
            else:
                if values.flags['F_CONTIGUOUS'] is False:
                    values = np.asfortranarray(values)

                self._values = values

                if self._check_shape_ok(self._values) is False:
                    raise ValueError('Wrong dimension of values')

            # make it masked
            self._values = ma.masked_greater(self._values, self._undef_limit)
            self._masked = True

        # _nsurfaces += 1

        if self._values is not None:
            self.logger.debug('Shape of value: and values')
            self.logger.debug('\n{}'.format(self._values.shape))
            self.logger.debug('\n{}'.format(repr(self._values)))

        self.logger.debug('Ran __init__ method for RegularSurface object')

    def __del__(self):
        self._delete_cvalues()

    @classmethod
    def methods(cls):
        """
        Returns the names of the methods in the class.

        >>> print(RegularSurface.methods())
        """
        return [x for x, y in cls.__dict__.items() if type(y) == FunctionType]

# =============================================================================
# Import and export
# =============================================================================

    def from_file(self, mfile, fformat='guess'):
        """Import surface (regular map) from file.

        Args:
            mfile (str): Name of file
            fformat (str): File format, guess/irap_binary is currently
                supported

        Returns:
            Object instance, optionally

        Example:
            Here the from_file method is used to initiate the object
            directly::

            >>> mymapobject = RegularSurface().from_file('myfile.x')


        """

        self._values = None

        if (os.path.isfile(mfile)):
            pass
        else:
            self.logger.critical('Not OK file')
            raise os.error

        if (fformat is None or fformat == 'irap_binary' or fformat == 'guess'):
            self._import_irap_binary(mfile)
        else:
            self.logger.error('Invalid file format')

        return self

    def to_file(self, mfile, fformat='irap_binary'):
        """Export surface (regular map) to file

        Args:
            mfile (str): Name of file
            fformat (str): File format, irap_binary/irap_classic

        Example::

            >>> x = RegularSurface()
            >>> x.from_file('myfile.x', fformat = 'irap_ascii')
            >>> x.values = x.values + 300
            >>> x.to_file('myfile2.x', fformat = 'irap_ascii')

        """

        self.logger.debug('Enter method...')
        self.logger.info('Export to file...')
        if (fformat == 'irap_ascii'):
            self._export_irap_ascii(mfile)
        elif (fformat == 'irap_binary'):
            self._export_irap_binary(mfile)
        else:
            self.logger.critical('Invalid file format')

    def from_roxar(self, project, type='horizon', name=None, category=None,
                   realisation=0):
        """
        Load a surface from a Roxar RMS project.

        Args:
            project (str): Name of project (as folder) if outside RMS, og just
                use the magig 'project' if within RMS.
            type (str): 'horizon', 'clipboard', etc
            name (str): Name of surface/map
            category (str): For horizon only: for example 'DS_extracted'
            realiation (int): Realisation number, default is 0

        Returns:
            Object instance updated

        Example:
            Here the from_roxar method is used to initiate the object
            directly::

            >>> mymap = RegularSurface()
            >>> mymap.from_roxar(etc)

        """

        if type == 'horizon' and name is None or category is None:
            self.logger.error('Need to spesify name and categori for '
                              'horizon')
        elif type != 'horizon':
            self.logger.error('Only horizon type is supported so far')
            raise Exception

        self._import_horizon_roxapi(project, name, category,
                                    realisation)

        return self

# =============================================================================
# Get and Set properties (tend to pythonic properties rather than javaic get
# & set syntax)
# =============================================================================

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
    def values(self):
        """The map values, as 2D masked numpy (float64) of shape (ncol, nrow)."""
        self._update_values()
        return self._values

    @values.setter
    def values(self, values):
        self.logger.debug('Enter method...')

        if (isinstance(values, np.ndarray) and
                not isinstance(values, ma.MaskedArray)):

            values = ma.array(values)

        if self._check_shape_ok(values) is False:
            raise ValueError

        self._values = values
        self._cvalues = None

        self.logger.debug('Values shape: {}'.format(self._values.shape))
        self.logger.debug('Flags: {}'.format(self._values.flags.c_contiguous))

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

    def copy(self):
        """Copy a xtgeo.surface.RegularSurface object to another instance::

            >>> mymapcopy = mymap.copy()

        """
        xsurf = RegularSurface(ncol=self.ncol, nrow=self.nrow, xinc=self.xinc,
                               yinc=self.yinc, xori=self.xori, yori=self.yori,
                               rotation=self.rotation, values=self.values)
        return xsurf


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
        except:
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
        m1 = ma.getmask(self.values)
        m2 = ma.getmask(other.values)
        if not np.array_equal(m1, m2):
            return False

        self.logger.debug('Surfaces have same topology')
        return True

    def get_value_from_xy(self, point=(0.0, 0.0)):
        """Return the map value given a X Y point.

        Args:
            point (float tuple): Position of X and Y coordinate
        Returns:
            The map value (interpolated). None if XY is outside defined map

        Example::
            mvalue = map.get_value_from_xy(point=(539291.12, 6788228.2))

        """
        xc, yc = point

        self.logger.debug('Enter value_from_cy')

        xtg_verbose_level = self._xtg.get_syslevel()

        # secure that carray is updated before SWIG/C:
        self._update_cvalues()

        # call C routine
        zc = _cxtgeo.surf_get_z_from_xy(float(xc), float(yc),
                                       self._ncol, self._nrow,
                                       self._xori, self._yori, self._xinc,
                                       self._yinc, self._yflip, self._rotation,
                                       self._cvalues, xtg_verbose_level)

        if zc > self._undef_limit:
            return None

        return zc

    def get_xy_value_from_ij(self, i, j):
        """Returns x, y, z(value) from i j location.

        If undefined cell, value is returned as None.
        """

        _cxtgeo.xtg_verbose_file('NONE')

        xtg_verbose_level = self._xtg.get_syslevel()

        if xtg_verbose_level < 0:
            xtg_verbose_level = 0

        if 1 <= i <= self.ncol and 1 <= j <= self.nrow:

            ier, xval, yval, value = (
                _cxtgeo.surf_xyz_from_ij(i, j,
                                         self.xori, self.xinc,
                                         self.yori, self.yinc,
                                         self.ncol, self.nrow, self._yflip,
                                         self.rotation, self.cvalues,
                                         0, xtg_verbose_level))
            if ier != 0:
                self.logger.critical('Error code {}, contact the author'.
                                     format(ier))
                sys.exit(9)

        else:
            raise Exception

        if value > self.undef_limit:
            value = None

        return xval, yval, value

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

        _cxtgeo.xtg_verbose_file('NONE')

        xtg_verbose_level = self._xtg.get_syslevel()

        if xtg_verbose_level < 0:
            xtg_verbose_level = 0

        xylist = []
        valuelist = []

        for j in range(self.nrow):
            for i in range(self.ncol):
                x, y, v = self.get_xy_value_from_ij(i + 1, j + 1)

                if v is not None:
                    if xyfmt is not None:
                        x = float('{:{width}}'.format(x, width=xyfmt))
                        y = float('{:{width}}'.format(y, width=xyfmt))
                    if valuefmt is not None:
                        v = float('{:{width}}'.format(v, width=valuefmt))
                    valuelist.append(v)
                    xylist.append((x, y))

        return xylist, valuelist

# =============================================================================
# Interacion with a cube
# =============================================================================

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

            cube = Cube()
            cube.from_file('some.segy')
            surf = RegularSurface()
            surf.from_file('s.gri')
            # update surf to sample cube values:
            surf.slice_cube(cube)

        Raises:
            Exception if maps have different definitions (topology)
        """

        _regsurf_cube.slice_cube(self, cube, zsurf=zsurf,
                                 sampling=sampling, mask=mask)

    def slice_cube_window(self, cube, zsurf=None, sampling='nearest',
                          mask=True, zrange=10, ndiv=None, attribute='max'):
        """Slice the cube within a vertical window and get the statistical
        attrubute.

        The statistical attribute can be min, max etc. Attributes are:

        * 'max' for maximum

        * 'min' for minimum

        * 'rms' for root mean square

        * 'var' for variance

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
            zrange (float): The one-sided range of the window, e.g. 10 meter
                (meter if in depth mode). The full window is +- zrange
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

        _regsurf_cube.slice_cube_window(self, cube, zsurf=zsurf,
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

        xtg_verbose_level = self._xtg.get_syslevel()
        self._update_cvalues()

        nvec = xyfence.shape[0]

        cxarr = self._convert_np_carr_double(xyfence[:, 0], nvec)
        cyarr = self._convert_np_carr_double(xyfence[:, 1], nvec)
        czarr = self._convert_np_carr_double(xyfence[:, 2], nvec)

        istat = _cxtgeo.surf_get_zv_from_xyv(nvec, cxarr, cyarr, czarr,
                                             self.ncol, self.nrow, self.xori,
                                             self.yori, self.xinc, self.yinc,
                                             self._yflip, self.rotation,
                                             self.cvalues,
                                             xtg_verbose_level)

        if istat != 0:
            self.logger.warning('Seem to be rotten')

        zarr = self._convert_carr_double_np(czarr, nlen=nvec)

        xyfence[:, 2] = zarr
        xyfence = ma.masked_greater(xyfence, self._undef_limit)
        xyfence = ma.mask_rows(xyfence)

        return xyfence

    def hc_thickness_from_3dprops(self, xprop=None, yprop=None,
                                  hcpfzprop=None, zoneprop=None,
                                  zone_minmax=None, layer_minmax=None):
        """Make a thickness weighted HC thickness map.

        Make a HC thickness map based on numpy arrays of properties
        from a 3D grid.

        Note that the input hcpfzprop is hydrocarbon fraction multiplied
        with thickness, which can be achieved by e.g.:
        cpfz = dz*poro*ntg*shc or by hcpfz = dz*hcpv/vbulk

        Args:
            xprop (ndarray): 3D numpy array of X coordinates
            yprop (ndarray): 3D numpy array of Y coordinates
            hcpfzprop (ndarray): 3D numpy array of HC fraction multiplied
                with DZ per cell.
            zoneprop (ndarray): (optional) 3D numpy array indicating zonation
                property
            zone_minmax (tuple): (optional) 2 element list indicating start
                and stop zonation (both start and end spec are included)
            layer_minmax (tuple): (optional) 2 element list indicating start
                and stop grid layers (both start and end spec are included)

        Returns:
            True if operation went OK (but check result!), False if not
        """

        # logging of input:

        a = hcpfzprop
        self.logger.debug('HCPFZ MIN MAX MEAN {} {} {}'.
                          format(a.min(), a.max(), a.mean()))
        a = xprop
        self.logger.debug('XCOORD MIN MAX MEAN {} {} {}'.
                          format(a.min(), a.max(), a.mean()))
        a = yprop
        self.logger.debug('YCOORD MIN MAX MEAN {} {} {}'.
                          format(a.min(), a.max(), a.mean()))

        if zoneprop is not None:
            a = zoneprop
            self.logger.debug('HCPFZ MIN MAX MEAN {} {} {}'.
                              format(a.min(), a.max(), a.mean()))


        ncol, nrow, nlay = xprop.shape

        if layer_minmax is None:
            layer_minmax = (1, nlay)
        else:
            minmax = list(layer_minmax)
            if minmax[0] < 1:
                minmax[0] = 1
            if minmax[1] > nlay:
                minmax[1] = nlay
            layer_minmax = tuple(minmax)

        if zone_minmax is None:
            zone_minmax = (1, 99999)

        if zone_minmax is not None:
            self.logger.info('ZONE_MINMAX {}'.format(zone_minmax))
        else:
            self.logger.debug('ZONE_MINMAX not given...')

        self.logger.info('LAYER_MINMAX {}'.format(layer_minmax))

        self.logger.debug('Grid layout is {} {} {}'.format(ncol, nrow, nlay))

        # do not allow rotation...
        if self._rotation < -0.1 or self._rotation > 0.1:
            self.logger.error('Cannot use rotated maps. Return')
            return False

        xmax = self._xori + self._xinc * self._ncol
        ymax = self._yori + self._yinc * self._nrow
        xi = np.linspace(self._xori, xmax, self._ncol)
        yi = np.linspace(self._yori, ymax, self._nrow)

        xi, yi = np.meshgrid(xi, yi, indexing='ij')

        # filter and compute per K layer (start count on 0)
        for k0 in range(layer_minmax[0] - 1, layer_minmax[1]):

            k1 = k0 + 1   # layer counting base is 1 for k1

            self.logger.info('Mapping for layer ' + str(k1) + '...')
            self.logger.debug('K0 counter is {}'.format(k0))

            if k1 == layer_minmax[0]:
                self.logger.info('Initialize zsum ...')
                zsum = np.zeros((self._ncol, self._nrow))

            # this should actually never happen...
            if k1 < layer_minmax[0] or k1 > layer_minmax[1]:
                self.logger.info('SKIP (layer_minmax)')
                continue

            zonecopy = np.copy(zoneprop[:, :, k0])

            self.logger.debug('Zone MEAN is {}'.format(zonecopy.mean()))

            actz = zonecopy.mean()
            if actz < zone_minmax[0] or actz > zone_minmax[1]:
                self.logger.info('SKIP (not active zone)')
                continue


            # get slices per layer of relevant props
            xcopy = np.copy(xprop[:, :, k0])
            ycopy = np.copy(yprop[:, :, k0])
            zcopy = np.copy(hcpfzprop[:, :, k0])


            propsum = zcopy.sum()
            if (abs(propsum) < 1e-12):
                self.logger.debug('Z property sum is {}'.format(propsum))
                self.logger.info('Too little HC, skip layer K = {}'.format(k1))
                continue
            else:
                self.logger.debug('Z property sum is {}'.format(propsum))

            # debugging info...
            self.logger.debug(xi.shape)
            self.logger.debug(yi.shape)
            self.logger.debug('XI min and max {} {}'.format(xi.min(),
                                                            xi.max()))
            self.logger.debug('YI min and max {} {}'.format(yi.min(),
                                                            yi.max()))
            self.logger.debug('XPROP min and max {} {}'.format(xprop.min(),
                                                               xprop.max()))
            self.logger.debug('YPROP min and max {} {}'.format(yprop.min(),
                                                               yprop.max()))
            self.logger.debug('HCPROP min and max {} {}'
                              .format(hcpfzprop.min(), hcpfzprop.max()))

            # need to make arrays 1D
            self.logger.debug('Reshape and filter ...')
            x = np.reshape(xcopy, -1, order='F')
            y = np.reshape(ycopy, -1, order='F')
            z = np.reshape(zcopy, -1, order='F')

            xc = np.copy(x)

            x = x[xc < self._undef_limit]
            y = y[xc < self._undef_limit]
            z = z[xc < self._undef_limit]

            self.logger.debug('Reshape and filter ... done')

            self.logger.debug('Map ... layer = {}'.format(k1))

            try:
                zi = scipy.interpolate.griddata((x, y), z, (xi, yi),
                                                method='linear',
                                                fill_value=0.0)
            except ValueError:
                self.logger.info('Not able to grid layer {}'.format(k1))
                continue

            self.logger.info('ZI shape is {}'.format(zi.shape))
            self.logger.debug('Map ... done')

            zsum = zsum + zi
            self.logger.info('Sum of HCPB layer is {}'.format(zsum.mean()))

        self.values = zsum
        self.logger.debug(repr(self._values))

        self.logger.debug('Exit from hc_thickness_from_3dprops')

        return True

    def avg_from_3dprop(self, xprop=None, yprop=None,
                        mprop=None, dzprop=None, layer_minmax=None,
                        truncate_le=None, zoneprop=None, zone_minmax=None,
                        sampling=1):
        """
        Make an average map (DZ weighted) based on numpy arrays of
        properties from a 3D grid.

        The 3D arrays mush be undef numpies of size (nx,ny,nz). Undef
        entries must be given DZ=0

        Args:
            xprop: 3D numpy of all X coordinates (also inactive cells)
            yprop: 3D numpy of all Y coordinates (also inactive cells)
            mprop: 3D numpy of requested property (e.g. porosity) all
            dzprop: 3D numpy of dz values (for weighting)
                NB zero for undef cells
            layer_minmax: Optional. A list with start layer and end
                layer (1 counting)
            truncate_le (float): Optional. Truncate value (mask) if
                value is less
            zoneprop: 3D numpy to a zone property
            zone_minmax: a list with from-to zones to combine
                (e.g. [1,3])

        Returns:
            Nothing explicit, but updates the surface object.
        """

        ncol, nrow, nlay = xprop.shape

        if layer_minmax is None:
            layer_minmax = (1, 99999)

        if zone_minmax is None:
            zone_minmax = (1, 99999)

        usezoneprop = True
        if zoneprop is None:
            usezoneprop = False

        # avoid artifacts from inactive cells that slips through somehow...
        dzprop[mprop > _cxtgeo.UNDEF_LIMIT] = 0.0

        self.logger.info('Layer from: {}'.format(layer_minmax[0]))
        self.logger.info('Layer to: {}'.format(layer_minmax[1]))
        self.logger.debug('Layout is {} {} {}'.format(ncol, nrow, nlay))

        self.logger.info('Zone from: {}'.format(zone_minmax[0]))
        self.logger.info('Zone to: {}'.format(zone_minmax[1]))
        self.logger.info('Zone is :')
        self.logger.info(zoneprop)

        # do not allow rotation...
        if self._rotation < -0.1 or self._rotation > 0.1:
            self.logger.error('Cannut use rotated maps. Return')
            return

        xmax = self._xori + self._xinc * self._ncol
        ymax = self._yori + self._yinc * self._nrow
        xi = np.linspace(self._xori, xmax, self._ncol)
        yi = np.linspace(self._yori, ymax, self._nrow)

        xi, yi = np.meshgrid(xi, yi, indexing='ij')

        sf = sampling

        self.logger.debug('ZONEPROP:')
        self.logger.debug(zoneprop)
        # compute per K layer (start on count 1)

        first = True
        for k in range(1, nlay + 1):

            if k < layer_minmax[0] or k > layer_minmax[1]:
                self.logger.info('SKIP LAYER {}'.format(k))
                continue
            else:
                self.logger.info('USE LAYER {}'.format(k))

            if usezoneprop:
                zonecopy = ma.copy(zoneprop[::sf, ::sf, k - 1:k])

                zzz = int(round(zonecopy.mean()))
                if zzz < zone_minmax[0] or zzz > zone_minmax[1]:
                    continue

            self.logger.info('Mapping for ' + str(k) + '...')

            xcopy = np.copy(xprop[::, ::, k - 1:k])
            ycopy = np.copy(yprop[::, ::, k - 1:k])
            zcopy = np.copy(mprop[::, ::, k - 1:k])
            dzcopy = np.copy(dzprop[::, ::, k - 1:k])

            if first:
                wsum = np.zeros((self._ncol, self._nrow))
                dzsum = np.zeros((self._ncol, self._nrow))
                first = False

            self.logger.debug(zcopy)

            xc = np.reshape(xcopy, -1, order='F')
            yc = np.reshape(ycopy, -1, order='F')
            zv = np.reshape(zcopy, -1, order='F')
            dz = np.reshape(dzcopy, -1, order='F')

            zvdz = zv * dz

            try:
                zvdzi = scipy.interpolate.griddata((xc[::sf], yc[::sf]),
                                                   zvdz[::sf],
                                                   (xi, yi),
                                                   method='linear',
                                                   fill_value=0.0)
            except ValueError:
                continue

            try:
                dzi = scipy.interpolate.griddata((xc[::sf], yc[::sf]),
                                                 dz[::sf],
                                                 (xi, yi),
                                                 method='linear',
                                                 fill_value=0.0)
            except ValueError:
                continue

            self.logger.debug(zvdzi.shape)

            wsum = wsum + zvdzi
            dzsum = dzsum + dzi

            self.logger.debug(wsum[0:20, 0:20])

        dzsum[dzsum == 0.0] = 1e-20
        vv = wsum / dzsum
        vv = ma.masked_invalid(vv)
        if truncate_le:
            vv = ma.masked_less(vv, truncate_le)

        self.values = vv
#        self._values=dzsum

    def quickplot(self, filename='default.png', title='QuickPlot',
                  infotext=None, xvalues=None, yvalues=None,
                  minmax=(None, None), xlabelrotation=None,
                  colortable='rainbow'):
        """Fast plot of maps using matplotlib.

        Args:
            filename (str): Name of plot file.
            title (str): Title of plot
            infotext (str): Additonal info on plot.
            minmax (tuple): Tuple of min and max values to be plotted. Note
                that values outside range will be set equal to range limits
            xlabelrotation (float): Rotation in degrees of X labels.
            colortable (str): Name of matplotlib or RMS file or XTGeo
                colortable. Default is matplotlib's 'rainbow'
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
                           colortable=colortable)

        mymap.savefig(filename)

    def distance_from_point(self, point=(0, 0), azimuth=0.0):
        """
        Make map values as horizontal distance from a point with azimuth
        direction.
        """

        x, y = point

        xtg_verbose_level = self._xtg.get_syslevel()

        # secure that carray is updated:
        self._update_cvalues()

        # call C routine
        ier = _cxtgeo.surf_get_dist_values(
            self._xori, self._xinc, self._yori, self._yinc, self._ncol,
            self._nrow, self._rotation, x, y, azimuth, self._cvalues, 0,
            xtg_verbose_level)

        if ier != 0:
            self.logger.error('Something went wrong...')
            raise ValueError

    def translate_coordinates(self, translate=(0, 0, 0)):
        """
        Translate a map in X Y VALUE space.

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

    # #########################################################################
    # PRIVATE STUFF
    # #########################################################################

    # =========================================================================
    # IMPORT routines
    # =========================================================================

    def _import_irap_binary(self, mfile):

        sdata = _regsurf_import.import_irap_binary(mfile)

        self._ncol = sdata['ncol']
        self._nrow = sdata['nrow']
        self._xori = sdata['xori']
        self._yori = sdata['yori']
        self._xinc = sdata['xinc']
        self._yinc = sdata['yinc']
        self._rotation = sdata['rotation']
        self._cvalues = sdata['cvalues']

        self._values = None

    # =========================================================================
    # EXPORT routines

    # this is temporary; shall be replaces with a cxtgeo method
    def _export_irap_ascii(self, mfile, exportmethod=2):
        """
        Private routine for export of surface to IRAP ASCII format
        """

        f = open(mfile, 'w')

        # this is only correct for nonrotated maps
        xmin = self.xori
        xmax = self.xori + self.xinc * self.ncol

        ymin = self.yori
        ymax = self.yori + self.yinc * self.nrow

        # print he IRAP ASCII header
        f.write('{0:10d}  {1:10d}  {2:10.2f} {3:10.2f}\n'.
                format(-996, self.nrow,
                       self.xinc,
                       self.yinc))

        f.write('{0:10.2f}      {1:10.2f}  {2:10.2f}  {3:10.2f}\n'.
                format(xmin, xmax, ymin, ymax))

        f.write('{0:10d}  {1:10.2f}      {2:10.2f}      {3:10.2f}\n'.
                format(self.ncol,
                       self.rotation,
                       self.xori,
                       self.yori))

        f.write('     0   0   0    0     0     0        0\n')

        # print the numpy part
        a = self.get_zval()  # 1D numpy, F order

        a[np.isnan(a)] = _cxtgeo.UNDEF_MAP_IRAP

        a[a > self._undef_limit] = _cxtgeo.UNDEF_MAP_IRAP

        if (exportmethod == 1):
            # savetxt gives only one column, but perhaps faster
            np.savetxt(f, a, fmt='%10.3f')
        else:
            i = 1
            for x in np.nditer(a):
                f.write('{0:14.4f}'.format(float(x)))
                i += 1
                if (i == 7):
                    i = 1
                    f.write('\n')

        f.close()

    def _export_irap_binary(self, mfile):

        # need to call the C function...
        _cxtgeo.xtg_verbose_file('NONE')

        # update numpy to c_array
        self._update_cvalues()

        xtg_verbose_level = self._xtg.get_syslevel()

        if xtg_verbose_level < 0:
            xtg_verbose_level = 0

        _cxtgeo.surf_export_irap_bin(mfile, self._ncol, self._nrow, self._xori,
                                     self._yori, self._xinc, self._yinc,
                                     self._rotation, self._cvalues, 0,
                                     xtg_verbose_level)

    # =========================================================================
    # ROXAR API routines

    def _import_horizon_roxapi(self, project, name, category,
                               realisation):
        """
        Import a Horizon surface via ROXAR API spec
        """
        import roxar

        if project is not None and isinstance(project, str):
            projectname = project
            with roxar.Project.open_import(projectname) as proj:
                try:
                    rox = proj.horizons[name][category].get_grid(realisation)
                    self._roxapi_horizon_to_xtgeo(rox)
                except KeyError as ke:
                    self.logger.error(ke)
        else:
            rox = project.horizons[name][category].get_grid(realisation)
            self._roxapi_horizon_to_xtgeo(rox)

    def _roxapi_horizon_to_xtgeo(self, rox):
        """
        Local function for tranforming surfaces from ROXAPI to XTGeo
        object.
        """
        self.logger.info('Surface from roxapi to xtgeo...')
        self._xori, self._yori = rox.origin
        self._ncol, self._nrow = rox.dimensions
        self._xinc, self._yinc = rox.increment
        self._rotation = rox.rotation
        self._values = rox.get_values()

    # =========================================================================
    # Helper methods, for internal usage
    # -------------------------------------------------------------------------
    # copy self (update) values from SWIG carray to numpy, 1D array

    def _update_values(self):
        n = self._ncol * self._nrow

        if self._cvalues is None and self._values is not None:
            return

        elif self._cvalues is None and self._values is None:
            self.logger.critical('_cvalues and _values is None in '
                                 '_update_values. STOP')
            sys.exit(9)

        x = _cxtgeo.swig_carr_to_numpy_1d(n, self._cvalues)

        x = np.reshape(x, (self._ncol, self._nrow), order='F')

        # make it masked
        x = ma.masked_greater(x, self._undef_limit)

        self._values = x

        self._delete_cvalues()

    # copy (update) values from numpy to SWIG, 1D array

    def _update_cvalues(self):
        self.logger.debug('Enter update cvalues method...')
        n = self._ncol * self._nrow

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
        x = ma.filled(self._values, self._undef)
        x = np.reshape(x, -1, order='F')

        self._cvalues = _cxtgeo.new_doublearray(n)

        _cxtgeo.swig_numpy_to_carr_1d(x, self._cvalues)
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


def main():
    import logging

    FORMAT = '%(name)s %(asctime)-15s =>  %(message)s'
    logging.basicConfig(format=FORMAT, level=logging.DEBUG)

    logger = logging.getLogger(__name__)

    logger.debug('Run')

    s = RegularSurface()

    print(s.ncol)


if __name__ == '__main__':
    if __package__ is None:
        print('NON PACKAGE MODE - TESTING ONLY')
        from os import path
        sys.path.append(path.dirname(path.dirname(path.abspath('.'))))
    else:
        pass

    main()
