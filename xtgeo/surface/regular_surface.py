# -*- coding: utf-8 -*-
"""Module/class for regular surfaces with XTGeo."""

from __future__ import print_function

import os
import sys
import math
import numpy as np
import numpy.ma as ma
import scipy.interpolate
import os.path
from types import FunctionType


import cxtgeo.cxtgeo as _cxtgeo
from xtgeo.common import XTGeoDialog
from xtgeo.plot import Map
# from xtgeo._version import __version__
import logging


__author__ = 'Jan C. Rivenaes'

# =============================================================================
# Globals (always chack that these are same as in CLIB/CXTGEO)
# =============================================================================

# =============================================================================
# Class constructor
# Properties:
# _nx       =  number of rows (X, cycling fastest)
# _ny       =  number of columns (Y)
# _xori     =  X origin
# _yori     =  Y origin
# _xinc     =  X increment
# _yinc     =  Y increment
# _rotation =  Rotation in degrees, anti-clock relative to X axis (aka school)
# _values   =  Numpy 2D array of doubles, of shape (nx,ny)
# _cvalues  =  Pointer to C array (SWIG).
#
# Note: The _values (2D array) may be C or F contiguous. As longs as it stays
# 2D it does not matter. However, when reshaped from a 1D array, or the
# other way, we need to know, as the file i/o (ie CXTGEO) is F contiguous!
#
# =============================================================================


class RegularSurface(object):
    """
    Class for a regular surface in the xtgeo framework.

    The regular surface instance is usually initiated by
    import from file, but can also be made from scratch.
    The values can be accessed by the user as a 2D numpy float64 array
    (masked numpy).

    Attributes:
        nx: Integer for number of X direction columns
        ny: Integer for number of Y direction rows
        xori: X (East) origon coordinate
        yori: Y (North) origin coordinate
        xinc: X increment
        yinc: Y increment
        rotation: rotation in degrees, anticlock from X axis between 0, 360
        values: 2D masked numpy array of shape (nx,ny), F order

    Example:
        Initiate a class and import::

          from xtgeo.surface import RegularSurface
          x = RegularSurface()
          x.from_file('some.irap', fformat='irap_binary')

    """

    def __init__(self, *args, **kwargs):
        """
        The __init__ (constructor) method.

        The instance can be made either from file or by a spesification::

        >>> x1 = RegularSurface('somefilename')  # assume Irap binary
        >>> x2 = RegularSurface('somefilename', fformat='irap_ascii')
        >>> x3 = RegularSurface().from_file('somefilename',
                                            fformat='irap_binary')
        >>> x4 = RegularSurface()
        >>> x4.from_file('somefilename', fformat='irap_binary')
        >>> x5 = RegularSurface(nx=20, ny=10, xori=2000.0, yori=2000.0,
                                rotation=0.0, xinc=25.0, yinc=25.0,
                                values=np.zeros((20,10)))

        Args:
            xori (float): Origin of grid X (East) coordinate
            yori (float): Origin of grid Y (North) coordinate
            xinc (float): Increment in X
            yinc (float): Increment in Y
            nx (int): Number of columns, X
            ny (int): Number of rows, Y
            rotation (float): Rotation angle (deg.), from X axis, anti-clock
            values (ndarray): 2D numpy of shape (nx,ny))

        """

        clsname = "{}.{}".format(type(self).__module__, type(self).__name__)
        self.logger = logging.getLogger(clsname)
        self.logger.addHandler(logging.NullHandler())

        self._xtg = XTGeoDialog()

        self._undef = _cxtgeo.UNDEF
        self._undef_limit = _cxtgeo.UNDEF_LIMIT
        self._cvalues = None     # carray swig C pointer of map values

        if args:
            # make instance from file import
            mfile = args[0]
            fformat = kwargs.get('fformat', 'irap_binary')
            self.from_file(mfile, fformat=fformat)

        else:
            # make instance by kw spesification
            self._xori = kwargs.get('xori', 0.0)
            self._yori = kwargs.get('yori', 0.0)
            self._nx = kwargs.get('nx', 5)
            self._ny = kwargs.get('ny', 3)
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
                                        dtype=np.double)
            else:
                self._values = values

                if self._check_shape_ok(self._values) is False:
                    self.logger.error("Wrong dimension of values")
                    raise ValueError

            # make it masked
            self._values = ma.masked_greater(self._values, self._undef_limit)

        # _nsurfaces += 1

        if self._values is not None:
            self.logger.debug("Shape of value: and values")
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

    def from_file(self, mfile, fformat="irap_binary"):
        """
        Import surface (regular map) from file.

        Args:
            mfile (str): Name of file
            fformat (str): File format, irap_binary is currently supported

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
            self.logger.critical("Not OK file")
            raise os.error

        if (fformat is None or fformat == "irap_binary"):
            self._import_irap_binary(mfile)
        else:
            self.logger.error("Invalid file format")

        return self

    def to_file(self, mfile, fformat="irap_binary"):
        """
        Export surface (regular map) to file

        Args:
            mfile (str): Name of file
            fformat (str): File format, irap_binary/irap_classic

        Example::

            >>> x = RegularSurface()
            >>> x.from_file('myfile.x', fformat = 'irap_ascii')
            >>> x.values = x.values + 300
            >>> x.to_file('myfile2.x', fformat = 'irap_ascii')

        """

        self.logger.debug("Enter method...")
        self.logger.info("Export to file...")
        if (fformat == "irap_ascii"):
            self._export_irap_ascii(mfile)
        elif (fformat == "irap_binary"):
            self._export_irap_binary(mfile)
        else:
            self.logger.critical("Invalid file format")

    def from_roxar(self, project, type="horizon", name=None, category=None,
                   realisation=0):
        """
        Load a surface from a Roxar RMS project.

        Args:
            project (str): Name of project (as folder) if outside RMS, og just
                use the magig 'project' if within RMS.
            type (str): 'horizon', 'clipboard', etc
            name (str): Name of surface/map
            category (str): For horizon only: for example "DS_extracted"
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
            self.logger.error("Need to spesify name and categori for "
                              "horizon")
        elif type != 'horizon':
            self.logger.error("Only horizon type is supported so far")
            raise Exception

        self._import_horizon_roxapi(project, name, category,
                                    realisation)

        return self

# =============================================================================
# Get and Set properties (tend to pythonic properties rather than javaic get
# & set syntax)
# =============================================================================

    @property
    def nx(self):
        """
        The NX (or N-Idir) number, as property.
        """
        self.logger.debug("Enter method...")
        return self._nx

    @nx.setter
    def nx(self, n):
        self.logger.debug("Enter method...")
        self.logger.warning("Cannot change nx")

    @property
    def ny(self):
        """
        The NY (or N-Jdir) number, as property.
        """
        self.logger.debug("Enter method...")
        return self._ny

    @ny.setter
    def ny(self, n):
        self.logger.warning("Cannot change ny")

    @property
    def rotation(self):
        """
        The rotation, anticlock from X axis, in degrees [0..360].
        """
        return self._rotation

    @rotation.setter
    def rotation(self, rota):
        if rota >= 0 and rota < 360:
            self._rotation = rota
        else:
            raise ValueError

    @property
    def xinc(self):
        """
        The X increment (or I dir increment)
        """
        self.logger.debug("Enter method...")
        return self._xinc

    @property
    def yinc(self):
        """
        The Y increment (or I dir increment)
        """
        self.logger.debug("Enter method...")
        return self._yinc

    @property
    def xori(self):
        """
        The X coordinate origin of the map (can be modified)
        """
        return self._xori

    @xori.setter
    def xori(self, xnew):
        self._xori = xnew

    @property
    def yori(self):
        """
        The Y coordinate origin of the map (can be modified)
        """
        return self._yori

    @yori.setter
    def yori(self, ynew):
        self._yori = ynew

    @property
    def values(self):
        """
        The map values, as 2D masked numpy (float64) of shape (nx, ny)
        """
        self.logger.debug("Enter method to get values...")
        self._update_values()
        return self._values

    @values.setter
    def values(self, values):
        self.logger.debug("Enter method...")

        if (isinstance(values, np.ndarray) and
                not isinstance(values, ma.MaskedArray)):

            values = ma.array(values)

        if self._check_shape_ok(values) is False:
            raise ValueError

        self._values = values
        self._cvalues = None

        self.logger.debug("Values shape: {}".format(self._values.shape))
        self.logger.debug("Flags: {}".format(self._values.flags.c_contiguous))

    @property
    def cvalues(self):
        """
        The map values, as 1D C pointer i.e. a reference only (Fortran order).
        """
        self.logger.debug("Enter method...")
        self._update_cvalues()
        return self._cvalues

    @cvalues.setter
    def cvalues(self, cvalues):
        self.logger.warn("Not possible!")

    @property
    def undef(self):
        """
        Returns the undef value, to be used when in the get_zval method
        """
        self.logger.debug("Enter method...")
        return self._undef

    @property
    def undef_limit(self):
        """
        Returns the undef_limit value, to be used when in the get_zval method
        """
        self.logger.debug("Enter method...")
        return self._undef_limit

    def get_zval(self):
        """
        Get an an 1D, numpy array of the map values (not masked).

        Note that undef values are very large numbers (see undef property).
        Also, this will reorder a 2D values array to column fastest, i.e.
        get stuff into Fortran order.
        """

        self._update_values()

        self.logger.debug("Enter method...")
        self.logger.debug("Shape: {}".format(self._values.shape))

        if self._check_shape_ok(self._values) is False:
            raise ValueError

        # unmask the self._values numpy array, by filling the masked
        # values with undef value
        self.logger.debug("Fill the masked...")
        x = ma.filled(self._values, self._undef)

        # make it 1D (Fortran order)
        self.logger.debug("1D")

        x = np.reshape(x, -1, order='F')

        self.logger.debug("1D done {}".format(x.shape))

        return x

    def set_zval(self, x):
        """
        Set a 1D (unmasked) numpy array. The numpy array must be
        in Fortran order (i columns (nx) fastest).

        Will convert it to a 2D masked array internally.
        """

        self._update_values()

        # not sure if this is right always?...
        x = np.reshape(x, (self._nx, self._ny), order='F')

        # make it masked
        x = ma.masked_greater(x, self._undef_limit)

        self._values = x

    def get_rotation(self):
        """
        Returns the surface roation, in degrees, from X, anti-clock.
        """
        return self._rotation

    def get_nx(self):
        """ Same as nx (for backward compatibility) """
        return self._nx

    def get_ny(self):
        """ Same as ny (for backward compatibility) """
        return self._ny

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
        """
        Copy a xtgeo.surface.RegularSurface object to another instance::

            >>> mymapcopy = mymap.copy()

        """
        x = RegularSurface(nx=self.nx, ny=self.ny, xinc=self.xinc,
                           yinc=self.yinc, xori=self.xori, yori=self.yori,
                           rotation=self.rotation, values=self.values)
        return x

    def similarity_index(self, other):
        """
        Report the degree of similarity between two maps, by comparing mean.

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
        """
        Check that two object has the same topology, i.e. map defintions such
        as origin, dimensions, number of defined cells...

        Args:
            other (surface object): The other surface to compare with

        Returns:
            True of same topology, False if not
        """
        if (self.nx != other.nx or self.ny != other.ny or
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
        """
        Return the map value given a X Y point.

        Args:
            point (float tuple): Position of X and Y coordinate
        Returns:
            The map value (interpolated). None if XY is outside defined map

        Example::
            mvalue = map.get_value_from_xy(point=(539291.12, 6788228.2))

        """
        x, y = point

        self.logger.debug('Enter value_from_cy')

        xtg_verbose_level = self._xtg.get_syslevel()

        # secure that carray is updated before SWIG/C:
        self._update_cvalues()

        # call C routine
        z = _cxtgeo.surf_get_z_from_xy(float(x), float(y), self._nx, self._ny,
                                       self._xori, self._yori, self._xinc,
                                       self._yinc, self._rotation,
                                       self._cvalues, xtg_verbose_level)

        if z > self._undef_limit:
            return None

        return z

# =============================================================================
# Interacion with a cube
# =============================================================================

    def slice_cube(self, cube, zsurf=None, sampling=0, mask=True):
        """
        Slice the cube and update the instance surface to it samples cube
        values.

        Args:
            cube (object): Instance of a Cube()
            zsurf (surface object): Instance of a depth (or time) map.
                If None, then the surface instance itself is used a slice
                criteria. Note that zsurf must have same map defs as the
                surface instance.
            sampling (int): 0 for nearest node (default), 1 for trilinear
                interpolation.
            mask (bool): If True (default), then the map values outside
                the cube will be undef.

        Example::
            cube = Cube()
            cube.from_file("some.segy")
            surf = RegularSurface()
            surf.from_file("s.gri")
            # update surf to sample cube values:
            surf.slice_cube(cube)

        Raises:
            Exception if maps have different definitions (topology)
        """

        xtg_verbose_level = self._xtg.get_syslevel()

        if zsurf is not None:
            other = zsurf
        else:
            other = self.copy()

        if not self.compare_topology(other):
            raise Exception

        if mask:
            opt2 = 0
        else:
            opt2 = 1

        self._update_cvalues()
        other._update_cvalues()

        self.logger.debug("Running method from C..:")
        istat = _cxtgeo.surf_slice_cube(cube.nx, cube.ny, cube.nz, cube.xori,
                                        cube.xinc, cube.yori, cube.yinc,
                                        cube.zori, cube.zinc, cube.rotation,
                                        cube.yflip, cube.cvalues,
                                        self.nx, self.ny, self.xori, self.xinc,
                                        self.yori, self.yinc,
                                        self.rotation, other.cvalues,
                                        self.cvalues, sampling, opt2,
                                        xtg_verbose_level)

        self.logger.debug("Running method from C.. done")

        if istat != 0:
            self.logger.warning("Seem to be rotten")

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
                                             self.nx, self.ny, self.xori,
                                             self.yori, self.xinc, self.yinc,
                                             self.rotation, self.cvalues,
                                             xtg_verbose_level)

        if istat != 0:
            self.logger.warning("Seem to be rotten")

        zarr = self._convert_carr_double_np(czarr, nlen=nvec)

        xyfence[:, 2] = zarr
        xyfence = ma.masked_greater(xyfence, self._undef_limit)
        xyfence = ma.mask_rows(xyfence)

        return xyfence

    def hc_thickness_from_3dprops(self, xprop=None, yprop=None,
                                  hcpfzprop=None, zoneprop=None,
                                  zone_minmax=None, layer_minmax=None):
        """
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
            zone_minmax (array): (optional) 2 element list indicating start
                and stop zonation (both start and end spec are included)
            layer_minmax (array): (optional) 2 element list indicating start
                and stop grid layers (both start and end spec are included)

        Returns:
            True if operation went OK (but check result!), False if not

        """

        self.logger.debug('Enter hc_thickness_from_3dprops')

        ncol, nrow, nlay = xprop.shape

        if layer_minmax is None:
            layer_minmax = (1, 99999)

        if zone_minmax is None:
            zone_minmax = (1, 99999)

        self.logger.debug('Layout is {} {} {}'.format(ncol, nrow, nlay))

        # do not allow rotation...
        if self._rotation < -0.1 or self._rotation > 0.1:
            self.logger.error("Cannut use rotated maps. Return")
            return False

        xmax = self._xori + self._xinc * self._nx
        ymax = self._yori + self._yinc * self._ny
        xi = np.linspace(self._xori, xmax, self._nx)
        yi = np.linspace(self._yori, ymax, self._ny)

        xi, yi = np.meshgrid(xi, yi, indexing='ij')

        # filter and compute per K layer (start count on 1)
        for k in range(1, nlay + 1):

            if k < layer_minmax[0] or k > layer_minmax[1]:
                continue

            zonecopy = np.copy(zoneprop[:, :, k - 1:k])

            if zone_minmax[0] > zonecopy.mean() > zone_minmax[1]:
                continue

            self.logger.info('Mapping for ' + str(k) + '...')

            # get slices per layer of relevant props
            xcopy = np.copy(xprop[:, :, k - 1:k])
            ycopy = np.copy(yprop[:, :, k - 1:k])
            zcopy = np.copy(hcpfzprop[:, :, k - 1:k])

            if k == layer_minmax[0]:
                zsum = np.zeros((self._nx, self._ny))

            propsum = zcopy.sum()
            if (propsum < 1e-12):
                self.logger.debug("Skip layer K = {}".format(k))
                continue
            else:
                self.logger.debug("Z property sum is {}".format(propsum))

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
            self.logger.debug('Repr XI?: \n{}'.format(repr(xi)))
            self.logger.debug('Repr XPROP?: \n{}'.format(repr(xprop)))
            self.logger.debug('Repr HC?: \n{}'.format(repr(hcpfzprop)))

            # need to make arrays 1D
            self.logger.debug("Reshape and filter ...")
            x = np.reshape(xcopy, -1, order='F')
            y = np.reshape(ycopy, -1, order='F')
            z = np.reshape(zcopy, -1, order='F')

            xc = np.copy(x)

            x = x[xc < self._undef_limit]
            y = y[xc < self._undef_limit]
            z = z[xc < self._undef_limit]

            self.logger.debug("Reshape and filter ... done")

            self.logger.debug("Map ...")

            try:
                zi = scipy.interpolate.griddata((x, y), z, (xi, yi),
                                                method='linear',
                                                fill_value=0.0)
            except ValueError:
                continue

            self.logger.info("ZI shape is {}".format(zi.shape))
            self.logger.debug("Map ... done")

            zsum = zsum + zi
            self.logger.info("Sum of HCPB layer is {}".format(zsum.mean()))

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

        self.logger.info("Layer from: {}".format(layer_minmax[0]))
        self.logger.info("Layer to: {}".format(layer_minmax[1]))
        self.logger.debug('Layout is {} {} {}'.format(ncol, nrow, nlay))

        self.logger.info("Zone from: {}".format(zone_minmax[0]))
        self.logger.info("Zone to: {}".format(zone_minmax[1]))
        self.logger.info("Zone is :")
        self.logger.info(zoneprop)

        # do not allow rotation...
        if self._rotation < -0.1 or self._rotation > 0.1:
            self.logger.error("Cannut use rotated maps. Return")
            return

        xmax = self._xori + self._xinc * self._nx
        ymax = self._yori + self._yinc * self._ny
        xi = np.linspace(self._xori, xmax, self._nx)
        yi = np.linspace(self._yori, ymax, self._ny)

        xi, yi = np.meshgrid(xi, yi, indexing='ij')

        sf = sampling

        self.logger.debug("ZONEPROP:")
        self.logger.debug(zoneprop)
        # compute per K layer (start on count 1)

        first = True
        for k in range(1, nlay + 1):

            if k < layer_minmax[0] or k > layer_minmax[1]:
                self.logger.info("SKIP LAYER {}".format(k))
                continue
            else:
                self.logger.info("USE LAYER {}".format(k))

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
                wsum = np.zeros((self._nx, self._ny))
                dzsum = np.zeros((self._nx, self._ny))
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

    def quickplot(self, filename="default.png", title='QuickPlot',
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
            self._xori, self._xinc, self._yori, self._yinc, self._nx,
            self._ny, self._rotation, x, y, azimuth, self._cvalues, 0,
            xtg_verbose_level)

        if ier != 0:
            self.logger.error("Something went wrong...")
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

# #############################################################################
# PRIVATE STUFF
# #############################################################################

    # =========================================================================
    # IMPORT routines
    # =========================================================================

    def _import_irap_binary(self, mfile):

        self.logger.debug("Enter function...")
        # need to call the C function...
        _cxtgeo.xtg_verbose_file("NONE")

        xtg_verbose_level = self._xtg.get_syslevel()

        if xtg_verbose_level < 0:
            xtg_verbose_level = 0

        ptr_mx = _cxtgeo.new_intpointer()
        ptr_my = _cxtgeo.new_intpointer()
        ptr_xori = _cxtgeo.new_doublepointer()
        ptr_yori = _cxtgeo.new_doublepointer()
        ptr_xinc = _cxtgeo.new_doublepointer()
        ptr_yinc = _cxtgeo.new_doublepointer()
        ptr_rot = _cxtgeo.new_doublepointer()
        ptr_dum = _cxtgeo.new_doublepointer()
        ptr_ndef = _cxtgeo.new_intpointer()

        if (os.path.exists(mfile)):
            self.logger.info("File is ok")
        else:
            self.logger.error("No such file!")

        # read with mode 0, to get mx my
        _cxtgeo.surf_import_irap_bin(mfile, 0, ptr_mx, ptr_my, ptr_xori,
                                     ptr_yori, ptr_xinc, ptr_yinc, ptr_rot,
                                     ptr_dum, ptr_ndef, 0, xtg_verbose_level)

        mx = _cxtgeo.intpointer_value(ptr_mx)
        my = _cxtgeo.intpointer_value(ptr_my)

        self._cvalues = _cxtgeo.new_doublearray(mx * my)

        # read with mode 1, to get the map
        _cxtgeo.surf_import_irap_bin(mfile, 1, ptr_mx, ptr_my, ptr_xori,
                                     ptr_yori, ptr_xinc, ptr_yinc, ptr_rot,
                                     self._cvalues, ptr_ndef, 0,
                                     xtg_verbose_level)

        self._nx = _cxtgeo.intpointer_value(ptr_mx)
        self._ny = _cxtgeo.intpointer_value(ptr_my)
        self._xori = _cxtgeo.doublepointer_value(ptr_xori)
        self._yori = _cxtgeo.doublepointer_value(ptr_yori)
        self._xinc = _cxtgeo.doublepointer_value(ptr_xinc)
        self._yinc = _cxtgeo.doublepointer_value(ptr_yinc)
        self._rotation = _cxtgeo.doublepointer_value(ptr_rot)

        # convert carray pointer to numpy
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
        xmax = self.xori + self.xinc * self.nx

        ymin = self.yori
        ymax = self.yori + self.yinc * self.ny

        # print he IRAP ASCII header
        f.write('{0:10d}  {1:10d}  {2:10.2f} {3:10.2f}\n'.
                format(-996, self.ny,
                       self.xinc,
                       self.yinc))

        f.write('{0:10.2f}      {1:10.2f}  {2:10.2f}  {3:10.2f}\n'.
                format(xmin, xmax, ymin, ymax))

        f.write('{0:10d}  {1:10.2f}      {2:10.2f}      {3:10.2f}\n'.
                format(self.nx,
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
            np.savetxt(f, a, fmt="%10.3f")
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
        _cxtgeo.xtg_verbose_file("NONE")

        # update numpy to c_array
        self._update_cvalues()

        xtg_verbose_level = self._xtg.get_syslevel()

        if xtg_verbose_level < 0:
            xtg_verbose_level = 0

        _cxtgeo.surf_export_irap_bin(mfile, self._nx, self._ny, self._xori,
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
        self.logger.info("Surface from roxapi to xtgeo...")
        self._xori, self._yori = rox.origin
        self._nx, self._ny = rox.dimensions
        self._xinc, self._yinc = rox.increment
        self._rotation = rox.rotation
        self._values = rox.get_values()

    # =========================================================================
    # Helper methods, for internal usage
    # -------------------------------------------------------------------------
    # copy self (update) values from SWIG carray to numpy, 1D array

    def _update_values(self):
        n = self._nx * self._ny

        if self._cvalues is None and self._values is not None:
            return

        elif self._cvalues is None and self._values is None:
            self.logger.critical("_cvalues and _values is None in "
                                 "_update_values. STOP")
            sys.exit(9)

        x = _cxtgeo.swig_carr_to_numpy_1d(n, self._cvalues)

        x = np.reshape(x, (self._nx, self._ny), order='F')

        # make it masked
        x = ma.masked_greater(x, self._undef_limit)

        self._values = x

        self._delete_cvalues()

    # copy (update) values from numpy to SWIG, 1D array

    def _update_cvalues(self):
        self.logger.debug("Enter update cvalues method...")
        n = self._nx * self._ny

        if self._values is None and self._cvalues is not None:
            self.logger.debug("CVALUES unchanged")
            return

        elif self._cvalues is None and self._values is None:
            self.logger.critical("_cvalues and _values is None in "
                                 "_update_cvalues. STOP")
            sys.exit(9)

        elif self._cvalues is not None and self._values is None:
            self.logger.critical("_cvalues and _values are both present in "
                                 "_update_cvalues. STOP")
            sys.exit(9)

        # make a 1D F order numpy array, and update C array
        x = ma.filled(self._values, self._undef)
        x = np.reshape(x, -1, order='F')

        self._cvalues = _cxtgeo.new_doublearray(n)

        _cxtgeo.swig_numpy_to_carr_1d(x, self._cvalues)
        self.logger.debug("Enter method... DONE")

        self._values = None

    def _delete_cvalues(self):
        self.logger.debug("Enter delete cvalues values method...")

        if self._cvalues is not None:
            _cxtgeo.delete_doublearray(self._cvalues)

        self._cvalues = None
        self.logger.debug("Enter method... DONE")

    # check if values shape is OK (return True or False)

    def _check_shape_ok(self, values):
        (nx, ny) = values.shape
        if nx != self._nx or ny != self._ny:
            self.logger.error("Wrong shape: Dimens of values {} {} vs {} {}"
                              .format(nx, ny, self._nx, self._ny))
            return False
        return True

    def _convert_np_carr_double(self, np_array, nlen):
        """
        Convert numpy 1D array to C array, assuming double type
        """
        carr = _cxtgeo.new_doublearray(nlen)

        _cxtgeo.swig_numpy_to_carr_1d(np_array, carr)

        return carr

    def _convert_carr_double_np(self, carray, nlen=None):
        """
        Convert a C array to numpy, assuming double type.
        """
        if nlen is None:
            nlen = len(self._df.index)

        nparray = _cxtgeo.swig_carr_to_numpy_1d(nlen, carray)

        return nparray


def main():
    import logging

    FORMAT = '%(name)s %(asctime)-15s =>  %(message)s'
    logging.basicConfig(format=FORMAT, level=logging.DEBUG)

    logger = logging.getLogger(__name__)

    logger.debug("Run")

    s = RegularSurface()

    print(s.nx)


if __name__ == '__main__':
    if __package__ is None:
        print("NON PACKAGE MODE - TESTING ONLY")
        from os import path
        sys.path.append(path.dirname(path.dirname(path.abspath('.'))))
    else:
        pass

    main()
