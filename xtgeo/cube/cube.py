"""Module for a seismic (or whatever) cube."""

from __future__ import print_function
from __future__ import division

import numpy as np
import os.path
import logging
import tempfile
import sys

import cxtgeo.cxtgeo as _cxtgeo
from xtgeo.common import XTGeoDialog

from xtgeo.cube import _cube_import
from xtgeo.cube import _cube_export

#
# Note: The _values (3D array) may be C or F contiguous. As longs as it stays
# 3D it does not matter. However, when reshaped from a 1D array, or the
# other way, we need to know, as the file i/o (ie CXTGEO) is F contiguous!
#
# Either:
# _values (the numpy 32 bit float version)
# _cvalues (the pointer to C array)


class Cube(object):
    """Class for a (seismic) cube in the XTGeo framework.

    The values is a numpy, 3D Float 4 bytes.
    """

    def __init__(self,
                 xori=0.0,
                 yori=0.0,
                 zori=0.0,
                 nx=5,
                 ny=3,
                 nz=2,
                 xinc=25.0,
                 yinc=25.0,
                 zinc=4.0,
                 rotation=0.0,
                 yflip=1,
                 # the numbers here will be incremental in F order;
                 # hint turn head right and imagine the map...
                 # e.g. lower right corner is value 5
                 values=np.zeros((5, 3, 2), dtype=np.float32)
                 ):

        clsname = '{}.{}'.format(type(self).__module__, type(self).__name__)
        self.logger = logging.getLogger(clsname)
        self.logger.addHandler(logging.NullHandler())

        self._xtg = XTGeoDialog()

        self._xori = xori
        self._yori = yori
        self._nx = nx
        self._ny = ny
        self._nz = nz
        self._xinc = xinc
        self._yinc = yinc
        self._zinc = zinc
        self._rotation = rotation
        self._yflip = yflip
        self._values = values      # numpy of map values
        self._cvalues = None       # carray swig C pointer of map values

        self._undef = _cxtgeo.UNDEF
        self._undef_limit = _cxtgeo.UNDEF_LIMIT

    def __del__(self):
        if self._cvalues is not None:
            _cxtgeo.delete_floatarray(self._cvalues)

    # =========================================================================
    # Get and Set properties (tend to pythonic properties rather than
    # javaic get & set syntax)
    # =========================================================================

    @property
    def nx(self):
        """The NX (or N-Idir or column) number, as property."""
        return self._nx

    @nx.setter
    def nx(self, n):
        self.logger.warning('Cannot change nx')

    @property
    def ny(self):
        """The NY (or N-Jdir or row) number, as property."""
        return self._ny

    @ny.setter
    def ny(self, n):
        self.logger.warning('Cannot change ny')

    @property
    def nz(self):
        """The NZ (or N-Kdir or layer) number, as property."""
        return self._nz

    @nz.setter
    def nz(self, n):
        self.logger.warning('Cannot change nz')

    @property
    def xori(self):
        """The XORI (origin corner) coordinate, as property."""
        return self._xori

    @xori.setter
    def xori(self, val):
        self.logger.warning('Changing xori is risky')
        self._xori = val

    @property
    def yori(self):
        """The YORI (origin corner) coordinate, as property."""
        return self._yori

    @yori.setter
    def yori(self, val):
        self.logger.warning('Changing yori is risky')
        self._yori = val

    @property
    def zori(self):
        """The ZORI (origin corner) coordinate, as property."""
        return self._zori

    @zori.setter
    def zori(self, val):
        self.logger.warning('Changing zori is risky')
        self._zori = val

    @property
    def xinc(self):
        """The XINC (increment X) as property."""
        return self._xinc

    @xinc.setter
    def xinc(self, val):
        self.logger.warning('Changing xinc is risky')
        self._xinc = val


    @property
    def yinc(self):
        """The YINC (increment Y), as property."""
        return self._yinc

    @yinc.setter
    def yinc(self, val):
        self.logger.warning('Changing yinc is risky')
        self._yinc = val

    @property
    def zinc(self):
        """ The ZINC (increment Z), as property."""
        return self._zinc

    @zinc.setter
    def zinc(self, val):
        self.logger.warning('Changing zinc is risky')
        self._zinc = val

    @property
    def rotation(self):
        """The rotation (inline, anticlock X axis in degrees)."""
        return self._rotation

    @rotation.setter
    def rotation(self, val):
        self.logger.warning('Changing rotation is risky')
        self._rotation = val

    @property
    def yflip(self):
        """The YFLIP indicator, 1 is normal, -1 means Y flipped.

        YFLIP = 1 means a LEFT HANDED coordinate system with Z axis
        positive down, while inline follow East (X) and Xline follows
        North (Y), when rotation is zero.
        """
        return self._yflip

    @property
    def values(self):
        """The values, as a 3D numpy (nx, ny, nz), 4 byte float."""
        self._update_values()
        return self._values

    @values.setter
    def values(self, values):

        if not isinstance(values, np.ndarray):
            raise ValueError('Input is not a numpy array')

        vshape = values.shape

        if vshape != (self._nx, self._ny, self._nz):
            raise ValueError('Wrong dimensions of input numpy')

        self._values = values
        self._cvalues = None

    @property
    def cvalues(self):
        """The cvalues, as a SWIG pointer to C memory (float array)."""
        self._update_cvalues()
        return self._cvalues

    # =========================================================================
    # Import and export
    # =========================================================================

    def from_file(self, sfile, fformat='segy', engine=0):
        """Import cube data from file.

        Args:
            sfile (str): Filename
            fformat (str, optional): file format segy(default)/rms_regular
            engine (int, optional): For SEGY reader, 0 is builtin (default),
                1 is SEGYIO

        Example::
            >>> zz = Cube()
            >>> zz.from_file('some.segy')
        """
        if (os.path.isfile(sfile)):
            pass
        else:
            self.logger.critical('Not OK file')
            raise os.error

        if (fformat == 'rms_regular'):
            self._import_cube(sfile, sformat='rmsreg')
        elif (fformat == 'segy'):
            self._import_cube(sfile, sformat='segy', scanheadermode=False,
                              scantracemode=False, engine=engine)
        elif (fformat == 'storm'):
            self._import_cube(sfile, sformat='storm')
        else:
            self.logger.error('Invalid file format')

    def to_file(self, sfile, fformat='rms_regular'):
        """Export cube data to file.

        Args:
            sfile (str): Filename
            fformat (str, optional): file format rms_regular (default)

        Example::
            >>> zz = Cube()
            >>> zz.to_file('some.rmsreg')
        """

        if (fformat == 'rms_regular'):
            self._export_cube(sfile)
        else:
            self.logger.error('Invalid file format')

    def scan_segy_header(self, sfile, outfile=None):
        """Scan a SEGY file header and print info to screen or file.

        Args:
            sfile (str): Name of SEGY file
            outfile (str): File where store header info, if empty or None
                output goes to STDOUT.
        """

        flag = False
        # if outfile is none, it means that you want to print on STDOUT
        if outfile is None:
            f = tempfile.NamedTemporaryFile(delete=False)
            f.close()
            outfile = f.name
            self.logger.info('TMP file name is {}'.format(outfile))
            flag = True

        _cube_import.import_segy(sfile, scanheadermode=True,
                                 scantracemode=False,
                                 outfile=outfile)

        if flag:
            self.logger.info('OUTPUT to screen...')
            with open(outfile, 'r') as out:
                for line in out:
                    print(line, end='')
            os.remove(outfile)

    def scan_segy_traces(self, sfile, outfile=None):
        """Scan a SEGY file traces and print limits info to STDOUT or file.

        Args:
            sfile (str): Name of SEGY file
            outfile: File where store scanned trace info, if empty or None
                output goes to STDOUT.
        """

        flag = False
        # if outfile is none, it means that you want tp plot on STDOUT
        if outfile is None:
            f = tempfile.NamedTemporaryFile(delete=False)
            f.close()
            outfile = f.name
            self.logger.info('TMP file name is {}'.format(outfile))
            flag = True

        _cube_import.import_segy(sfile, scanheadermode=False,
                                 scantracemode=True, outfile=outfile)

        if flag:
            self.logger.info('OUTPUT to screen...')
            with open(outfile, 'r') as out:
                for line in out:
                    print(line, end='')
            os.remove(outfile)

    def swapaxes(self):
        """
        Swap the axes inline vs xline, keep origin
        """
        _cxtgeo.xtg_verbose_file('NONE')

        xtg_verbose_level = self._xtg.get_syslevel()

        nx = _cxtgeo.new_intpointer()
        ny = _cxtgeo.new_intpointer()
        yflip = _cxtgeo.new_intpointer()
        xinc = _cxtgeo.new_doublepointer()
        yinc = _cxtgeo.new_doublepointer()
        rota = _cxtgeo.new_doublepointer()

        _cxtgeo.intpointer_assign(nx, self._nx)
        _cxtgeo.intpointer_assign(ny, self._ny)
        _cxtgeo.intpointer_assign(yflip, self._yflip)

        _cxtgeo.doublepointer_assign(xinc, self._xinc)
        _cxtgeo.doublepointer_assign(yinc, self._yinc)
        _cxtgeo.doublepointer_assign(rota, self._rotation)

        self._update_cvalues()

        ier = _cxtgeo.cube_swapaxes(nx, ny, self.nz, yflip, self.xori, xinc,
                                    self.yori, yinc, rota, self._cvalues,
                                    0, xtg_verbose_level)
        if ier != 0:
            raise Exception

        self._nx = _cxtgeo.intpointer_value(nx)
        self._ny = _cxtgeo.intpointer_value(ny)
        self._yflip = _cxtgeo.intpointer_value(yflip)

        self._xinc = _cxtgeo.doublepointer_value(xinc)
        self._yinc = _cxtgeo.doublepointer_value(yinc)
        self._rotation = _cxtgeo.doublepointer_value(rota)

    # =========================================================================
    # PRIVATE METHODS
    # should not be applied outside the class;
    # =========================================================================

    def _import_cube(self, sfile, sformat='segy', scanheadermode=False,
                     scantracemode=False, outfile=None, engine=0):
        """Import Cube data from file and make instance."""

        if sformat == 'segy':
            if engine == 1:
                sdata = _cube_import.import_segy_io(sfile)
            else:
                sdata = _cube_import.import_segy(sfile,
                                                 scanheadermode=scanheadermode,
                                                 scantracemode=scantracemode,
                                                 outfile=outfile)
        elif sformat == 'storm':
            sdata = _cube_import.import_stormcube(sfile)

        self._nx = sdata['nx']
        self._ny = sdata['ny']
        self._nz = sdata['nz']
        self._xori = sdata['xori']
        self._xinc = sdata['xinc']
        self._yori = sdata['yori']
        self._yinc = sdata['yinc']
        self._zori = sdata['zori']
        self._zinc = sdata['zinc']
        self._rotation = sdata['rotation']
        self._cvalues = sdata['cvalues']
        self._values = sdata['values']
        self._yflip = sdata['yflip']

    def _export_cube(self, sfile):

        xtg_verbose_level = self._xtg.get_syslevel()

        self._update_cvalues()

        _cube_export.export_rmsreg(self.nx, self.ny, self.nz,
                                   self.xori, self.yori, self.zori,
                                   self.xinc, self.yinc, self.zinc,
                                   self.rotation, self.yflip,
                                   self.cvalues,
                                   sfile, xtg_verbose_level)

    # =========================================================================
    # Low Level methods
    # should not be applied outside the class
    # =========================================================================

    # -------------------------------------------------------------------------
    # Helper methods C <---> Numpy
    # -------------------------------------------------------------------------

    # copy (update) values from SWIG carray to numpy, 3D array, Fortran order
    def _update_values(self):

        if self._cvalues is None and self._values is None:
            self.logger.critical('Something is wrong. STOP!')
            sys.exit(9)

        elif self._cvalues is None:
            return self._values

        self.logger.debug('Updating numpy values...')
        n = self._nx * self._ny * self._nz
        x = _cxtgeo.swig_carr_to_numpy_f1d(n, self._cvalues)

        x = np.reshape(x, (self._nx, self._ny, self._nz), order='F')

        self._values = x
        self.logger.debug('Updating numpy values... done')

        xtype = self._values.dtype
        self.logger.info('VALUES of type {}'.format(xtype))

        # free the C values (to save memory)
        _cxtgeo.delete_floatarray(self._cvalues)

        self._cvalues = None

    # copy (update) values from numpy to SWIG, 1D array
    def _update_cvalues(self):
        self.logger.debug("Enter update cvalues method...")
        n = self._nx * self._ny * self._nz

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
        x = self._values.copy()
        x = np.reshape(x, -1, order='F')

        self._cvalues = _cxtgeo.new_floatarray(n)

        _cxtgeo.swig_numpy_to_carr_f1d(x, self._cvalues)
        self.logger.debug("Enter method... DONE")

        self._values = None
