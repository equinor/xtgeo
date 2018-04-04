# coding: utf-8
"""Module for a seismic (or whatever) cube."""
from __future__ import print_function
from __future__ import division

import numpy as np
import os.path
import tempfile
import sys
from warnings import warn

import cxtgeo.cxtgeo as _cxtgeo
from xtgeo.common import XTGeoDialog

from xtgeo.cube import _cube_import
from xtgeo.cube import _cube_export
from xtgeo.cube import _cube_utils
from xtgeo.cube import _cube_roxapi

xtg = XTGeoDialog()
logger = xtg.functionlogger(__name__)


# =============================================================================
# METHODS as wrappers to class init + import

def cube_from_file(mfile, fformat='guess'):
    """This makes an instance of a Cube directly from file import.

    Args:
        mfile (str): Name of file
        fformat (str): See :meth:`Cube.from_file`

    Example::

        import xtgeo
        mycube = xtgeo.cube_from_file('some_cube.segy')
    """

    obj = Cube()

    obj.from_file(mfile, fformat=fformat)

    return obj


def cube_from_roxar(project, name):
    """This makes an instance of a Cube directly from roxar input.

    Example::

        import xtgeo
        mycube = xtgeo.cube_from_roxar(project, 'DepthCube')

    """

    obj = Cube()

    obj.from_roxar(project, name)

    return obj


class Cube(object):
    """Class for a (seismic) cube in the XTGeo framework.

    The values is a numpy array, 3D Float 4 bytes. The
    array is (ncol, nrow, nlay) regular 3D numpy,
    with internal C ordering (nlay fastest).

    Default format for cube values are float32.

    The cube object instance can be initialized by either a
    spesification, or via a file import. The import is most
    common, and usually SEGY, but also other formats are
    available (or will be).

    Examples::

        from xtgeo.cube import Cube

        # a user defined cube:
        vals = np.zeros((40, 30, 10), dtype=np.float32)

        mycube = Cube(xori=100.0, yori=200.0, ncol=40, nrow=30,
                      nlay=10, rotation=30, values=vals)

        # or from a file
        mycube = Cube('somefile.segy')

    """

    def __init__(self, *args, **kwargs):
        """Initiate a Cube instance."""

        clsname = '{}.{}'.format(type(self).__module__, type(self).__name__)
        logger.info(clsname)

        if len(args) >= 1:
            fformat = kwargs.get('fformat', 'guess')
            self.from_file(args[0], fformat=fformat)
        else:
            self._xori = kwargs.get('xori', 0.0)
            self._yori = kwargs.get('yori', 0.0)
            self._zori = kwargs.get('zori', 0.0)
            self._ncol = kwargs.get('ncol', 5)
            self._nrow = kwargs.get('nrow', 3)
            self._nlay = kwargs.get('nlay', 2)
            self._xinc = kwargs.get('xinc', 25.0)
            self._yinc = kwargs.get('yinc', 25.0)
            self._zinc = kwargs.get('zinc', 2.0)
            self._yflip = kwargs.get('yflip', 1)
            self._values = kwargs.get('values', None)
            self._rotation = kwargs.get('rotation', 0.0)
            if self._values is None:
                vals = np.zeros((self._ncol, self._nrow, self._nlay),
                                dtype=np.float32)
                self._values = vals
            self._segyfile = kwargs.get('segyfile', None)

        self._undef = _cxtgeo.UNDEF
        self._undef_limit = _cxtgeo.UNDEF_LIMIT

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

    # =========================================================================
    # Get and Set properties (tend to pythonic properties rather than
    # javaic get & set syntax)
    # =========================================================================

    @property
    def ncol(self):
        """The NCOL (NX or I dir) number (read-only)."""
        return self._ncol

    @property
    def nx(self):
        """The NCOL (NX or I dir) number
        (deprecated, use ncol instead)."""

        warn('Use <ncol> instead of <nx>', DeprecationWarning)
        return self._ncol

    @property
    def nrow(self):
        """The NROW (NY or J dir) number (read-only)."""
        return self._nrow

    @property
    def ny(self):
        """The NROW (NY or J dir) number.
        (deprecated, use nrow instead)."""

        warn('Use <nrow> instead of <ny>', DeprecationWarning)
        return self._nrow

    @property
    def nlay(self):
        """The NLAY (or NZ or K dir) number (read-only)."""
        return self._nlay

    @property
    def nz(self):
        """The NLAY (NZ or K dir) number.
        (deprecated, use nlay instead)."""

        warn('Use <nlay> instead of <nz>', DeprecationWarning)
        return self._nlay

    @property
    def xori(self):
        """The XORI (origin corner) coordinate."""
        return self._xori

    @xori.setter
    def xori(self, val):
        logger.warning('Changing xori is risky')
        self._xori = val

    @property
    def yori(self):
        """The YORI (origin corner) coordinate."""
        return self._yori

    @yori.setter
    def yori(self, val):
        logger.warning('Changing yori is risky')
        self._yori = val

    @property
    def zori(self):
        """The ZORI (origin corner) coordinate."""
        return self._zori

    @zori.setter
    def zori(self, val):
        logger.warning('Changing zori is risky')
        self._zori = val

    @property
    def xinc(self):
        """The XINC (increment X) as property."""
        return self._xinc

    @xinc.setter
    def xinc(self, val):
        logger.warning('Changing xinc is risky')
        self._xinc = val

    @property
    def yinc(self):
        """The YINC (increment Y)."""
        return self._yinc

    @yinc.setter
    def yinc(self, val):
        logger.warning('Changing yinc is risky')
        self._yinc = val

    @property
    def zinc(self):
        """ The ZINC (increment Z)."""
        return self._zinc

    @zinc.setter
    def zinc(self, val):
        logger.warning('Changing zinc is risky')
        self._zinc = val

    @property
    def rotation(self):
        """The rotation (columns vector, anticlock from X axis in degrees)."""
        return self._rotation

    @rotation.setter
    def rotation(self, val):
        logger.warning('Changing rotation is risky')
        self._rotation = val

    @property
    def yflip(self):
        """The YFLIP indicator, 1 is normal, -1 means Y flipped.

        YFLIP = 1 means a LEFT HANDED coordinate system with Z axis
        positive down, while inline (col) follow East (X) and xline (rows)
        follows North (Y), when rotation is zero.
        """
        return self._yflip

    @property
    def segyfile(self):
        """The input segy file name (str), if any (or None) (read-only)."""
        return self._values

    @property
    def values(self):
        """The values, as a 3D numpy (ncol, nrow, nlay), 4 byte float."""
        return self._values

    @values.setter
    def values(self, values):

        if not isinstance(values, np.ndarray):
            raise ValueError('Input is not a numpy array')

        vshape = values.shape

        if vshape != (self._ncol, self._nrow, self._nlay):
            raise ValueError('Wrong dimensions of input numpy')

        values = np.ascontiguousarray(values, dtype=np.float32)

        self._values = values

    # =========================================================================
    # Copy etc
    # =========================================================================

    def copy(self):
        """Copy a xtgeo.cube.Cube object to another instance::

            >>> mycube2 = mycube.copy()

        """
        xcube = Cube(ncol=self.ncol, nrow=self.nrow, nlay=self.nlay,
                     xinc=self.xinc, yinc=self.yinc, zinc=self.zinc,
                     xori=self.xori, yori=self.yori, zori=self.zori,
                     yflip=self.yflip, segyfile=self.segyfile,
                     rotation=self.rotation, values=self.values.copy())
        return xcube

    def swapaxes(self):
        """Swap the axes inline vs xline, keep origin."""

        _cube_utils.swapaxes(self)

    # =========================================================================
    # Import and export
    # =========================================================================

    def from_file(self, sfile, fformat='guess', engine='segyio'):
        """Import cube data from file.

        If fformat is not provided, the file type will be guessed based
        on file extension (e.g. segy og sgy for SEGY format)

        Args:
            sfile (str): Filename
            fformat (str): file format guess/segy/rms_regular
                where 'guess' is default
            engine (str): For the SEGY reader, 'xtgeo' is builtin
                while 'segyio' uses the SEGYIO library (default)

        Raises:
            IOError if the file cannot be read (e.g. not found)

        Example::

            >>> zz = Cube()
            >>> zz.from_file('some.segy')


        """
        if (os.path.isfile(sfile)):
            pass
        else:
            logger.critical('Not OK file')
            raise IOError('Input file for Cube cannot be read')

        # work on file extension
        froot, fext = os.path.splitext(sfile)
        if fformat == 'guess':
            if len(fext) == 0:
                logger.critical('File extension missing. STOP')
                sys.exit(9)
            else:
                fformat = fext.lower().replace('.', '')

        if 'rms' in fformat.lower():
            _cube_import.import_rmsregular(self, sfile)
        elif (fformat.lower() == 'segy' or fformat.lower() == 'sgy'):
            _cube_import.import_segy_io(self, sfile)
        elif (fformat == 'storm'):
            _cube_import.import_stormcube(self, sfile)
        else:
            logger.error('Invalid or unknown file format')

    def to_file(self, sfile, fformat='segy'):
        """Export cube data to file.

        Args:
            sfile (str): Filename
            fformat (str, optional): file format 'segy' (default) or
                'rms_regular'

        Example::
            >>> zz = Cube('some.segy')
            >>> zz.to_file('some.rmsreg')
        """

        if (fformat == 'segy'):
            _cube_export.export_segy(self, sfile)
        elif (fformat == 'rms_regular'):
            _cube_export.export_rmsreg(self, sfile)
        else:
            logger.error('Invalid file format')

    def from_roxar(self, project, name):
        """Import (transfer) a cube from a Roxar seismic object to XTGeo.

        Args:
            project (str): Inside RMS use the magic 'project', else use
                path to RMS project
            name (str): Name of cube within RMS project.

        Raises:
            To be described...

        Example::

            zz = Cube()
            zz.from_roxar(project, 'truth_reek_seismic_depth_2000')

        """
        _cube_roxapi.import_cube_roxapi(self, project, name)

    def to_roxar(self, project, name, folder=None, domain='time',
                 compression=('wavelet', 5)):
        """Export (transfer) a cube from a XTGeo cube object to Roxar data.

        Args:
            project (str): Inside RMS use the magic 'project', else use
                path to RMS project
            name (str): Name of cube (seismic data) within RMS project.
            folder (str): Cubes may be stored under a folder in the tree.
            domain (str): 'time' (default) or 'depth'
            compression (tuple): description to come...

        Raises:
            To be described...

        Example::

            zz = xtgeo.cube.Cube('myfile.segy')
            zz.to_roxar(project, 'reek_cube')

            # alternative
            zz = xtgeo.cube_from_file('myfile.segy')
            zz.to_roxar(project, 'reek_cube')

        """
        _cube_roxapi.export_cube_roxapi(self, project, name)

    def scan_segy_traces(self, sfile, outfile=None):
        """Scan a SEGY file traces and print limits info to STDOUT or file.

        Args:
            sfile (str): Name of SEGY file
            outfile: File where store scanned trace info, if empty or None
                output goes to STDOUT.
        """

        oflag = False
        # if outfile is none, it means that you want to plot on STDOUT
        if outfile is None:
            f = tempfile.NamedTemporaryFile(delete=False)
            f.close()
            outfile = f.name
            logger.info('TMP file name is {}'.format(outfile))
            oflag = True

        _cube_import.import_segy(sfile, scanheadermode=False,
                                 scantracemode=True, outfile=outfile)

        if oflag:
            pass
            logger.info('OUTPUT to screen...')
            with open(outfile, 'r') as out:
                for line in out:
                    print(line.rstrip('\r\n'))
            os.remove(outfile)

    def scan_segy_header(self, sfile, outfile=None):
        """Scan a SEGY file header and print info to screen or file.

        Args:
            sfile (str): Name of SEGY file
            outfile (str): File where store header info, if empty or None
                output goes to screen (STDOUT).
        """

        flag = False
        # if outfile is none, it means that you want to print on STDOUT
        if outfile is None:
            f = tempfile.NamedTemporaryFile(delete=False)
            f.close()
            outfile = f.name
            logger.info('TMP file name is {}'.format(outfile))
            flag = True

        _cube_import.import_segy(sfile, scanheadermode=True,
                                 scantracemode=False,
                                 outfile=outfile)

        if flag:
            logger.info('OUTPUT to screen...')
            with open(outfile, 'r') as out:
                for line in out:
                    print(line.rstrip('\r\n'))
            os.remove(outfile)
