# -*- coding: utf-8 -*-
"""Module for a 3D grid property.

The grid property instances may or may not belong to a grid geometry
object. Normally the instance is created when importing a grid
property from file, but it can also be created directly, as e.g.:

poro = GridProperty(ncol=233, nrow=122, nlay=32)

The grid property values (instance.values) by themselves are 3D masked numpy
as either float64 (double) or int32 (if discrete), and undefined cells are
masked. The array order is now C_CONTIGUOUS. (i.e. not in Eclipse manner).
A 1D view (C order) is achieved by the values1d property, e.g.::

  poronumpy = poro.values1d

"""
from __future__ import print_function, absolute_import

import sys
import numpy as np
import numpy.ma as ma
import os.path

import cxtgeo.cxtgeo as _cxtgeo

from xtgeo.common.exceptions import DateNotFoundError, KeywordFoundNoDateError
from xtgeo.common.exceptions import KeywordNotFoundError

from xtgeo.common import XTGeoDialog
from xtgeo.grid3d import Grid3D
from xtgeo.grid3d import _gridprop_op1
from xtgeo.grid3d import _gridprop_import
from xtgeo.grid3d import _gridprop_import_obsolete
from xtgeo.grid3d import _gridprop_roxapi
from xtgeo.grid3d import _gridprop_export
from ._gridprops_io import _get_fhandle

xtg = XTGeoDialog()
logger = xtg.functionlogger(__name__)


class GridProperty(Grid3D):
    """Class for a single 3D grid property.

    An instance may or may not belong to a grid (geometry) object. E.g. for
    for ROFF, ncol, nrow, nlay are given in the file. Note that in such
    cases the undef cells may (or may not) are flagged as a very high number.
    """

    def __init__(self, *args, **kwargs):
        """The __init__ (constructor) method, can be ran empty, or with two
        variant of input arguments.

        Args:
            ncol (int): Number of columns.
            nrow (int): Number of rows.
            nlay (int): Number of layers.
            values (numpy): A 3D masked numpy of shape (ncol, nrow, nlay).
            name (str): Name of property.
            discrete (bool): True if discrete property
                (default is false).

        Alternatively, the same arguments as the from_file() method
        can be used.

        Returns:
            A GridProperty object instance.

        Raises:
            RuntimeError if something goes wrong (e.g. file not found)

        Examples::

            myprop = GridProperty()
            myprop.from_file('emerald.roff', name='PORO')

            # or

            myprop = GridProperty(ncol=12, nrow=17, nlay=10,
                                  values=ma.ones((12, 17, 10),
                                  dtype=np.float64),
                                  discrete=True, name='MyValue')

            # or

            myprop = GridProperty('emerald.roff', name='PORO')

        """

        clsname = '{}.{}'.format(type(self).__module__, type(self).__name__)
        logger.info(clsname)

        ncol = kwargs.get('ncol', 5)
        nrow = kwargs.get('nrow', 12)
        nlay = kwargs.get('nlay', 2)
        values = kwargs.get('values', None)
        name = kwargs.get('name', 'unknown')
        date = kwargs.get('date', None)
        discrete = kwargs.get('discrete', False)
        grid = kwargs.get('grid', None)

        self._ncol = ncol
        self._nrow = nrow
        self._nlay = nlay

        self._grid = grid           # grid geometry object

        self._isdiscrete = discrete

        testmask = False
        if values is None:
            values = ma.zeros((ncol, nrow, nlay))
            testmask = True

        if values.shape != (ncol, nrow, nlay):
            values = values.reshape((ncol, nrow, nlay), order='C')

        if not isinstance(values, ma.MaskedArray):
            values = ma.array(values)

        self._values = values       # numpy version of properties (as 3D array)

        self._dtype = values.dtype

        self._name = name           # property name
        self._date = None           # property may have an assosiated date
        self._codes = {}            # code dictionary (for discrete)
        self._ncodes = 1            # Number of codes in dictionary

        self._undef = _cxtgeo.UNDEF
        self._undef_limit = _cxtgeo.UNDEF_LIMIT

        self._undef_i = _cxtgeo.UNDEF_INT
        self._undef_ilimit = _cxtgeo.UNDEF_INT_LIMIT

        self._roxorigin = False  # true if the object comes from the ROXAPI

        if self._isdiscrete:
            self._values = self._values.astype(np.int32)
            self._dtype = self._values.dtype
            self._roxar_dtype = np.int8

        if testmask:
            # make some undef cells (for test)
            self._values[0:4] = self._undef
            # make it masked
            self._values = ma.masked_greater(self._values, self._undef_limit)

        if len(args) == 1:
            # make instance through file import

            logger.debug('Import from file...')
            fformat = kwargs.get('fformat', 'guess')
            name = kwargs.get('name', 'unknown')
            date = kwargs.get('date', None)
            grid = kwargs.get('grid', None)
            apiv = kwargs.get('apiversion', 2)
            self.from_file(args[0], fformat=fformat, name=name,
                           grid=grid, date=date, apiversion=apiv)

    # =========================================================================
    # Properties
    # =========================================================================

    # -------------------------------------------------------------------------
    @property
    def name(self):
        """Returns or rename the property name."""
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    # -------------------------------------------------------------------------
    @property
    def grid(self):
        """Return the XTGeo grid geometry object (read only)"""
        return self._grid

    @grid.setter
    def grid(self, thegrid):
        # need many checks to get this to work
        pass
        # if isinstance(thegrid, Grid):
        #     self._grid = thegrid
        # else:
        #     raise RuntimeWarning('The given grid is not a Grid instance')

    # -------------------------------------------------------------------------
    @property
    def isdiscrete(self):
        """Return True if property is discrete"""
        return self._isdiscrete

    # -------------------------------------------------------------------------
    @property
    def dtype(self):
        """Return the values numpy dtype"""
        return str(self._values.dtype)

    # -------------------------------------------------------------------------
    @property
    def date(self):
        """Returns or rename the property date on YYYYMMDD numerical format."""
        return self._date

    @date.setter
    def date(self, date):
        self._date = date

    # -------------------------------------------------------------------------
    @property
    def codes(self):
        """The property codes as a dictionary."""
        return self._codes

    @codes.setter
    def codes(self, cdict):
        self._codes = cdict.copy()

    @property
    def ncodes(self):
        """Number of codes"""
        return len(self._codes)

    # -------------------------------------------------------------------------
    @property
    def values(self):
        """ Return or set the grid property as a masked 1D numpy array"""
        return self._values

    @values.setter
    def values(self, values):
        if isinstance(values, np.ndarray) and\
           not isinstance(values, ma.MaskedArray):

            values = ma.array(values)
            values = values.reshape((self._ncol, self._nrow, self._nlay))

        self._values = values

        logger.debug('Values shape: {}'.format(self._values.shape))
        logger.debug('Flags: {}'.format(self._values.flags.c_contiguous))

    # @property
    # def npvalues(self):
    #     """ Return an ordinary unmasked 1D numpy array"""
    #     self._update_values()
    #     vv = self._values.copy()
    #     if self.isdiscrete:
    #         vv = ma.filled(vv, self._undef_i)
    #     else:
    #         vv = ma.filled(vv, self._undef)

    #     return vv

    @property
    def ntotal(self):
        """Returns total number of cells ncol*nrow*nlay"""
        return self._ncol * self._nrow * self._nlay

    @property
    def roxorigin(self):
        """Returns True if the property comes from ROXAPI"""
        return self._roxorigin

    @property
    def values3d(self):
        """For backward compatibility (use values instead)"""
        return self._values

    @values3d.setter
    def values3d(self, values):
        # kept for backwards compatibility
        self.values = values

    @property
    def values1d(self):
        """Returns a 1D view of values (masked numpy)"""
        return (self._values.reshape(-1))

    @property
    def undef(self):
        """Get the undef value for floats or ints numpy arrays."""
        if self._isdiscrete:
            return self._undef_i
        else:
            return self._undef

    @property
    def undef_limit(self):
        """Returns the undef limit number, which is slightly less than the
        undef value.

        Hence for numerical precision, one can force undef values
        to a given number, e.g.::

           x[x<x.undef_limit]=999

        Undef limit values cannot be changed.

        """
        if self._isdiscrete:
            return self._undef_ilimit
        else:
            return self._undef_limit

    # =========================================================================
    # Various methods
    # =========================================================================

    def get_npvalues3d(self, fill_value=None):
        """Get a pure numpy copy (not masked) copy of the values, 3D shape

        Args:
            fill_value: Value of masked entries. Default is None which
                means the XTGeo UNDEF value (a high number), different
                for a continuous or discrete property"""
        # this is a function, not a property by design
        fvalue = _cxtgeo.UNDEF
        if self._isdiscrete:
            fvalue = _cxtgeo.UNDEF_INT

        npv3d = ma.filled(self.values, fill_value=fvalue)
        return npv3d

    def get_active_npvalues1d(self):
        """ Return the grid property as a 1D numpy array (copy), active
        cells only"""
        vact = self.values1d.copy()
        vact = vact[~vact.mask]
        return np.array(vact)

    def copy(self, newname=None):
        """Copy a xtgeo.grid3d.GridProperty() object to another instance.

        ::

            >>> mycopy = xx.copy(newname='XPROP')
        """

        if newname is None:
            newname = self.name + '_copy'

        xprop = GridProperty(ncol=self._ncol, nrow=self._nrow, nlay=self._nlay,
                             values=self._values, name=newname,
                             grid=self._grid)

        return xprop

    def mask_undef(self):
        """Make UNDEF values masked."""
        if self._isdiscrete:
            self._values = ma.masked_greater(self._values, self._undef_ilimit)
        else:
            self._values = ma.masked_greater(self._values, self._undef_limit)

    # =========================================================================
    # Import and export
    # =========================================================================

    def from_file(self, pfile, fformat='guess', name='unknown',
                  grid=None, date=None, apiversion=2):
        """
        Import grid property from file, and makes an instance of this class

        Note that the the property may be linked to its geometrical grid,
        through the grid= option. Sometimes this is required.

        Args:
            file (str): name of file to be imported
            fformat (str): file format to be used roff/init/unrst
                (guess is default).
            name (str): name of property to import
            date (int or str): For restart files, date on YYYYMMDD format. Also
                the YYYY-MM-DD form is allowed (string), and for Eclipse,
                mnemonics like 'first', 'last' is also allowed.
            grid (Grid object): Grid Object to link too (optional).
            apiversion (int): Internal XTGeo API setting for Ecl input (1 or 2)

        Examples::

           x = GridProperty()
           x.from_file('somefile.roff', fformat='roff')
           #
           mygrid = Grid('ECL.EGRID')
           pressure_1 = GridProperty()
           pressure_1.from_file('ECL.UNRST', name='PRESSURE', date='first',
                                grid=mygrid)

        Returns:
           True if success, otherwise False
        """

        # it may be that pfile already is an open file; hence a filehandle
        # instead. Check for this, and skip tests of so
        pfile_is_not_fhandle = True
        fhandle, pclose = _get_fhandle(pfile)
        if not pclose:
            pfile_is_not_fhandle = False

        if pfile_is_not_fhandle:
            if os.path.isfile(pfile):
                logger.debug('File {} exists OK'.format(pfile))
            else:
                logger.critical('No such file: {}'.format(pfile))
                raise IOError

            # work on file extension
            froot, fext = os.path.splitext(pfile)
            if fformat == 'guess':
                if len(fext) == 0:
                    logger.critical('File extension missing. STOP')
                    sys.exit(9)
                else:
                    fformat = fext.lower().replace('.', '')

            logger.debug("File name to be used is {}".format(pfile))
            logger.debug("File format is {}".format(fformat))

        ier = 0
        if (fformat == 'roff'):
            logger.info('Importing ROFF...')
            ier = _gridprop_import.import_roff(self, pfile, name, grid=grid,
                                               apiversion=apiversion)

        elif (fformat.lower() == 'init'):
            ier = self._import_ecl_output(pfile, name=name, etype=1,
                                          grid=grid, apiversion=apiversion)
        elif (fformat.lower() == 'unrst'):
            if date is None:
                raise ValueError('Restart file, but no date is given')
            elif isinstance(date, str):
                if '-' in date:
                    date = int(date.replace('-', ''))
                elif date == 'first':
                    date = 0
                elif date == 'last':
                    date = 9
                else:
                    date = int(date)

            ier = self._import_ecl_output(pfile, name=name, etype=5,
                                          grid=grid,
                                          date=date, apiversion=apiversion)
        else:
            logger.warning('Invalid file format')
            sys.exit(1)

        if ier == 22:
            raise DateNotFoundError('Date {} not found when importing {}'
                                    .format(date, name))
        elif ier == 23:
            raise KeywordNotFoundError('Keyword {} not found for date {} '
                                       'when importing'.format(name, date))
        elif ier == 24:
            raise KeywordFoundNoDateError('Keyword {} found but not for date '
                                          '{} when importing'
                                          .format(name, date))
        elif ier == 25:
            raise KeywordNotFoundError('Keyword {} not found when importing'
                                       .format(name))
        elif ier != 0:
            raise RuntimeError('Somethin went wrong, code {}'.format(ier))

        return self

    def from_roxar(self, projectname, gname, pname, realisation=0):

        """Import grid model property from RMS project, and makes an instance.

        Arguments:
            projectname (str): Name of RMS project; use pure 'project'
                if inside RMS
            gfile (str): Name of grid model
            pfile (str): Name of grid property
            projectname (str): Name of RMS project; None if within a project
            realisation (int): Realisation number (default 0 first)

        """
        _gridprop_roxapi.import_prop_roxapi(
            self, projectname, gname, pname, realisation)

    def to_roxar(self, projectname, gname, pname, saveproject=False,
                 realisation=0):

        """Store a grid model property into a RMS project.

        Arguments:
            projectname (str): Name of RMS project ('project' if inside a
                RMS project)
            gfile (str): Name of grid model
            pfile (str): Name of grid property
            projectname (str): Name of RMS project (None if inside a project)
            saveproject (bool): If True, a saveproject job will be ran.
            realisation (int): Realisation number (default 0 first)

        """
        _gridprop_roxapi.export_prop_roxapi(
            self, projectname, gname, pname, saveproject=saveproject,
            realisation=0)

    def to_file(self, pfile, fformat='guess', name=None):
        """
        Export grid property to file.

        Args:
            pfile (str): file name
            fformat (str): file format to be used. The default 'guess' is
                roff which is the only supported currently, which is either
                'roff' or 'roff_binary' for binary, and 'roffasc'
                or 'roff_ascii' for ASCII (text).
            name (str): If provided, will give property name; else the existing
                name of the instance will used.
        """
        logger.debug('Export property to file...')

        # guess based on file extension (todo)
        if fformat == 'guess':
            fformat = 'roff'

        if 'roff' in fformat:
            if name is None:
                name = self.name

            binary = True
            if 'asc' in fformat:
                binary = False

            # for later usage
            append = False
            last = True

            _gridprop_export.export_roff(self, pfile, name, append=append,
                                         last=last, binary=binary)

    def get_xy_value_lists(self, grid=None, mask=True):
        """Get lists of xy coords and values for Webportal format.

        The coordinates are on the form (two cells)::

            [[[(x1,y1), (x2,y2), (x3,y3), (x4,y4)],
            [(x5,y5), (x6,y6), (x7,y7), (x8,y8)]]]

        Args:
            grid (object): The XTGeo Grid object for the property
            mask (bool): If true (default), inactive cells will be omitted,
                otherwise cell geometries will be listed and property will
                have value -999 in undefined cells.

        Example::

            grid = Grid()
            grid.from_file('../xtgeo-testdata/3dgrids/bri/b_grid.roff')
            prop = GridProperty()
            prop.from_file('../xtgeo-testdata/3dgrids/bri/b_poro.roff',
                           grid=grid, name='PORO')

            clist, valuelist = prop.get_xy_value_lists(grid=grid, mask=False)


        """

        clist, vlist = _gridprop_op1.get_xy_value_lists(self, grid=grid,
                                                        mask=mask)
        return clist, vlist

    # =========================================================================
    # PRIVATE METHODS
    # should not be applied outside the class
    # =========================================================================

    # -------------------------------------------------------------------------
    # Import methods for various formats
    # -------------------------------------------------------------------------

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Import Eclipse's INIT and UNRST formats
    # type: 1 binary INIT
    #       5 binary unified restart
    # These Eclipse files need the NX NY NZ and the ACTNUM array as they
    # only store the active cells for most vectors. Hence, the grid sizes
    # and the actnum array shall be provided, and that is done via the grid
    # geometry object directly. Note that this import only takes ONE property
    # at the time... See GridProperies class.
    #
    # Returns: 0: if OK
    #          1: Parameter and or Date is missing
    #         -9: Some serious error
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _import_ecl_output(self, pfile, name=None, etype=1, date=None,
                           grid=None, apiversion=2):

        if apiversion == 1:
            # use the old obsolete version
            logger.info('API version 1')
            ier = _gridprop_import_obsolete.import_eclbinary_v1(
                self, pfile, name=name, etype=etype, date=date, grid=grid)

        elif apiversion == 2:
            # use the new version
            logger.info('API version 2')
            ier = _gridprop_import.import_eclbinary_v2(self, pfile, name=name,
                                                       etype=etype, date=date,
                                                       grid=grid)

        return ier

    def discrete_to_continuous(self):
        """Convert from discrete to continuous values"""

        if self.isdiscrete:
            logger.info('Converting to continuous ...')
            val = self.values.copy()
            val = val.astype('float64')
            self.values = val
            self._isdiscrete = False
            self._dtype = val.dtype
            self._codes = {}
            self._ncodes = 1
        else:
            logger.info('No need to convert, already continuous')


# =============================================================================
# functions outside the class, for rapid access. Will be exposed as
# xxx = xtgeo.gridproperty_from_file. Cf __init__ at top level


def gridproperty_from_file(pfile, fformat='guess', name='unknown',
                           grid=None, date=None, apiversion=2):

    obj = GridProperty()
    obj.from_file(pfile, fformat=fformat, name=name, grid=grid, date=date,
                  apiversion=apiversion)

    return obj


def gridproperty_from_roxar(project, gname, pname, realisation=0):
    print('FROM ROXAR!!')
    obj = GridProperty()
    obj.from_roxar(project, gname, pname, realisation=realisation)

    return obj
