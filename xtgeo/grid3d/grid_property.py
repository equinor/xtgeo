# -*- coding: utf-8 -*-
"""Class for a 3D grid property.

The grid property instances may or may not belong to a grid geometry
object. Normally the instance is created when importing a grid
property from file, but it can also be created directly, as e.g.:

poro = GridProperty(nx=233, ny=122, nz=32)

The grid property values by themselves are 1D numpy as floatxx
precision not as RMS do, (double will represent float32 from e.g.
ROFF) or int (repr int, short, byte...). However, the order will be
Fortran like in Eclipse manner, and masked (while RMS only list the
active cells).

There will however be methods that can output the properties in RMS style,
some sunny day.
"""

import sys
import numpy as np
import numpy.ma as ma
import os.path
import logging

import cxtgeo.cxtgeo as _cxtgeo
from xtgeo.common import XTGeoDialog
from .grid3d import Grid3D
from . import _grid_property_op1

# =============================================================================
# Class constructor
#
# Some key properties:
# _nx        =  number of rows (X, cycling fastest)
# _ny        =  number of columns (Y)
# _nz        =  number of layers (Z)
# _values    =  Numpy 1D array of doubles or int (masked)
# _cvalues   =  SWIG pointer to C array
# NOTE: either _values OR _cvalues will exist; hence the other is "None"
# etc
# =============================================================================


class GridProperty(Grid3D):
    """Class for a single 3D grid property.

    An instance may or may not belong to a grid (geometry) object. E.g. for
    for ROFF, nx, ny, nz are given in the file. Note that in such
    cases the undef cells may (or may not) flagegs as a very high number.
    """

    def __init__(self, nx=5, ny=12, nz=2, values=None, name='Dummy',
                 discrete=False):
        """
        The __init__ (constructor) method, normally ran without args.

        Args:
            nx (int): Number of columns.
            ny (int): Number of rows.
            nz (int): Number of layers.
            values (numpy): A 1D numpy of size nx*ny*nz.
            name (str): Name of property.
            discrete (bool): True if discrete property
                (default is false).

        Returns:
            A GridProperty object.

        Example::

            myprop = GridProperty()
            myprop.from_file('emerald.roff', name='PORO')

        """

        clsname = '{}.{}'.format(type(self).__module__, type(self).__name__)
        self.logger = logging.getLogger(clsname)
        self.logger.addHandler(logging.NullHandler())

        self._nx = nx
        self._ny = ny
        self._nz = nz

        self._isdiscrete = discrete

        if values is None:
            values = np.zeros(5 * 12 * 2)

        self._values = values       # numpy version of properties (as 1D array)
        self._cvalues = None        # carray swig pointer version of map values

        self._dtype = values.dtype

        self._name = name           # property name
        self._date = None           # property may have an assosiated date
        self._codes = {}            # code dictionary (for discrete)
        self._ncodes = 1            # Number of codes in dictionary

        self._undef = _cxtgeo.UNDEF
        self._undef_limit = _cxtgeo.UNDEF_LIMIT

        self._undef_i = _cxtgeo.UNDEF_INT
        self._undef_ilimit = _cxtgeo.UNDEF_INT_LIMIT

        if self._isdiscrete:
            self._values = self._values.astype(np.int32)
            self._dtype = self._values.dtype

        # make some undef cells (for test)
        self._values[0:4] = self._undef

        # make it masked
        self._values = ma.masked_greater(self._values, self._undef_limit)

        self._xtg = XTGeoDialog()

    def __del__(self):
        self._delete_cvalues()

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
    def isdiscrete(self):
        """Return True if property is discrete"""
        return self._isdiscrete

    # -------------------------------------------------------------------------
    @property
    def dtype(self):
        """Return the numpy dtype"""
        return self._dtype

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
        """The property codes as a dict"""
        return self._codes

    @codes.setter
    def codes(self, cdict):
        self._codes = cdict

    @property
    def ncodes(self):
        """Number of codes"""
        return self._ncodes

    # -------------------------------------------------------------------------
    @property
    def cvalues(self):
        """Return the grid property C (SWIG) pointer."""
        self._update_cvalues()
        return self._cvalues

    # -------------------------------------------------------------------------
    @property
    def values(self):
        """ Return or set the grid property as a masked 1D numpy array"""
        self._update_values()
        return self._values

    @values.setter
    def values(self, values):

        self._update_values()

        if (isinstance(values, np.ndarray) and
            not isinstance(values, ma.MaskedArray)):
            values = ma.array(values)

        self._values = values
        self._cvalues = None

        self.logger.debug('Values shape: {}'.format(self._values.shape))
        self.logger.debug('Flags: {}'.format(self._values.flags.c_contiguous))

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
        """Returns total number of cells nx*ny*nz"""
        return self._nx * self._ny * self._nz

    @property
    def values3d(self):
        """
        Return or set the grid property as a masked 3D numpy array of
        shape [nx][ny][nz].
        """
        self._update_values()
        values3d = ma.copy(self._values)
        values3d = ma.reshape(values3d, (self._nx, self._ny, self._nz),
                              order='F')

        return values3d

    @values3d.setter
    def values3d(self, values):
        # flatten the input (a dimension check is needed...)
        self._update_values()
        self._values = ma.reshape(values, -1, order='F')

    @property
    def undef(self):
        """
        Get the undef value for floats or ints numpy arrays
        """
        if self._isdiscrete:
            return self._undef_i
        else:
            return self._undef

    @property
    def undef_limit(self):
        """
        Returns the undef limit number, which is slightly less than the
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

    def copy(self, newname=None):
        """
        Copy a xtgeo.grid3d.GridProperty() object to another instance::

            >>> mycopy = xx.copy(newname='XPROP')
        """

        if newname is None:
            newname = self.name + '_copy'

        x = GridProperty(nx=self._nx, ny=self._ny, nz=self._nz,
                         values=self._values, name=newname)

        return x

    def mask_undef(self):
        """
        Make UNDEF values masked
        """
        if self._isdiscrete:
            self._values = ma.masked_greater(self._values, self._undef_ilimit)
        else:
            self._values = ma.masked_greater(self._values, self._undef_limit)

    # =========================================================================
    # Import and export
    # =========================================================================

    def from_file(self, pfile, fformat='roff', name='unknown',
                  grid=None, date=None):
        """
        Import grid property from file, and makes an instance of this class

        Args:
            file (str): name of file to be imported
            fformat (str): file format to be used roff/init/unrst
                (roff is default).
            name (str): name of property to import
            date (int): For restart files, date on YYYYMMDD format.
            grid (Grid object): Grid Object to link too (optional).

        Example::

           x = GridProperty()
           x.from_file('somefile.roff', fformat='roff')

        Returns:
           True if success, otherwise False
        """

        if os.path.isfile(pfile):
            self.logger.debug('File {} exists OK'.format(pfile))
        else:
            print('No such file: {}'.format(pfile))
            self.logger.critical('No such file: {}'.format(pfile))
            sys.exit(1)

        # work on file extension
        froot, fext = os.path.splitext(pfile)
        if fformat == 'guess':
            if len(fext) == 0:
                self.logger.critical('File extension missing. STOP')
                sys.exit(9)
            else:
                fformat = fext.lower().replace('.', '')

        self.logger.debug("File name to be used is {}".format(pfile))
        self.logger.debug("File format is {}".format(fformat))

        ier = 0
        if (fformat == 'roff'):
            ier = self._import_roff(pfile, name)
        elif (fformat.lower() == 'init'):
            ier = self._import_ecl_output(pfile, name=name, etype=1,
                                          grid=grid)
        elif (fformat.lower() == 'unrst'):
            ier = self._import_ecl_output(pfile, name=name, etype=5,
                                          grid=grid,
                                          date=date)
        else:
            self.logger.warning('Invalid file format')
            sys.exit(1)

        # would be better with excpetion handling?
        print('IER IS', ier)
        if ier == 0:
            return True
        else:
            return False

        return self

    def to_file(self, pfile, fformat='roff', name=None):
        """
        Export grid property to file.

        Args:
            pfile (str): file name
            fformat (str): file format to be used (roff is the only
                supported currently, which is roff binary)
            name (str): If provided, will give property name; else the existing
                name of the instance will used.
        """
        self.logger.debug('Export property to file...')
        if (fformat == 'roff'):
            if name is None:
                name = self.name
            self._export_roff(pfile, name)

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

        clist, vlist = _grid_property_op1.get_xy_value_lists(self, grid=grid,
                                                             mask=mask)
        return clist, vlist

    # =========================================================================
    # PRIVATE METHODS
    # should not be applied outside the class
    # =========================================================================

    # -------------------------------------------------------------------------
    # Import methods for various formats
    # -------------------------------------------------------------------------

    def _import_roff(self, pfile, name):

        # need to call the C function...
        _cxtgeo.xtg_verbose_file('NONE')

        xtg_verbose_level = self._xtg.get_syslevel()

        self.logger.debug('Looking for {} in file {}'.format(name, pfile))

        ptr_nx = _cxtgeo.new_intpointer()
        ptr_ny = _cxtgeo.new_intpointer()
        ptr_nz = _cxtgeo.new_intpointer()
        ptr_ncodes = _cxtgeo.new_intpointer()
        ptr_type = _cxtgeo.new_intpointer()

        ptr_idum = _cxtgeo.new_intpointer()
        ptr_ddum = _cxtgeo.new_doublepointer()

        # read with mode 0, to scan for nx, ny, nz and ndcodes, and if
        # property is found
        self.logger.debug('Entering C library... with level {}'.
                          format(xtg_verbose_level))

        # note that ...'
        ier, codenames = _cxtgeo.grd3d_imp_prop_roffbin(pfile,
                                                        0,
                                                        ptr_type,
                                                        ptr_nx,
                                                        ptr_ny,
                                                        ptr_nz,
                                                        ptr_ncodes,
                                                        name,
                                                        ptr_idum,
                                                        ptr_ddum,
                                                        ptr_idum,
                                                        0,
                                                        xtg_verbose_level)

        if (ier == -1):
            msg = 'Cannot find property name {}'.format(name)
            self.logger.critical(msg)
            return ier

        self._nx = _cxtgeo.intpointer_value(ptr_nx)
        self._ny = _cxtgeo.intpointer_value(ptr_ny)
        self._nz = _cxtgeo.intpointer_value(ptr_nz)
        self._ncodes = _cxtgeo.intpointer_value(ptr_ncodes)

        ptype = _cxtgeo.intpointer_value(ptr_type)

        ntot = self._nx * self._ny * self._nz

        if (self._ncodes <= 1):
            self._ncodes = 1
            self._codes = {0: 'undef'}

        self.logger.debug('NTOT is ' + str(ntot))
        self.logger.debug('Grid size is '
                          '{} {} {}'.format(self._nx, self._ny, self._nz))

        self.logger.debug('Number of codes: {}'.format(self._ncodes))

        # allocate

        if ptype == 1:  # float, assign to double
            ptr_pval_v = _cxtgeo.new_doublearray(ntot)
            ptr_ival_v = _cxtgeo.new_intarray(1)
            self._isdiscrete = False
            self._dtype = 'float64'

        elif ptype > 1:
            ptr_pval_v = _cxtgeo.new_doublearray(1)
            ptr_ival_v = _cxtgeo.new_intarray(ntot)
            self._isdiscrete = True
            self._dtype = 'int32'

        self.logger.debug('Is Property discrete? {}'.format(self._isdiscrete))

        # number of codes and names
        ptr_ccodes_v = _cxtgeo.new_intarray(self._ncodes)

        # NB! note the SWIG trick to return modified char values; use cstring.i
        # inn the config and %cstring_bounded_output(char *p_codenames_v, NN);
        # Then the argument for *p_codevalues_v in C is OMITTED here!

        ier, cnames = _cxtgeo.grd3d_imp_prop_roffbin(pfile,
                                                     1,
                                                     ptr_type,
                                                     ptr_nx,
                                                     ptr_ny,
                                                     ptr_nz,
                                                     ptr_ncodes,
                                                     name,
                                                     ptr_ival_v,
                                                     ptr_pval_v,
                                                     ptr_ccodes_v,
                                                     0,
                                                     xtg_verbose_level)

        if self._isdiscrete:
            self._cvalues = ptr_ival_v
        else:
            self._cvalues = ptr_pval_v

        self._values = None

        self.logger.debug('CNAMES: {}'.format(cnames))

        # now make dictionary of codes
        if self._isdiscrete:
            cnames = cnames.replace(';', '')
            cname_list = cnames.split('|')
            cname_list.pop()  # some rubbish as last entry
            ccodes = []
            for i in range(0, self._ncodes):
                ccodes.append(_cxtgeo.intarray_getitem(ptr_ccodes_v, i))

            self.logger.debug(cname_list)
            self._codes = dict(zip(ccodes, cname_list))
            self.logger.debug('CODES (value: name): {}'.format(self._codes))

        return 0
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Import Eclipse's INIT and UNRST formats
    # type: 1 binary INIT
    #       5 binary unified restart
    # These Eclipse files need the NX NY NZ and the ACTNUM array as they
    # only store the active cells for most vectors. Hence, the grid sizes
    # and the actnum array shall be provided, and that is done via the grid
    # geometry object directly. Note that this import only takes ONE property
    # at the time. But the C function has the potential to take many props
    # in one go; perhaps need some kind of metaclass for this? (as this class
    # only takes one property at the time). See GridProperies class.
    #
    # Returns: 0: if OK
    #          1: Parameter and or Date is missing
    #         -9: Some serious error
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _import_ecl_output(self, pfile, name=None, etype=1, date=None,
                           grid=None):

        _cxtgeo.xtg_verbose_file('NONE')

        xtg_verbose_level = self._xtg.get_syslevel()

        self.logger.debug('Scanning NX NY NZ for checking...')
        ptr_nx = _cxtgeo.new_intpointer()
        ptr_ny = _cxtgeo.new_intpointer()
        ptr_nz = _cxtgeo.new_intpointer()

        _cxtgeo.grd3d_scan_ecl_init_hd(1, ptr_nx, ptr_ny, ptr_nz,
                                       pfile, xtg_verbose_level)

        self.logger.debug('Scanning NX NY NZ for checking... DONE')

        nx0 = _cxtgeo.intpointer_value(ptr_nx)
        ny0 = _cxtgeo.intpointer_value(ptr_ny)
        nz0 = _cxtgeo.intpointer_value(ptr_nz)

        if grid.nx != nx0 or grid.ny != ny0 or \
                grid.nz != nz0:
            self.logger.error('Errors in dimensions property vs grid')
            return -9

        self._nx = nx0
        self._ny = ny0
        self._nz = nz0

        # split date and populate array
        self.logger.debug('Date handling...')
        if date is None:
            date = 99998877
        date = str(date)
        self.logger.debug('DATE is {}'.format(date))
        day = int(date[6:8])
        mon = int(date[4:6])
        yer = int(date[0:4])

        self.logger.debug('DD MM YYYY input is {} {} {}'.format(day, mon, yer))

        if etype == 1:
            self._name = name
        else:
            self._name = name + '_' + date

        ptr_day = _cxtgeo.new_intarray(1)
        ptr_month = _cxtgeo.new_intarray(1)
        ptr_year = _cxtgeo.new_intarray(1)

        _cxtgeo.intarray_setitem(ptr_day, 0, day)
        _cxtgeo.intarray_setitem(ptr_month, 0, mon)
        _cxtgeo.intarray_setitem(ptr_year, 0, yer)

        ptr_dvec_v = _cxtgeo.new_doublearray(nx0 * ny0 * nz0)
        ptr_nktype = _cxtgeo.new_intarray(1)
        ptr_norder = _cxtgeo.new_intarray(1)
        ptr_dsuccess = _cxtgeo.new_intarray(1)

        usename = '{0:8s}|'.format(name)
        self.logger.debug('<{}>'.format(usename))

        if etype == 1:
            ndates = 0
        if etype == 5:
            ndates = 1
        self.logger.debug('Import via _cxtgeo... NX NY NX are '
                          '{} {} {}'.format(nx0, ny0, nz0))

        _cxtgeo.grd3d_import_ecl_prop(etype,
                                      nx0 * ny0 * nz0,
                                      grid._p_actnum_v,
                                      1,
                                      usename,
                                      ndates,
                                      ptr_day,
                                      ptr_month,
                                      ptr_year,
                                      pfile,
                                      ptr_dvec_v,
                                      ptr_nktype,
                                      ptr_norder,
                                      ptr_dsuccess,
                                      xtg_verbose_level)

        self.logger.debug('Import via _cxtgeo... DONE')
        # process the result:
        norder = _cxtgeo.intarray_getitem(ptr_norder, 0)
        if norder == 0:
            self.logger.debug('Got 1 item OK')
        else:
            self.logger.warning('Did not get any property name'
                                ': {} Missing date?'.format(name))
            self.logger.warning('NORDER is {}'.format(norder))
            return 1

        nktype = _cxtgeo.intarray_getitem(ptr_nktype, 0)

        if nktype == 1:
            self._cvalues = _cxtgeo.new_intarray(nx0 * ny0 * nz0)
            self._isdiscrete = True

            self._dtype = np.int32

            _cxtgeo.grd3d_strip_anint(nx0 * ny0 * nz0, 0,
                                      ptr_dvec_v, self._cvalues,
                                      xtg_verbose_level)
        else:
            self._cvalues = _cxtgeo.new_doublearray(nx0 * ny0 * nz0)
            self._isdiscrete = False

            _cxtgeo.grd3d_strip_adouble(nx0 * ny0 * nz0, 0,
                                        ptr_dvec_v, self._cvalues,
                                        xtg_verbose_level)

        # self._update_values()
        return 0

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Export ROFF format
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _export_roff(self, pfile, name):

        self.logger.debug('Exporting {} to file {}'.format(name, pfile))

        self._update_cvalues()
        # need to call the C function...
        _cxtgeo.xtg_verbose_file('NONE')

        xtg_verbose_level = self._xtg.get_syslevel()

        ptr_idum = _cxtgeo.new_intpointer()

        mode = 0  # binary
        if not self._isdiscrete:
            _cxtgeo.grd3d_export_roff_pstart(mode, self._nx, self._ny,
                                             self._nz, pfile,
                                             xtg_verbose_level)

        # now the actual data
        # only float data are supported for now!
        nsub = 0
        isub_to_export = 0
        if not self._isdiscrete:
            _cxtgeo.grd3d_export_roff_prop(mode, self._nx, self._ny, self._nz,
                                           nsub, isub_to_export, ptr_idum,
                                           name, 'double', ptr_idum,
                                           self._cvalues, 0, '',
                                           ptr_idum, pfile, xtg_verbose_level)
        else:
            self.logger.critical('INT export not supported yet')
            sys.exit(1)

        _cxtgeo.grd3d_export_roff_end(mode, pfile, xtg_verbose_level)

    # =========================================================================
    # PRIVATE HELPER (LOW LEVEL) METHODS
    # should not be applied outside the class
    # =========================================================================

    def _update_values(self):
        """copy (update) values from SWIG carray to numpy, 1D array"""
        self.logger.debug('Update numpy from C values')

        n = self.ntotal
        self.logger.debug('N is {}'.format(n))

        if self._cvalues is None and self._values is not None:
            return
        elif self._cvalues is None and self._values is None:
            self.logger.critical('_cvalues and _values is None in '
                                 '_update_values. STOP')
            sys.exit(9)

        if not self._isdiscrete:
            self.logger.debug('Entering conversion to numpy (float64) ...')
            x = _cxtgeo.swig_carr_to_numpy_1d(n, self._cvalues)

            # make it float64 or whatever(?) and mask it
            self._values = x.astype(self._dtype)
            self.mask_undef()

        else:
            self.logger.debug('Entering conversion to numpy (int32) ...')
            self._values = _cxtgeo.swig_carr_to_numpy_i1d(n, self._cvalues)

            # make it int32 (not as RMS?) and mask it
            self._values = self._values.astype(self._dtype)
            self.mask_undef()

        self._cvalues = None

    def _update_cvalues(self):
        """copy (update) values from numpy to SWIG, 1D array"""

        self.logger.debug('Entering conversion from numpy to C array ...')
        if self._values is None and self._cvalues is not None:
            return
        elif self._cvalues is None and self._values is None:
            self.logger.critical('_cvalues and _values is None in '
                                 '_update_values. STOP')
            sys.exit(9)

        x = self._values
        x = ma.filled(x, self._undef)
        self.logger.debug(x)
        self.logger.debug(x.shape)
        self.logger.debug(self.ntotal)

        if x.dtype == 'float64' and self._isdiscrete:
            x = x.astype('int32')
            self.logger.degug('Casting has been done')

        if self._isdiscrete is False:
            self._cvalues = _cxtgeo.new_doublearray(self.ntotal)
            _cxtgeo.swig_numpy_to_carr_1d(x, self._cvalues)
        else:
            self._cvalues = _cxtgeo.new_intarray(self.ntotal)
            _cxtgeo.swig_numpy_to_carr_i1d(x, self._cvalues)

        self._values = None

    def _delete_cvalues(self):
        """Delete cpointer"""
        self.logger.debug('Enter delete cvalues values method...')

        if self._cvalues is not None:
            if self._isdiscrete:
                _cxtgeo.delete_intarray(self._cvalues)
            else:
                _cxtgeo.delete_doublearray(self._cvalues)

        self._cvalues = None
        self.logger.debug('Enter method... DONE')

    def _check_shape_ok(self, values):
        """Check if chape of values is OK"""
        (nx, ny, nz) = values.shape
        if nx != self._nx or ny != self._ny or nz != self._nz:
            self.logger.error('Wrong shape: Dimens of values {} {} {}'
                              'vs {} {} {}'.format(nx, ny, self._nx, self._ny))
            return False
        return True

    def discrete_to_continuous(self):
        """Convert from discrete to continuous values"""

        if self.isdiscrete:
            val = self.values.copy()
            val = val.astype('float64')
            self.values = val
            self._isdiscrete = False
            self._dtype = val.dtype
            self._codes = {}
            self._ncodes = 1
        else:
            self.logger.debug('No need to convert, already continuous')


def main():
    """Initial testing with main()"""
    logger = logging.getLogger(__name__)

    pfile = '../../../../testdata/Zone/gullfaks2_poro.roffbin'

    poro = GridProperty()
    poro.from_file(pfile, fformat='roff', name='PORO')

    xtg = XTGeoDialog()

    xtg.info(poro.values)

    logger.debug('NX method is inherited from Grid class: {}'.format(poro.nx))

    poro.values = poro.values + 3

    xtg.info(poro.values)

    pfile = '../../../../testdata/Zone/gullfaks2_zone.roffbin'

    zones = GridProperty()
    zones.from_file(pfile, fformat='roff', name='Zone')

    xtg.info(zones.values)

    # undef for int
    # x = GridProperty.get_float_undef()
    x = zones.undef_limit
    zones.undef_limit = 999888
    y = zones.undef_limit

    print('X is {} and Y is {}'.format(x, y))


if __name__ == '__main__':
    main()
