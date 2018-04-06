"""GridProperty (not GridProperies) import functions"""

from __future__ import print_function, absolute_import

import numpy as np
import numpy.ma as ma

import cxtgeo.cxtgeo as _cxtgeo
import xtgeo
from xtgeo.common import XTGeoDialog
from xtgeo.common.exceptions import DateNotFoundError
from xtgeo.common.exceptions import KeywordFoundNoDateError
from xtgeo.common.exceptions import KeywordNotFoundError
from xtgeo.grid3d import _gridprop_lowlevel

from ._gridprops_io import _get_fhandle, _close_fhandle

xtg = XTGeoDialog()

logger = xtg.functionlogger(__name__)

_cxtgeo.xtg_verbose_file('NONE')
xtg_verbose_level = xtg.get_syslevel()


def _import_eclbinary_v2(self, pfile, name=None, etype=1, date=None,
                         grid=None):
    """Import, private to this routine.

    Raises:
        DateNotFoundError: If restart no neot contain requested date.
        KeywordFoundNoDateError: If keyword is found but not at given date.
        KeywordNotFoundError: If Keyword is not found.
        RuntimeError: Mismatch in grid vs property, etc.

    """

    logger.info('pfile is {}, name is {}, etype is {}, date is {}, '
                'grid is {}'.format(pfile, name, etype, date, grid))

    fhandle, pclose = _get_fhandle(pfile)

    gprops = xtgeo.grid3d.GridProperties()
    nentry = 0

    datefound = True
    if etype == 5:
        datefound = False
        logger.info('Look for date {}'.format(date))
        # scan for date and find SEQNUM entry number
        dtlist = gprops.scan_dates(fhandle)
        for ientry, dtentry in enumerate(dtlist):
            logger.info('ientry {} dtentry {}'.format(ientry, dtentry))
            if str(dtentry[1]) == str(date):
                datefound = True
                nentry = ientry
                break

        if not datefound:
            msg = ('In {}: Date {} not found, nentry={}'
                   .format(pfile, date, nentry))
            xtg.warn(msg)
            raise DateNotFoundError(msg)

    # scan file for property
    logger.info('Make kwlist')
    kwlist = gprops.scan_keywords(fhandle, fformat='xecl', maxkeys=100000,
                                  dataframe=False, dates=True)

    # first INTEHEAD is needed to verify grid dimensions:
    for kwitem in kwlist:
        if kwitem[0] == 'INTEHEAD':
            kwname, kwtype, kwlen, kwbyte, kwdate = kwitem
            break

    # read INTEHEAD record:
    intehead = eclbin_record(fhandle, kwname, kwlen, kwtype, kwbyte)
    ncol, nrow, nlay = intehead[8:11].tolist()

    self._ncol = ncol
    self._nrow = nrow
    self._nlay = nlay

    logger.info('Grid dimensions in INIT or RESTART file: {} {} {}'
                .format(ncol, nrow, nlay))

    if grid.ncol != ncol or grid.nrow != nrow or grid.nlay != nlay:
        msg = ('In {}: Errors in dimensions prop: {} {} {} vs grid: {} {} {} '
               .format(pfile, ncol, nrow, nlay,
                       grid.ncol, grid.ncol, grid.nlay))
        raise RuntimeError(msg)

    # Restarts (etype == 5):
    # there are cases where keywords do not exist for all dates, e.g .'RV'.
    # The trick is to check for dates also...

    kwfound = False
    datefoundhere = False
    usedate = '0'
    restart = False

    if etype == 5:
        usedate = str(date)
        restart = True

    for kwitem in kwlist:
        kwname, kwtype, kwlen, kwbyte, kwdate = kwitem
        logger.debug('Keyword {} -  date: {} usedate: {}'
                     .format(kwname, kwdate, usedate))
        if name == kwname:
            kwfound = True

        if name == kwname and usedate == str(kwdate):
            logger.info('Keyword {} ok at date {}'.format(name, usedate))
            kwname, kwtype, kwlen, kwbyte, kwdate = kwitem
            datefoundhere = True
            break

    if restart:
        if datefound and not kwfound:
            msg = ('For {}: Date <{}> is found, but not keyword <{}>'
                   .format(pfile, date, name))
            xtg.warn(msg)
            raise KeywordNotFoundError(msg)

        if not datefoundhere and kwfound:
            msg = ('For {}: The keyword <{}> exists but not for '
                   'date <{}>'.format(pfile, name, date))
            xtg.warn(msg)
            raise KeywordFoundNoDateError(msg)
    else:
        if not kwfound:
            msg = ('For {}: The keyword <{}> is not found'.format(pfile, name))
            xtg.warn(msg)
            raise KeywordNotFoundError(msg)

    # read record:
    values = eclbin_record(fhandle, kwname, kwlen, kwtype, kwbyte)

    if kwtype == 'INTE':
        self._isdiscrete = True
        use_undef = self._undef_i

        # make the code list
        uniq = np.unique(values).tolist()
        codes = dict(zip(uniq, uniq))
        self.codes = codes

    else:
        self._isdiscrete = False
        values = values.astype(np.float64)  # cast REAL (float32) to float64
        use_undef = self._undef
        self.codes = {}

    # arrays from Eclipse INIT or UNRST are usually for inactive values only.
    # Use the ACTNUM index array for vectorized numpy remapping
    actnum = grid.get_actnum().values
    allvalues = np.zeros((ncol * nrow * nlay), dtype=values.dtype) + use_undef

    if grid.actnum_indices.shape[0] == values.shape[0]:
        allvalues[grid.actnum_indices] = values

    if values.shape[0] == ncol * nrow * nlay:  # often the case for PORV array
        allvalues = values.copy()

    allvalues = allvalues.reshape((ncol, nrow, nlay), order='F')
    allvalues = np.asanyarray(allvalues, order='C')
    allvalues = ma.masked_where(actnum < 1, allvalues)

    _close_fhandle(fhandle, pclose)

    self._grid = grid
    self._values = allvalues

    if etype == 1:
        self._name = name
    else:
        self._name = name + '_' + str(date)
        self._date = date

    return 0


def import_eclbinary_v2(self, pfile, name=None, etype=1, date=None,
                        grid=None):
    ios = 0
    if name == 'SOIL':
        # some recursive magic here
        logger.info('Making SOIL from SWAT and SGAS ...')
        logger.info('PFILE is {}'.format(pfile))

        swat = xtgeo.grid3d.GridProperty()
        swat.from_file(pfile, name='SWAT', grid=grid,
                       date=date, fformat='unrst')

        sgas = xtgeo.grid3d.GridProperty()
        sgas.from_file(pfile, name='SGAS', grid=grid,
                       date=date, fformat='unrst')

        self.name = 'SOIL' + '_' + str(date)
        self.values = swat.values * -1 - sgas.values + 1.0
        self._nrow = swat.nrow
        self._ncol = swat.ncol
        self._nlay = swat.nlay
        self._grid = grid
        self._date = date

        del swat
        del sgas

    else:
        logger.info('Importing {}'.format(name))
        ios = _import_eclbinary_v2(self, pfile, name=name, etype=etype,
                                   date=date, grid=grid)

    return ios


def eclbin_record(fhandle, kwname, kwlen, kwtype, kwbyte):
    # read a binary Eclipse record via cxtgeo

    ilen = flen = dlen = 1

    if kwtype == 'INTE':
        ilen = kwlen
        kwntype = 1
    elif kwtype == 'REAL':
        flen = kwlen
        kwntype = 2
    elif kwtype == 'DOUB':
        dlen = kwlen
        kwntype = 3

    npint = np.zeros((ilen), dtype=np.int32)
    npflt = np.zeros((flen), dtype=np.float32)
    npdbl = np.zeros((dlen), dtype=np.float64)

    _cxtgeo.grd3d_read_eclrecord(fhandle, kwbyte, kwntype,
                                 npint, npflt, npdbl, xtg_verbose_level)

    if kwtype == 'INTE':
        return npint
    elif kwtype == 'REAL':
        return npflt
    elif kwtype == 'DOUB':
        return npdbl


def import_roff(self, pfile, name, grid=None):
    """Import ROFF format"""

    # there is a todo here to get it more robust for various cases,
    # e.g. that a ROFF file may contain both a grid an numerous
    # props

    logger.info('Looking for {} in file {}'.format(name, pfile))

    ptr_ncol = _cxtgeo.new_intpointer()
    ptr_nrow = _cxtgeo.new_intpointer()
    ptr_nlay = _cxtgeo.new_intpointer()
    ptr_ncodes = _cxtgeo.new_intpointer()
    ptr_type = _cxtgeo.new_intpointer()

    ptr_idum = _cxtgeo.new_intpointer()
    ptr_ddum = _cxtgeo.new_doublepointer()

    # read with mode 0, to scan for ncol, nrow, nlay and ndcodes, and if
    # property is found
    logger.debug('Entering C library... with level {}'.
                 format(xtg_verbose_level))

    # note that ...'
    ier, codenames = _cxtgeo.grd3d_imp_prop_roffbin(pfile,
                                                    0,
                                                    ptr_type,
                                                    ptr_ncol,
                                                    ptr_nrow,
                                                    ptr_nlay,
                                                    ptr_ncodes,
                                                    name,
                                                    ptr_idum,
                                                    ptr_ddum,
                                                    ptr_idum,
                                                    0,
                                                    xtg_verbose_level)

    if (ier == -1):
        msg = 'Cannot find property name {}'.format(name)
        logger.critical(msg)
        return ier

    self._ncol = _cxtgeo.intpointer_value(ptr_ncol)
    self._nrow = _cxtgeo.intpointer_value(ptr_nrow)
    self._nlay = _cxtgeo.intpointer_value(ptr_nlay)
    self._ncodes = _cxtgeo.intpointer_value(ptr_ncodes)

    ptype = _cxtgeo.intpointer_value(ptr_type)

    ntot = self._ncol * self._nrow * self._nlay

    if (self._ncodes <= 1):
        self._ncodes = 1
        self._codes = {0: 'undef'}

    logger.debug('NTOT is ' + str(ntot))
    logger.debug('Grid size is {} {} {}'
                 .format(self._ncol, self._nrow, self._nlay))

    logger.debug('Number of codes: {}'.format(self._ncodes))

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

    logger.debug('Is Property discrete? {}'.format(self._isdiscrete))

    # number of codes and names
    ptr_ccodes_v = _cxtgeo.new_intarray(self._ncodes)

    # NB! note the SWIG trick to return modified char values; use cstring.i
    # inn the config and %cstring_bounded_output(char *p_codenames_v, NN);
    # Then the argument for *p_codevalues_v in C is OMITTED here!

    ier, cnames = _cxtgeo.grd3d_imp_prop_roffbin(pfile,
                                                 1,
                                                 ptr_type,
                                                 ptr_ncol,
                                                 ptr_nrow,
                                                 ptr_nlay,
                                                 ptr_ncodes,
                                                 name,
                                                 ptr_ival_v,
                                                 ptr_pval_v,
                                                 ptr_ccodes_v,
                                                 0,
                                                 xtg_verbose_level)

    if self._isdiscrete:
        _gridprop_lowlevel.update_values_from_carray(self, ptr_ival_v,
                                                     np.int32,
                                                     delete=True)
    else:
        _gridprop_lowlevel.update_values_from_carray(self, ptr_pval_v,
                                                     np.float64,
                                                     delete=True)

    logger.debug('CNAMES: {}'.format(cnames))

    # now make dictionary of codes
    if self._isdiscrete:
        cnames = cnames.replace(';', '')
        cname_list = cnames.split('|')
        cname_list.pop()  # some rubbish as last entry
        ccodes = []
        for i in range(0, self._ncodes):
            ccodes.append(_cxtgeo.intarray_getitem(ptr_ccodes_v, i))

        logger.debug(cname_list)
        self._codes = dict(zip(ccodes, cname_list))
        logger.debug('CODES (value: name): {}'.format(self._codes))

    self._grid = grid

    return 0
