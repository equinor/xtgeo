"""GridProperty (not GridProperies) import functions"""

from __future__ import print_function, absolute_import

import numpy as np
import numpy.ma as ma

import cxtgeo.cxtgeo as _cxtgeo
import xtgeo
from xtgeo.common import XTGeoDialog
from ._gridprops_io import _get_fhandle, _close_fhandle

xtg = XTGeoDialog()

logger = xtg.functionlogger(__name__)

_cxtgeo.xtg_verbose_file('NONE')
xtg_verbose_level = xtg.get_syslevel()


def import_eclbinary_v1(prop, pfile, name=None, etype=1, date=None,
                        grid=None):

    logger.debug('Scanning NX NY NZ for checking...')
    ptr_ncol = _cxtgeo.new_intpointer()
    ptr_nrow = _cxtgeo.new_intpointer()
    ptr_nlay = _cxtgeo.new_intpointer()

    _cxtgeo.grd3d_scan_ecl_init_hd(1, ptr_ncol, ptr_nrow, ptr_nlay,
                                   pfile, xtg_verbose_level)

    logger.debug('Scanning NX NY NZ for checking... DONE')

    ncol0 = _cxtgeo.intpointer_value(ptr_ncol)
    nrow0 = _cxtgeo.intpointer_value(ptr_nrow)
    nlay0 = _cxtgeo.intpointer_value(ptr_nlay)

    if grid.ncol != ncol0 or grid.nrow != nrow0 or \
            grid.nlay != nlay0:
        logger.error('Errors in dimensions property vs grid')
        return -9

    prop._ncol = ncol0
    prop._nrow = nrow0
    prop._nlay = nlay0

    # split date and populate array
    logger.debug('Date handling...')
    if date is None:
        date = 99998877
    date = str(date)
    logger.debug('DATE is {}'.format(date))
    day = int(date[6:8])
    mon = int(date[4:6])
    yer = int(date[0:4])

    logger.debug('DD MM YYYY input is {} {} {}'.format(day, mon, yer))

    if etype == 1:
        prop._name = name
    else:
        prop._name = name + '_' + date
        logger.info('Active date is {}'.format(date))
        prop._date = date

    ptr_day = _cxtgeo.new_intarray(1)
    ptr_month = _cxtgeo.new_intarray(1)
    ptr_year = _cxtgeo.new_intarray(1)

    _cxtgeo.intarray_setitem(ptr_day, 0, day)
    _cxtgeo.intarray_setitem(ptr_month, 0, mon)
    _cxtgeo.intarray_setitem(ptr_year, 0, yer)

    ptr_dvec_v = _cxtgeo.new_doublearray(ncol0 * nrow0 * nlay0)
    ptr_nktype = _cxtgeo.new_intarray(1)
    ptr_norder = _cxtgeo.new_intarray(1)
    ptr_dsuccess = _cxtgeo.new_intarray(1)

    usename = '{0:8s}|'.format(name)
    logger.debug('<{}>'.format(usename))

    if etype == 1:
        ndates = 0
    if etype == 5:
        ndates = 1
    logger.debug('Import via _cxtgeo... NX NY NX are '
                 '{} {} {}'.format(ncol0, nrow0, nlay0))

    _cxtgeo.grd3d_import_ecl_prop(etype,
                                  ncol0 * nrow0 * nlay0,
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

    logger.debug('Import via _cxtgeo... DONE')
    # process the result:
    norder = _cxtgeo.intarray_getitem(ptr_norder, 0)
    if norder == 0:
        logger.debug('Got 1 item OK')
    else:
        logger.warning('Did not get any property name'
                       ': {} Missing date?'.format(name))
        logger.warning('NORDER is {}'.format(norder))
        return 1

    nktype = _cxtgeo.intarray_getitem(ptr_nktype, 0)

    if nktype == 1:
        prop._cvalues = _cxtgeo.new_intarray(ncol0 * nrow0 * nlay0)
        prop._isdiscrete = True

        prop._dtype = np.int32

        _cxtgeo.grd3d_strip_anint(ncol0 * nrow0 * nlay0, 0,
                                  ptr_dvec_v, prop._cvalues,
                                  xtg_verbose_level)
    else:
        prop._cvalues = _cxtgeo.new_doublearray(ncol0 * nrow0 * nlay0)
        prop._isdiscrete = False

        _cxtgeo.grd3d_strip_adouble(ncol0 * nrow0 * nlay0, 0,
                                    ptr_dvec_v, prop._cvalues,
                                    xtg_verbose_level)

    prop._grid = grid
    # prop._update_values()
    return 0


def import_eclbinary_v2(prop, pfile, name=None, etype=1, date=None,
                        grid=None):

    fhandle, pclose = _get_fhandle(pfile)

    gprops = xtgeo.grid3d.GridProperties()
    nentry = 0
    if etype == 5:
        datefound = False
        # scan for date and find SEQNUM entry number
        dtlist = gprops.scan_dates(fhandle)
        for ientry, dtentry in enumerate(dtlist):
            if dtentry[1] == date:
                datefound = True
                nentry = ientry
                break

        if not datefound:
            return 9

    # scan file for property
    kwlist = gprops.scan_keywords(fhandle, fformat='xecl', maxkeys=10000,
                                  dataframe=False)

    # first INTEHEAD is needed to verify grid dimensions:
    for kwitem in kwlist:
        if kwitem[0] == 'INTEHEAD':
            kwname, kwtype, kwlen, kwbyte = kwitem
            break

    # read INTEHEAD record:
    intehead = eclbin_record(fhandle, kwname, kwlen, kwtype, kwbyte)
    ncol, nrow, nlay = intehead[8:11].tolist()

    prop._ncol = ncol
    prop._nrow = nrow
    prop._nlay = nlay

    logger.info('Grid dimensions in INIT or RESTART file: {} {} {}'
                .format(ncol, nrow, nlay))

    if grid.ncol != ncol or grid.nrow != nrow or grid.nlay != nlay:
        logger.error('Errors in dimensions property vs grid')
        return -9

    kwfound = False
    ientry = 0
    for kwitem in kwlist:
        if name == kwitem[0]:
            kwname, kwtype, kwlen, kwbyte = kwitem
            if ientry == nentry:
                kwfound = True
                break
            else:
                ientry += 1

    if not kwfound:
        return 9

    # read record:
    values = eclbin_record(fhandle, kwname, kwlen, kwtype, kwbyte)

    if kwtype == 'INTE':
        prop._isdiscrete = True
        use_undef = prop._undef_i
        use_undeflimit = prop._undef_ilimit

        # make the code list
        uniq = np.unique(values).tolist()
        codes = dict(zip(uniq, uniq))
        prop.codes = codes

    else:
        prop._isdiscrete = False
        values.astype(np.float64)  # cast REAL (float32) to float64
        use_undef = prop._undef
        use_undeflimit = prop._undef_limit
        prop.codes = {}

    # arrays from Eclipse INIT or UNRST are usually for inactive values only.
    # Use the ACTNUM index array for vectorized numpy remapping
    allvalues = np.zeros((ncol * nrow * nlay), dtype=values.dtype) + use_undef
    logger.debug('Indices for actnum: {}'.format(grid.actnum_indices))
    logger.debug('Len for actnum indices: {}'
                 .format(grid.actnum_indices.shape[0]))
    logger.debug('Len for values: {}'
                 .format(values.shape[0]))

    if grid.actnum_indices.shape[0] == values.shape[0]:
        allvalues[grid.actnum_indices] = values

    allvalues = ma.masked_greater(allvalues, use_undeflimit)

    _close_fhandle(fhandle, pclose)

    prop._grid = grid
    prop._values = allvalues
    prop._cvalues = None

    if etype == 1:
        prop._name = name
    else:
        prop._name = name + '_' + str(date)
        prop._date = date

    return 0


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
