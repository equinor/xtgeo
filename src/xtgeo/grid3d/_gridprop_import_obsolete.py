"""GridProperty (not GridProperies) import functions"""

# Note: Obsolete; keep for a whilein case trouble!
# JRIV

from __future__ import print_function, absolute_import

import numpy as np

import xtgeo.cxtgeo.cxtgeo as _cxtgeo
from xtgeo.common import XTGeoDialog

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
