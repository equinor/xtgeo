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


def _import_eclbinary_v2(self, pfile, name=None, etype=1, date=None,
                         grid=None):
    """Import, private to this routine"""

    logger.info('pfile is {}, name is {}, etype is {}, date is {}, '
                'grid is {}'.format(pfile, name, etype, date, grid))

    fhandle, pclose = _get_fhandle(pfile)

    gprops = xtgeo.grid3d.GridProperties()
    nentry = 0
    if etype == 5:
        datefound = False
        # scan for date and find SEQNUM entry number
        dtlist = gprops.scan_dates(fhandle)
        for ientry, dtentry in enumerate(dtlist):
            if str(dtentry[1]) == str(date):
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

    self._ncol = ncol
    self._nrow = nrow
    self._nlay = nlay

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
    logger.debug('Indices for actnum: {}'.format(grid.actnum_indices))
    logger.debug('Len for actnum indices: {}'
                 .format(grid.actnum_indices.shape[0]))
    logger.debug('Len for values: {}'
                 .format(values.shape[0]))

    if grid.actnum_indices.shape[0] == values.shape[0]:
        allvalues[grid.actnum_indices] = values

    if values.shape[0] == ncol * nrow * nlay:  # often the case for PORV array
        allvalues = values.copy()

    allvalues = ma.masked_where(actnum < 1, allvalues)

    _close_fhandle(fhandle, pclose)

    self._grid = grid
    self._values = allvalues
    self._cvalues = None

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
