# coding: utf-8
"""Private module, Grid Import private functions"""

from __future__ import print_function, absolute_import

from collections import OrderedDict

import xtgeo.cxtgeo.cxtgeo as _cxtgeo
import xtgeo

xtg = xtgeo.common.XTGeoDialog()

logger = xtg.functionlogger(__name__)

_cxtgeo.xtg_verbose_file('NONE')
xtg_verbose_level = xtg.get_syslevel()

#
# NOTE:
# self is the xtgeo.grid3d.Grid instance
#


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Import roff binary
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def import_roff(self, gfile):

    logger.info('Working with file {}'.format(gfile))

    logger.info('Scanning...')
    ptr_ncol = _cxtgeo.new_intpointer()
    ptr_nrow = _cxtgeo.new_intpointer()
    ptr_nlay = _cxtgeo.new_intpointer()
    ptr_nsubs = _cxtgeo.new_intpointer()

    _cxtgeo.grd3d_scan_roff_bingrid(ptr_ncol, ptr_nrow, ptr_nlay,
                                    ptr_nsubs, gfile, xtg_verbose_level)

    self._ncol = _cxtgeo.intpointer_value(ptr_ncol)
    self._nrow = _cxtgeo.intpointer_value(ptr_nrow)
    self._nlay = _cxtgeo.intpointer_value(ptr_nlay)
    nsubs = _cxtgeo.intpointer_value(ptr_nsubs)

    ntot = self._ncol * self._nrow * self._nlay
    ncoord = (self._ncol + 1) * (self._nrow + 1) * 2 * 3
    nzcorn = self._ncol * self._nrow * (self._nlay + 1) * 4

    logger.info('NCOORD {}'.format(ncoord))
    logger.info('NZCORN {}'.format(nzcorn))
    logger.info('Reading...')

    ptr_num_act = _cxtgeo.new_intpointer()
    self._p_coord_v = _cxtgeo.new_doublearray(ncoord)
    self._p_zcorn_v = _cxtgeo.new_doublearray(nzcorn)
    self._p_actnum_v = _cxtgeo.new_intarray(ntot)
    subgrd_v = _cxtgeo.new_intarray(nsubs)

    _cxtgeo.grd3d_import_roff_grid(ptr_num_act, ptr_nsubs, self._p_coord_v,
                                   self._p_zcorn_v, self._p_actnum_v,
                                   subgrd_v, nsubs, gfile,
                                   xtg_verbose_level)

    logger.info('Number of active cells: {}'.format(self.nactive))
    logger.info('Number of subgrids: {}'.format(nsubs))

    if nsubs > 1:
        self._subgrids = OrderedDict()
        prev = 1
        for irange in range(nsubs):
            val = _cxtgeo.intarray_getitem(subgrd_v, irange)

            logger.info('VAL is %s', val)
            logger.info('RANGE is %s', range(prev, val + prev))
            self._subgrids['subgrid_' + str(irange)] = range(prev, val + prev)
            prev = val + prev
    else:
        self._subgrids = None

    logger.info('Subgrids array %s', self._subgrids)
