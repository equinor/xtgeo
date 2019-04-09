# coding: utf-8
"""Private module, Grid Import private functions"""

from __future__ import print_function, absolute_import

from collections import OrderedDict

import xtgeo.cxtgeo.cxtgeo as _cxtgeo
import xtgeo

xtg = xtgeo.common.XTGeoDialog()

logger = xtg.functionlogger(__name__)

_cxtgeo.xtg_verbose_file("NONE")
XTGDEBUG = xtg.get_syslevel()

#
# NOTE:
# self is the xtgeo.grid3d.Grid instance
#


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Import roff binary
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def import_roff(self, gfile):

    tstart = xtg.timer()
    logger.info("Working with file %s", gfile)

    logger.info("Scanning...")
    ptr_ncol = _cxtgeo.new_intpointer()
    ptr_nrow = _cxtgeo.new_intpointer()
    ptr_nlay = _cxtgeo.new_intpointer()
    ptr_nsubs = _cxtgeo.new_intpointer()

    _cxtgeo.grd3d_scan_roff_bingrid(
        ptr_ncol, ptr_nrow, ptr_nlay, ptr_nsubs, gfile, XTGDEBUG
    )

    self._ncol = _cxtgeo.intpointer_value(ptr_ncol)
    self._nrow = _cxtgeo.intpointer_value(ptr_nrow)
    self._nlay = _cxtgeo.intpointer_value(ptr_nlay)
    nsubs = _cxtgeo.intpointer_value(ptr_nsubs)

    ntot = self._ncol * self._nrow * self._nlay
    ncoord = (self._ncol + 1) * (self._nrow + 1) * 2 * 3
    nzcorn = self._ncol * self._nrow * (self._nlay + 1) * 4

    ptr_num_act = _cxtgeo.new_intpointer()
    self._p_coord_v = _cxtgeo.new_doublearray(ncoord)
    self._p_zcorn_v = _cxtgeo.new_doublearray(nzcorn)
    self._p_actnum_v = _cxtgeo.new_intarray(ntot)
    subgrd_v = _cxtgeo.new_intarray(nsubs)

    logger.info("Reading..., total number of cells is %s", ntot)
    _cxtgeo.grd3d_import_roff_grid(
        ptr_num_act,
        ptr_nsubs,
        self._p_coord_v,
        self._p_zcorn_v,
        self._p_actnum_v,
        subgrd_v,
        nsubs,
        gfile,
        XTGDEBUG,
    )

    logger.info("Reading done. Active cells: %s", self.nactive)
    logger.info("Number of subgrids: %s", nsubs)

    if nsubs > 1:
        self._subgrids = OrderedDict()
        prev = 1
        for irange in range(nsubs):
            val = _cxtgeo.intarray_getitem(subgrd_v, irange)

            logger.debug("VAL is %s", val)
            logger.debug("RANGE is %s", range(prev, val + prev))
            self._subgrids["subgrid_" + str(irange)] = range(prev, val + prev)
            prev = val + prev
    else:
        self._subgrids = None

    logger.debug("Subgrids array %s", self._subgrids)
    logger.info("Total time for ROFF import was %6.2fs", xtg.timer(tstart))
