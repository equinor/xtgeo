"""Grid import functions for Eclipse, new approach (i.e. version 2)"""

from __future__ import print_function, absolute_import

import xtgeo.cxtgeo.cxtgeo as _cxtgeo
import xtgeo
from xtgeo.common import XTGeoDialog

from xtgeo.grid3d._gridprop_import import eclbin_record

from xtgeo.common import _get_fhandle, _close_fhandle

xtg = XTGeoDialog()

logger = xtg.functionlogger(__name__)

_cxtgeo.xtg_verbose_file('NONE')
XTGDEBUG = xtg.get_syslevel()


def import_ecl_egrid(self, gfile):
    """Import, private to this routine.

    """
    fhandle, pclose = _get_fhandle(gfile)

    gprops = xtgeo.grid3d.GridProperties()

    # scan file for property
    logger.info('Make kwlist by scanning')
    kwlist = gprops.scan_keywords(fhandle, fformat='xecl', maxkeys=1000,
                                  dataframe=False, dates=False)
    bpos = {}
    for kwitem in kwlist:
        kwname, kwtype, kwlen, kwbyte = kwitem
        if kwname == 'GRIDHEAD':
            # read GRIDHEAD record:
            gridhead = eclbin_record(fhandle, 'GRIDHEAD', kwlen, kwtype,
                                     kwbyte)
            ncol, nrow, nlay = gridhead[1:4].tolist()
            logger.info('%s %s %s', ncol, nrow, nlay)
        elif kwname in ('COORD', 'ZCORN', 'ACTNUM', 'MAPAXES'):
            bpos[kwname] = kwbyte

    self._ncol = ncol
    self._nrow = nrow
    self._nlay = nlay

    logger.info('Grid dimensions in EGRID file: {} {} {}'
                .format(ncol, nrow, nlay))

    # allocate dimensions:
    ntot = self._ncol * self._nrow * self._nlay
    ncoord = (self._ncol + 1) * (self._nrow + 1) * 2 * 3
    nzcorn = self._ncol * self._nrow * (self._nlay + 1) * 4

    self._p_coord_v = _cxtgeo.new_doublearray(ncoord)
    self._p_zcorn_v = _cxtgeo.new_doublearray(nzcorn)
    self._p_actnum_v = _cxtgeo.new_intarray(ntot)

    nact = _cxtgeo.grd3d_imp_ecl_egrid(
        fhandle, self._ncol, self._nrow, self._nlay, bpos['MAPAXES'],
        bpos['COORD'], bpos['ZCORN'], bpos['ACTNUM'], self._p_coord_v,
        self._p_zcorn_v, self._p_actnum_v, XTGDEBUG)

    self._nactive = nact

    _close_fhandle(fhandle, pclose)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Import eclipse run suite: EGRID + properties from INIT and UNRST
# For the INIT and UNRST, props dates shall be selected
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def import_ecl_run(self, groot, initprops=None,
                   restartprops=None, restartdates=None):

    ecl_grid = groot + '.EGRID'
    ecl_init = groot + '.INIT'
    ecl_rsta = groot + '.UNRST'

    # import the grid
    import_ecl_egrid(self, ecl_grid)

    grdprops = xtgeo.grid3d.GridProperties()

    # import the init properties unless list is empty
    if initprops:
        grdprops.from_file(ecl_init, names=initprops, fformat='init',
                           dates=None, grid=self)

    # import the restart properties for dates unless lists are empty
    if restartprops and restartdates:
        grdprops.from_file(ecl_rsta, names=restartprops,
                           fformat='unrst', dates=restartdates,
                           grid=self)

    self.gridprops = grdprops
