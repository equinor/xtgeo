# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import

from xtgeo.common import XTGeoDialog
import xtgeo.cxtgeo.cxtgeo as _cxtgeo

xtg = XTGeoDialog()

_cxtgeo.xtg_verbose_file("NONE")
XTGDEBUG = xtg.get_syslevel()

logger = xtg.functionlogger(__name__)


def export_roff(self, gfile, option):
    """Export grid to ROFF format (binary)"""

    logger.debug("Export to ROFF...")

    nsubs = 0
    if self.subgrids is None:
        logger.debug("Create a pointer for subgrd_v ...")
        subgrd_v = _cxtgeo.new_intpointer()
    else:
        nsubs = len(self.subgrids)
        subgrd_v = _cxtgeo.new_intarray(nsubs)
        for inum, (sname, sarray) in enumerate(self.subgrids.items()):
            logger.info("INUM SUBGRID: %s %s", inum, sname)
            _cxtgeo.intarray_setitem(subgrd_v, inum, len(sarray))

    # get the geometrics list to find the xshift, etc
    gx = self.get_geometrics()

    _cxtgeo.grd3d_export_roff_grid(
        option,
        self._ncol,
        self._nrow,
        self._nlay,
        nsubs,
        0,
        gx[3],
        gx[5],
        gx[7],
        self._p_coord_v,
        self._p_zcorn_v,
        self._p_actnum_v,
        subgrd_v,
        gfile,
        XTGDEBUG,
    )

    # skip parameters for now (cf Perl code)

    # end tag
    _cxtgeo.grd3d_export_roff_end(option, gfile, XTGDEBUG)


def export_grdecl(self, gfile, mode):
    """Export grid to Eclipse GRDECL format (ascii, mode=1) or binary
    (mode=0).
    """

    logger.debug("Export to ascii or binary GRDECL...")

    _cxtgeo.grd3d_export_grdecl(
        self._ncol,
        self._nrow,
        self._nlay,
        self._p_coord_v,
        self._p_zcorn_v,
        self._p_actnum_v,
        gfile,
        mode,
        XTGDEBUG,
    )


def export_egrid(self, gfile):
    """Export grid to Eclipse EGRID format, binary."""

    logger.debug("Export to binary EGRID...")

    _cxtgeo.grd3d_export_egrid(
        self._ncol,
        self._nrow,
        self._nlay,
        self._p_coord_v,
        self._p_zcorn_v,
        self._p_actnum_v,
        gfile,
        0,
        XTGDEBUG,
    )
