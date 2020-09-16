# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import

import numpy as np
import xtgeo
from xtgeo.common import XTGeoDialog
import xtgeo.cxtgeo._cxtgeo as _cxtgeo

xtg = XTGeoDialog()

logger = xtg.functionlogger(__name__)


def export_roff(self, gfile, option):
    """Export grid to ROFF format (binary)"""
    if self._xtgformat == 1:
        _export_roff_v1(self, gfile, option)
    else:
        _export_roff_v2(self, gfile, option)


def _export_roff_v1(self, gfile, option):
    """Export grid to ROFF format (binary)"""

    self._xtgformat1()
    gfile = xtgeo._XTGeoFile(gfile, mode="wb")
    gfile.check_folder(raiseerror=OSError)

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
        self._coordsv,
        self._zcornsv,
        self._actnumsv,
        subgrd_v,
        gfile.name,
    )

    # end tag
    _cxtgeo.grd3d_export_roff_end(option, gfile.name)


def _export_roff_v2(self, gfile, ascii_fmt):
    """Export grid to ROFF format (binary/ascii) _xtgformat=2"""

    self._xtgformat2()

    gfile = xtgeo._XTGeoFile(gfile, mode="wb")
    gfile.check_folder(raiseerror=OSError)
    cfhandle = gfile.get_cfhandle()

    logger.debug("Export to ROFF... ascii_fmt = %s", ascii_fmt)

    subs = self.get_subgrids()
    if subs:
        sublist = np.array(list(subs.values()), dtype=np.int32)
    else:
        sublist = np.zeros((1), dtype=np.int32)

    # get the geometrics list to find the xshift, etc
    # gx = self.get_geometrics()

    _cxtgeo.grdcp3d_export_roff_bin_start_end(
        0, ascii_fmt, "grid", self.ncol, self.nrow, self.nlay, cfhandle
    )

    _cxtgeo.grdcp3d_export_roff_grid(
        ascii_fmt,
        self._ncol,
        self._nrow,
        self._nlay,
        0.0,
        0.0,
        0.0,
        sublist,
        self._coordsv,
        self._zcornsv,
        self._actnumsv,
        cfhandle,
    )

    # # TODO: export assosiated properties

    # end tag
    _cxtgeo.grdcp3d_export_roff_bin_start_end(
        1, ascii_fmt, "xxxx", self.ncol, self.nrow, self.nlay, cfhandle
    )
    gfile.cfclose()


def export_grdecl(self, gfile, mode):
    """Export grid to Eclipse GRDECL format (ascii, mode=1) or binary
    (mode=0).
    """

    logger.debug("Export to ascii or binary GRDECL...")

    _cxtgeo.grd3d_export_grdecl(
        self._ncol,
        self._nrow,
        self._nlay,
        self._coordsv,
        self._zcornsv,
        self._actnumsv,
        gfile,
        mode,
    )


def export_egrid(self, gfile):
    """Export grid to Eclipse EGRID format, binary."""

    logger.debug("Export to binary EGRID...")

    _cxtgeo.grd3d_export_egrid(
        self._ncol,
        self._nrow,
        self._nlay,
        self._coordsv,
        self._zcornsv,
        self._actnumsv,
        gfile,
        0,
    )
