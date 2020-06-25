# coding: utf-8
"""Private module, Grid Import private functions for ROFF format"""

from __future__ import print_function, absolute_import

from collections import OrderedDict


import numpy as np

import xtgeo.cxtgeo._cxtgeo as _cxtgeo
import xtgeo

from ._grid_roff_lowlevel import _rkwquery, _rkwxlist, _rkwxvec, _rkwxvec_coordsv
from ._grid_roff_lowlevel import _rkwxvec_zcornsv, _rkwxvec_prop
from . import _grid3d_utils as utils

xtg = xtgeo.common.XTGeoDialog()

logger = xtg.functionlogger(__name__)


def import_roff(self, gfile, xtgformat=1):

    gfile.get_cfhandle()

    if xtgformat == 1:
        _import_roff_xtgformat1(self, gfile)
    else:
        _import_roff_xtgformat2(self, gfile)

    gfile.cfclose()


def _import_roff_xtgformat1(self, gfile):
    """Import ROFF grids using xtgformat=1 storage"""

    # pylint: disable=too-many-statements

    # This routine do first a scan for all keywords. Then it grabs
    # the relevant data by only reading relevant portions of the input file

    kwords = utils.scan_keywords(gfile, fformat="roff")

    for kwd in kwords:
        logger.info(kwd)

    # byteswap:
    byteswap = _rkwquery(gfile, kwords, "filedata!byteswaptest", -1)

    self._ncol = _rkwquery(gfile, kwords, "dimensions!nX", byteswap)
    self._nrow = _rkwquery(gfile, kwords, "dimensions!nY", byteswap)
    self._nlay = _rkwquery(gfile, kwords, "dimensions!nZ", byteswap)
    logger.info("Dimensions in ROFF file %s %s %s", self._ncol, self._nrow, self._nlay)

    xshift = _rkwquery(gfile, kwords, "translate!xoffset", byteswap)
    yshift = _rkwquery(gfile, kwords, "translate!yoffset", byteswap)
    zshift = _rkwquery(gfile, kwords, "translate!zoffset", byteswap)
    logger.info("Shifts in ROFF file %s %s %s", xshift, yshift, zshift)

    xscale = _rkwquery(gfile, kwords, "scale!xscale", byteswap)
    yscale = _rkwquery(gfile, kwords, "scale!yscale", byteswap)
    zscale = _rkwquery(gfile, kwords, "scale!zscale", byteswap)
    logger.info("Scaling in ROFF file %s %s %s", xscale, yscale, zscale)

    subs = _rkwxlist(gfile, kwords, "subgrids!nLayers", byteswap, strict=False)
    if subs is not None and subs.size > 1:
        subs = subs.tolist()  # from numpy array to list
        nsubs = len(subs)
        self._subgrids = OrderedDict()
        prev = 1
        for irange in range(nsubs):
            val = subs[irange]
            self._subgrids["subgrid_" + str(irange)] = range(prev, val + prev)
            prev = val + prev
    else:
        self._subgrids = None

    ntot = self._ncol * self._nrow * self._nlay
    ncoord = (self._ncol + 1) * (self._nrow + 1) * 2 * 3
    nzcorn = self._ncol * self._nrow * (self._nlay + 1) * 4

    ncoord, nzcorn, ntot = self.vectordimensions

    self._coordsv = np.zeros(ncoord, dtype=np.float64)
    self._zcornsv = np.zeros(nzcorn, dtype=np.float64)
    self._actnumsv = np.zeros(ntot, dtype=np.int32)

    # read the pointers to the arrays
    p_cornerlines_v = _rkwxvec(gfile, kwords, "cornerLines!data", byteswap)
    p_splitenz_v = _rkwxvec(gfile, kwords, "zvalues!splitEnz", byteswap)
    p_zvalues_v = _rkwxvec(gfile, kwords, "zvalues!data", byteswap)
    p_act_v = _rkwxvec(gfile, kwords, "active!data", byteswap, strict=False)

    _cxtgeo.grd3d_roff2xtgeo_coord(
        self._ncol,
        self._nrow,
        self._nlay,
        xshift,
        yshift,
        zshift,
        xscale,
        yscale,
        zscale,
        p_cornerlines_v,
        self._coordsv,
    )

    _cxtgeo.grd3d_roff2xtgeo_zcorn(
        self._ncol,
        self._nrow,
        self._nlay,
        xshift,
        yshift,
        zshift,
        xscale,
        yscale,
        zscale,
        p_splitenz_v,
        p_zvalues_v,
        self._zcornsv,
    )

    # ACTIVE may be missing, meaning all cells are missing!
    option = 0
    if p_act_v is None:
        p_act_v = _cxtgeo.new_intarray(1)
        option = 1

    _cxtgeo.grd3d_roff2xtgeo_actnum(
        self._ncol, self._nrow, self._nlay, p_act_v, self._actnumsv, option
    )

    _cxtgeo.delete_floatarray(p_cornerlines_v)
    _cxtgeo.delete_floatarray(p_zvalues_v)
    _cxtgeo.delete_intarray(p_splitenz_v)
    _cxtgeo.delete_intarray(p_act_v)

    logger.debug("Calling C routines, DONE")


def _import_roff_xtgformat2(self, gfile):
    """Import ROFF grids using xtgformat=2 storage"""

    self._xtgformat = 2

    kwords = utils.scan_keywords(gfile, fformat="roff")

    for kwd in kwords:
        logger.info(kwd)

    # byteswap:
    byteswap = _rkwquery(gfile, kwords, "filedata!byteswaptest", -1)

    self._ncol = _rkwquery(gfile, kwords, "dimensions!nX", byteswap)
    self._nrow = _rkwquery(gfile, kwords, "dimensions!nY", byteswap)
    self._nlay = _rkwquery(gfile, kwords, "dimensions!nZ", byteswap)
    logger.info("Dimensions in ROFF file %s %s %s", self._ncol, self._nrow, self._nlay)

    xshift = _rkwquery(gfile, kwords, "translate!xoffset", byteswap)
    yshift = _rkwquery(gfile, kwords, "translate!yoffset", byteswap)
    zshift = _rkwquery(gfile, kwords, "translate!zoffset", byteswap)
    logger.info("Shifts in ROFF file %s %s %s", xshift, yshift, zshift)

    xscale = _rkwquery(gfile, kwords, "scale!xscale", byteswap)
    yscale = _rkwquery(gfile, kwords, "scale!yscale", byteswap)
    zscale = _rkwquery(gfile, kwords, "scale!zscale", byteswap)
    logger.info("Scaling in ROFF file %s %s %s", xscale, yscale, zscale)

    subs = _rkwxlist(gfile, kwords, "subgrids!nLayers", byteswap, strict=False)
    if subs is not None and subs.size > 1:
        subs = subs.tolist()  # from numpy array to list
        nsubs = len(subs)
        self._subgrids = OrderedDict()
        prev = 1
        for irange in range(nsubs):
            val = subs[irange]
            self._subgrids["subgrid_" + str(irange)] = range(prev, val + prev)
            prev = val + prev
    else:
        self._subgrids = None

    logger.info("Initilize arrays...")
    self._coordsv = np.zeros((self._ncol + 1, self._nrow + 1, 6), dtype=np.float64)
    self._zcornsv = np.zeros(
        (self._ncol + 1, self._nrow + 1, self._nlay + 1, 4), dtype=np.float32
    )
    logger.info("Initilize arrays... done")

    _rkwxvec_coordsv(
        self, gfile, kwords, byteswap, xshift, yshift, zshift, xscale, yscale, zscale,
    )

    logger.info("ZCORN related...")
    p_splitenz_v = _rkwxvec(gfile, kwords, "zvalues!splitEnz", byteswap)

    _rkwxvec_zcornsv(
        self,
        gfile,
        kwords,
        byteswap,
        xshift,
        yshift,
        zshift,
        xscale,
        yscale,
        zscale,
        p_splitenz_v,
    )
    logger.info("ZCORN related... done")

    logger.info("ACTNUM...")
    self._actnumsv = _rkwxvec_prop(
        self, gfile, kwords, "active!data", byteswap, strict=True,
    )
    logger.info("ACTNUM... done")
    logger.info("XTGFORMAT is %s", self._xtgformat)
