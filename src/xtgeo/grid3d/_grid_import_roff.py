# coding: utf-8
"""Private module, Grid Import private functions"""

from __future__ import print_function, absolute_import

from collections import OrderedDict

import xtgeo.cxtgeo.cxtgeo as _cxtgeo
import xtgeo
import xtgeo.common.xtgeo_system as xsys

from ._gridprop_import_roff import _rkwquery, _rkwxvec
from . import _grid3d_utils as utils

xtg = xtgeo.common.XTGeoDialog()

logger = xtg.functionlogger(__name__)

_cxtgeo.xtg_verbose_file("NONE")
XTGDEBUG = xtg.get_syslevel()

#
# NOTE:
# self is the xtgeo.grid3d.Grid instance
#


def import_roff(self, gfile, _roffapiv=1):
    if _roffapiv == 1:
        import_roff_v1(self, gfile)
    else:

        local_fhandle = not xsys.is_fhandle(gfile)
        fhandle = xsys.get_fhandle(gfile)

        import_roff_v2(self, fhandle)

        if not xsys.close_fhandle(fhandle, cond=local_fhandle):
            raise RuntimeError("Error in closing file handle for binary ROFF file")


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Import roff binary (current version, rather slow on windows)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def import_roff_v1(self, gfile):

    tstart = xtg.timer()
    logger.info("Working with file %s", gfile)

    logger.info("Scan file for dimensions...")
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

    logger.info(
        "Reading grid geometry..., total number of cells is %s (%.2f million)",
        ntot,
        float(ntot / 1.0e6),
    )
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

    logger.info("Reading grid geometry... DONE")
    logger.info(
        "Active cells: %s (%.2f million)", self.nactive, float(self.nactive) / 1.0e6
    )
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


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Import roff binary (new version, in prep!!)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def import_roff_v2(self, fhandle):

    """Import ROFF format, version 2 (improved version)"""

    # This routine do first a scan for all keywords. Then it grabs
    # the relevant data by only reading relevant portions of the input file

    fhandle = xsys.get_fhandle(fhandle)

    kwords = utils.scan_keywords(fhandle, fformat="roff")

    for kwd in kwords:
        logger.info(kwd)

    # byteswap:
    byteswap = _rkwquery(fhandle, kwords, "filedata!byteswaptest", -1)

    self._ncol = _rkwquery(fhandle, kwords, "dimensions!nX", byteswap)
    self._nrow = _rkwquery(fhandle, kwords, "dimensions!nY", byteswap)
    self._nlay = _rkwquery(fhandle, kwords, "dimensions!nZ", byteswap)
    logger.info("Dimensions in ROFF file %s %s %s", self._ncol, self._nrow, self._nlay)

    xshift = _rkwquery(fhandle, kwords, "translate!xoffset", byteswap)
    yshift = _rkwquery(fhandle, kwords, "translate!yoffset", byteswap)
    zshift = _rkwquery(fhandle, kwords, "translate!zoffset", byteswap)
    logger.info("Shifts in ROFF file %s %s %s", xshift, yshift, zshift)

    xscale = _rkwquery(fhandle, kwords, "scale!xscale", byteswap)
    yscale = _rkwquery(fhandle, kwords, "scale!yscale", byteswap)
    zscale = _rkwquery(fhandle, kwords, "scale!zscale", byteswap)
    logger.info("Scaling in ROFF file %s %s %s", xscale, yscale, zscale)

    # get the pointers to the arrays
    p_cornerlines_v = _rkwxvec(fhandle, kwords, "cornerLines!data", byteswap)
    p_zvalues_v = _rkwxvec(fhandle, kwords, "zvalues!data", byteswap)
    p_splitenz_v = _rkwxvec(fhandle, kwords, "zvalues!splitEnz", byteswap)
    p_act_v = _rkwxvec(fhandle, kwords, "active!data", byteswap)

    ntot = self._ncol * self._nrow * self._nlay
    ncoord = (self._ncol + 1) * (self._nrow + 1) * 2 * 3
    nzcorn = self._ncol * self._nrow * (self._nlay + 1) * 4

    self._p_coord_v = _cxtgeo.new_doublearray(ncoord)
    self._p_zcorn_v = _cxtgeo.new_doublearray(nzcorn)
    self._p_actnum_v = _cxtgeo.new_intarray(ntot)

    logger.debug("Calling C routines")

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
        self._p_coord_v,
        XTGDEBUG,
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
        self._p_zcorn_v,
        XTGDEBUG,
    )

    _cxtgeo.grd3d_roff2xtgeo_actnum(
        self._ncol, self._nrow, self._nlay, p_act_v, self._p_actnum_v, XTGDEBUG
    )

    logger.debug("Calling C routine, DONE")

    # xsys.close_fhandle(fhandle)
