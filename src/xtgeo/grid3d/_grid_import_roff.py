# coding: utf-8
"""Private module, Grid Import private functions"""

from __future__ import print_function, absolute_import

from collections import OrderedDict

import xtgeo.cxtgeo.cxtgeo as _cxtgeo
import xtgeo

from ._gridprop_import_roff import _rkwquery, _rkwxlist, _rkwxvec
from . import _grid3d_utils as utils

xtg = xtgeo.common.XTGeoDialog()

logger = xtg.functionlogger(__name__)


XTGDEBUG = 0

#
# NOTE:
# self is the xtgeo.grid3d.Grid instance
#


def import_roff(self, gfile, _roffapiv=2):

    if _roffapiv == 1:
        import_roff_v1(self, gfile)

    else:
        local_fhandle = False
        fhandle = gfile
        if isinstance(gfile, str):
            local_fhandle = True
            gfile = xtgeo._XTGeoCFile(gfile)
            fhandle = gfile.fhandle

        import_roff_v2(self, fhandle)

        if local_fhandle and not gfile.close(cond=local_fhandle):
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

    _cxtgeo.grd3d_scan_roff_bingrid(ptr_ncol, ptr_nrow, ptr_nlay, ptr_nsubs, gfile)

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

    # pylint: disable=too-many-statements

    # This routine do first a scan for all keywords. Then it grabs
    # the relevant data by only reading relevant portions of the input file

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

    subs = _rkwxlist(fhandle, kwords, "subgrids!nLayers", byteswap, strict=False)
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

    # get the pointers to the arrays
    p_cornerlines_v = _rkwxvec(fhandle, kwords, "cornerLines!data", byteswap)
    p_zvalues_v = _rkwxvec(fhandle, kwords, "zvalues!data", byteswap)
    p_splitenz_v = _rkwxvec(fhandle, kwords, "zvalues!splitEnz", byteswap)
    p_act_v = _rkwxvec(fhandle, kwords, "active!data", byteswap, strict=False)

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
    )

    # ACTIVE may be missing, meaning all cells are missing!
    option = 0
    if p_act_v is None:
        p_act_v = _cxtgeo.new_intarray(1)
        option = 1

    _cxtgeo.grd3d_roff2xtgeo_actnum(
        self._ncol, self._nrow, self._nlay, p_act_v, self._p_actnum_v, option
    )

    _cxtgeo.delete_floatarray(p_cornerlines_v)
    _cxtgeo.delete_floatarray(p_zvalues_v)
    _cxtgeo.delete_intarray(p_splitenz_v)
    _cxtgeo.delete_intarray(p_act_v)

    logger.debug("Calling C routines, DONE")

    # xsys.close_fhandle(fhandle)
