# coding: utf-8
"""Private module, Grid Import private functions for ROFF format."""

from collections import OrderedDict

import numpy as np

import xtgeo.cxtgeo._cxtgeo as _cxtgeo
import xtgeo

from ._grid_roff_lowlevel import _rkwquery, _rkwxlist, _rkwxvec
from . import _grid3d_utils as utils

xtg = xtgeo.common.XTGeoDialog()

logger = xtg.functionlogger(__name__)


def import_roff(gfile, xtgformat=1):
    """Import binary ROFF format."""
    gfile.get_cfhandle()
    if xtgformat == 1:
        args = _import_roff_xtgformat1(gfile)
    gfile.cfclose()
    return args


def _import_roff_xtgformat1(gfile):
    """Import ROFF grids using xtgformat=1 storage."""
    # pylint: disable=too-many-statements

    # This routine do first a scan for all keywords. Then it grabs
    # the relevant data by only reading relevant portions of the input file

    logger.info("Importing using xtgformat 1")
    kwords = utils.scan_keywords(gfile, fformat="roff")

    for kwd in kwords:
        logger.info(kwd)

    # byteswap:
    byteswap = _rkwquery(gfile, kwords, "filedata!byteswaptest", -1)
    ncol = _rkwquery(gfile, kwords, "dimensions!nX", byteswap)
    nrow = _rkwquery(gfile, kwords, "dimensions!nY", byteswap)
    nlay = _rkwquery(gfile, kwords, "dimensions!nZ", byteswap)

    logger.info("Dimensions in ROFF file %s %s %s", ncol, nrow, nlay)

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
        subgrids = OrderedDict()
        for irange, subgrid in enumerate(subs.tolist()):
            subgrids["subgrid_" + str(irange)] = subgrid
    else:
        subgrids = None
    ntot = ncol * nrow * nlay
    ncoord = (ncol + 1) * (nrow + 1) * 2 * 3
    nzcorn = ncol * nrow * (nlay + 1) * 4

    coordsv = np.zeros(ncoord, dtype=np.float64)
    zcornsv = np.zeros(nzcorn, dtype=np.float64)
    actnumsv = np.zeros(ntot, dtype=np.int32)

    # read the pointers to the arrays
    p_cornerlines_v = _rkwxvec(gfile, kwords, "cornerLines!data", byteswap)
    p_splitenz_v = _rkwxvec(gfile, kwords, "zvalues!splitEnz", byteswap)
    p_zvalues_v = _rkwxvec(gfile, kwords, "zvalues!data", byteswap)
    p_act_v = _rkwxvec(gfile, kwords, "active!data", byteswap, strict=False)

    _cxtgeo.grd3d_roff2xtgeo_coord(
        ncol,
        nrow,
        nlay,
        xshift,
        yshift,
        zshift,
        xscale,
        yscale,
        zscale,
        p_cornerlines_v,
        coordsv,
    )

    _cxtgeo.grd3d_roff2xtgeo_zcorn(
        ncol,
        nrow,
        nlay,
        xshift,
        yshift,
        zshift,
        xscale,
        yscale,
        zscale,
        p_splitenz_v,
        p_zvalues_v,
        zcornsv,
    )

    # ACTIVE may be missing, meaning all cells are active?
    option = 0
    if p_act_v is None:
        p_act_v = _cxtgeo.new_intarray(1)
        option = 1

    _cxtgeo.grd3d_roff2xtgeo_actnum(ncol, nrow, nlay, p_act_v, actnumsv, option)

    _cxtgeo.delete_floatarray(p_cornerlines_v)
    _cxtgeo.delete_floatarray(p_zvalues_v)
    _cxtgeo.delete_intarray(p_splitenz_v)
    _cxtgeo.delete_intarray(p_act_v)

    logger.debug("Calling C routines, DONE")
    args = {
        "ncol": ncol,
        "nrow": nrow,
        "nlay": nlay,
        "coordsv": coordsv,
        "zcornsv": zcornsv,
        "actnumsv": actnumsv,
        "xshift": xshift,
        "yshift": yshift,
        "zshift": zshift,
        "xscale": xscale,
        "yscale": yscale,
        "zscale": zscale,
        "subgrids": subgrids,
        "xtgformat": 1,
    }
    return args
