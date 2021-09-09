# -*- coding: utf-8 -*-
"""Regular surface vs Grid3D"""


import numpy as np
import numpy.ma as ma

import xtgeo
import xtgeo.cxtgeo._cxtgeo as _cxtgeo
from xtgeo.common import XTGeoDialog
from xtgeo.grid3d import _gridprop_lowlevel

xtg = XTGeoDialog()

logger = xtg.functionlogger(__name__)

#


# self = RegularSurface instance!
# pylint: disable=protected-access


def slice_grid3d(self, grid, prop, zsurf=None, sbuffer=1):
    """Private function for the Grid3D slicing."""

    grid._xtgformat1()
    if zsurf is not None:
        other = zsurf
    else:
        logger.info('The current surface is copied as "other"')
        other = self.copy()
    if not self.compare_topology(other, strict=False):
        raise RuntimeError("Topology of maps differ. Stop!")

    zslice = other.copy()

    nsurf = self.ncol * self.nrow

    p_prop = _gridprop_lowlevel.update_carray(prop, discrete=False)

    istat, updatedval = _cxtgeo.surf_slice_grd3d(
        self.ncol,
        self.nrow,
        self.xori,
        self.xinc,
        self.yori,
        self.yinc,
        self.rotation,
        self.yflip,
        zslice.get_values1d(),
        nsurf,
        grid.ncol,
        grid.nrow,
        grid.nlay,
        grid._coordsv,
        grid._zcornsv,
        grid._actnumsv,
        p_prop,
        sbuffer,
    )

    if istat != 0:
        logger.warning("Problem, ISTAT = %s", istat)

    self.set_values1d(updatedval)

    return istat


def from_grid3d(grid, template=None, where="top", mode="depth", rfactor=1):
    """Private function for deriving a surface from a 3D grid.

    Note that rotated maps are currently not supported!

    .. versionadded:: 2.1
    """
    if where == "top":
        klayer = 1
        option = 0
    elif where == "base":
        klayer = grid.nlay
        option = 1
    else:
        klayer, what = where.split("_")
        klayer = int(klayer)
        if grid.nlay < klayer < 0:
            raise ValueError("Klayer out of range in where={}".format(where))
        option = 0
        if what == "base":
            option = 1

    if rfactor < 0.5:
        raise KeyError("Refinefactor rfactor is too small, should be >= 0.5")

    args = _update_regsurf(template, grid, rfactor=float(rfactor))
    args["rotation"] = 0.0
    # call C function to make a map
    val = args["values"]
    val = val.ravel(order="K")
    val = ma.filled(val, fill_value=xtgeo.UNDEF)

    svalues = val * 0.0 + xtgeo.UNDEF
    ivalues = svalues.copy()
    jvalues = svalues.copy()

    grid._xtgformat1()
    _cxtgeo.surf_sample_grd3d_lay(
        grid.ncol,
        grid.nrow,
        grid.nlay,
        grid._coordsv,
        grid._zcornsv,
        grid._actnumsv,
        klayer,
        args["ncol"],
        args["nrow"],
        args["xori"],
        args["xinc"],
        args["yori"],
        args["yinc"],
        args["rotation"],
        svalues,
        ivalues,
        jvalues,
        option,
    )

    logger.info("Extracted surfaces from 3D grid...")
    svalues = np.ma.masked_greater(svalues, xtgeo.UNDEF_LIMIT)
    ivalues = np.ma.masked_greater(ivalues, xtgeo.UNDEF_LIMIT)
    jvalues = np.ma.masked_greater(jvalues, xtgeo.UNDEF_LIMIT)

    if mode == "i":
        ivalues = ivalues.reshape((args["ncol"], args["nrow"]))
        ivalues = ma.masked_invalid(ivalues)
        args["values"] = ivalues
        return args, None, None

    if mode == "j":
        jvalues = jvalues.reshape((args["ncol"], args["nrow"]))
        jvalues = ma.masked_invalid(jvalues)
        args["values"] = jvalues
        return args, None, None

    svalues = svalues.reshape((args["ncol"], args["nrow"]))
    svalues = ma.masked_invalid(svalues)
    args["values"] = svalues

    return args, ivalues, jvalues


def _update_regsurf(template, grid, rfactor=1.0):
    args = {}
    if template is None:
        # need to estimate map settings from the existing grid. this
        # may a bit time consuming for large grids.
        geom = grid.get_geometrics(
            allcells=True, cellcenter=True, return_dict=True, _ver=2
        )

        xlen = 1.1 * (geom["xmax"] - geom["xmin"])
        ylen = 1.1 * (geom["ymax"] - geom["ymin"])
        xori = geom["xmin"] - 0.05 * xlen
        yori = geom["ymin"] - 0.05 * ylen
        # take same xinc and yinc

        xinc = yinc = (1.0 / rfactor) * 0.5 * (geom["avg_dx"] + geom["avg_dy"])
        ncol = int(xlen / xinc)
        nrow = int(ylen / yinc)

        args["xori"] = xori
        args["yori"] = yori
        args["xinc"] = xinc
        args["yinc"] = yinc
        args["ncol"] = ncol
        args["nrow"] = nrow
        args["values"] = np.ma.zeros((ncol, nrow), dtype=np.float64)
    else:
        args["xori"] = template.xori
        args["yori"] = template.yori
        args["xinc"] = template.xinc
        args["yinc"] = template.yinc
        args["ncol"] = template.ncol
        args["nrow"] = template.nrow
        args["values"] = template.values.copy()
    return args
