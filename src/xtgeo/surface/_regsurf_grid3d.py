# -*- coding: utf-8 -*-
"""Regular surface vs Grid3D"""
from __future__ import division, absolute_import
from __future__ import print_function

import numpy as np

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


def from_grid3d(self, grid, template=None, where="top", mode="depth", rfactor=1):
    """Private function for deriving a surface from a 3D grid.

    Note that rotated maps are currently not supported!

    .. versionadded:: 2.1.0
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

    _update_regsurf(self, template, grid, rfactor=float(rfactor))

    # call C function to make a map
    svalues = self.get_values1d() * 0.0 + xtgeo.UNDEF
    ivalues = svalues.copy()
    jvalues = svalues.copy()

    _cxtgeo.surf_sample_grd3d_lay(
        grid.ncol,
        grid.nrow,
        grid.nlay,
        grid._coordsv,
        grid._zcornsv,
        grid._actnumsv,
        klayer,
        self.ncol,
        self.nrow,
        self.xori,
        self.xinc,
        self.yori,
        self.yinc,
        self.rotation,
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
        self.set_values1d(ivalues)
        return None

    if mode == "j":
        self.set_values1d(jvalues)
        return None

    self.set_values1d(svalues)
    isurf = self.copy()
    jsurf = self.copy()
    isurf.set_values1d(ivalues)
    jsurf.set_values1d(jvalues)
    return isurf, jsurf  # needed in special cases


def _update_regsurf(self, template, grid, rfactor=1.0):

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

        self._xori = xori
        self._yori = yori
        self._xinc = xinc
        self._yinc = yinc
        self._ncol = ncol
        self._nrow = nrow
        self._values = np.ma.zeros((ncol, nrow), dtype=np.float64)
    else:
        self._xori = template.xori
        self._yori = template.yori
        self._xinc = template.xinc
        self._yinc = template.yinc
        self._ncol = template.ncol
        self._nrow = template.nrow
        self._values = template.values.copy()
