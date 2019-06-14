# -*- coding: utf-8 -*-

"""Some grid utilities, file scanning etc (methods with no class)"""
from __future__ import division, absolute_import
from __future__ import print_function

import numpy as np

import xtgeo
from xtgeo.grid3d import _gridprop_lowlevel as gl
import xtgeo.cxtgeo.cxtgeo as _cxtgeo

xtg = xtgeo.common.XTGeoDialog()
logger = xtg.functionlogger(__name__)

XTGDEBUG = xtg.get_syslevel()
_cxtgeo.xtg_verbose_file("NONE")


def get_randomline(
    self,
    fencespec,
    prop,
    zmin=None,
    zmax=None,
    zincrement=1.0,
    hincrement=None,
    atleast=5,
    nextend=2,
):

    logger.info("Getting randomline...")

    _update_tmpvars(self)

    xcoords = fencespec[:, 0]
    ycoords = fencespec[:, 1]
    hcoords = fencespec[:, 3]

    if zmin is None:
        zmin = self._tmp["topd"].values.min()
    if zmax is None:
        zmax = self._tmp["basd"].values.max()

    nzsam = int((zmax - zmin) / zincrement)
    nsamples = xcoords.shape[0] * nzsam

    _ier, values = _cxtgeo.grid3d_get_randomline(
        xcoords,
        ycoords,
        zmin,
        zmax,
        nzsam,

        self._tmp["topd"].ncol,
        self._tmp["topd"].nrow,
        self._tmp["topd"].xori,
        self._tmp["topd"].yori,
        self._tmp["topd"].xinc,
        self._tmp["topd"].yinc,
        self._tmp["topd"].xori,
        self._tmp["topd"].rotation,
        self._tmp["topd"].yflip,
        self._tmp["topi"].get_values1d(),
        self._tmp["topj"].get_values1d(),
        self._tmp["basi"].get_values1d(),
        self._tmp["basj"].get_values1d(),

        self.ncol,
        self.nrow,
        self._p_coord_v,
        self._p_zcorn_v,
        self._p_actnum_v,
        gl.update_carray(prop),
        nsamples,
        0,
        XTGDEBUG
    )

    values[values > xtgeo.UNDEF_LIMIT] = np.nan
    arr = values.reshape((xcoords.shape[0], nzsam)).T

    logger.info("Getting randomline... DONE")
    return (hcoords[0], hcoords[-1], zmin, zmax, arr)


def _update_tmpvars(self):
    """The self._tmp variables are needed to speed up calculations.

    If they are already created, the no need to recreate
    """

    if "onegrid" not in self._tmp:
        logger.info("Make a tmp onegrid instance...")
        self._tmp["onegrid"] = self.copy()
        self._tmp["onegrid"].reduce_to_one_layer()
        logger.info("Make a tmp onegrid instance... DONE")
    else:
        logger.info("Re-use existing tmp onegrid instance...")

    if "topd" not in self._tmp:
        logger.info("Make a set of tmp surfaces for I J locations + depth...")
        self._tmp["topd"] = xtgeo.RegularSurface()
        self._tmp["topi"], self._tmp["topj"] = self._tmp["topd"].from_grid3d(
            where="top"
        )

        self._tmp["basd"] = xtgeo.RegularSurface()
        self._tmp["basi"], self._tmp["basj"] = self._tmp["topd"].from_grid3d(
            where="base"
        )

        logger.info("Make a set of tmp surfaces for I J locations + depth... DONE")
    else:
        logger.info("Re-use existing tmp surfaces for I J")
