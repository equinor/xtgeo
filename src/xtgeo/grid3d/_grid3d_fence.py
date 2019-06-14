# -*- coding: utf-8 -*-

"""Some grid utilities, file scanning etc (methods with no class)"""
from __future__ import division, absolute_import
from __future__ import print_function

import xtgeo
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


    xcoords = fencespec[:, 0]
    ycoords = fencespec[:, 1]
    hcoords = fencespec[:, 3]

    if zmin is None:
        zmin = 1200;
    if zmax is None:
        zmax = 1900;

    _ier, values = _cxtgeo.grid3d_get_randomline(
        xcoords,
        ycoords,
        zmin,
        zmax,
        nzsam,

        self._tmp["ijmaps"][0].ncol,
        self._tmp["ijmaps"][0].nrow,
        self._tmp["ijmaps"][0].xori,
        self._tmp["ijmaps"][0].yori,
        self._tmp["ijmaps"][0].xinc,
        self._tmp["ijmaps"][0].yinc,
        self._tmp["ijmaps"][0].xori,
        self._tmp["ijmaps"][0].rotation,
        self._tmp["ijmaps"][0].yflip,
        self._tmp["ijmaps"][0].values,
        self._tmp["ijmaps"][1].values,
        self._tmp["ijmaps"][2].values,
        self._tmp["ijmaps"][3].values,

    logger.info("Getting randomline... DONE")


def _update_tmpvars(self):
    """The self._tmpvars are needed to speed up calculations.

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
        self._tmp["topi"], self._tmp["topj"] = self._tmp["topd"].from_grid3d(where="top")

        self._tmp["basd"] = xtgeo.RegularSurface()
        self._tmp["basi"], self._tmp["basj"] = self._tmp["topd"].from_grid3d(where="base")

        logger.info("Make a set of tmp surfaces for I J locations + depth... DONE")
    else:
        logger.info("Re-use existing tmp surfaces for I J")
