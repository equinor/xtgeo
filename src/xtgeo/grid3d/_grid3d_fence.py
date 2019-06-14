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

    if "onegrid" not in self._tmp:
        logger.info("Make a tmp onegrid instance...")
        self._tmp["onegrid"] = self.copy()
        self._tmp["onegrid"].reduce_to_one_layer()
        logger.info("Make a tmp onegrid instance... DONE")
    else:
        logger.info("Re-use existing tmp onegrid instance...")

    if "ijmaps" not in self._tmp:
        logger.info("Make a set of tmp surfaces for I J locations...")
        # self._tmp["ijmaps"] =
        logger.info("Make a set of tmp surfaces for I J locations... DONE")
    else:
        logger.info("Re-use existing tmp surfaces for I J")

    logger.info("Getting randomline... DONE")
