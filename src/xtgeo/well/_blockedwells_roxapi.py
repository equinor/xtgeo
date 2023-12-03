# -*- coding: utf-8 -*-
"""Well input and output, private module for ROXAPI"""


from xtgeo.common import XTGeoDialog, null_logger
from xtgeo.roxutils import RoxUtils

from .blocked_well import blockedwell_from_roxar

xtg = XTGeoDialog()
logger = null_logger(__name__)


# Import from ROX api
# --------------------------------------------------------------------------------------
def import_bwells_roxapi(
    self, project, gname, bwname, lognames=None, ijk=True, realisation=0
):  # pragma: no cover
    """Private function for loading project and ROXAPI blockwell import"""

    logger.debug("Opening RMS project ...")
    rox = RoxUtils(project, readonly=True)

    _roxapi_import_bwells(self, rox, gname, bwname, lognames, ijk, realisation)

    rox.safe_close()


def _roxapi_import_bwells(
    self, rox, gname, bwname, lnames, ijk, realisation
):  # pragma: no cover
    """Private function for ROXAPI blocked wells import"""

    if gname in rox.project.grid_models:
        gmodel = rox.project.grid_models[gname]
        logger.debug("RMS grid model <%s> OK", gname)
    else:
        raise ValueError(f"No such grid name present: {gname}")

    if bwname in gmodel.blocked_wells_set:
        bwset = gmodel.blocked_wells_set[bwname]
        logger.debug("Blocked well set <%s> OK", bwname)
    else:
        raise ValueError(f"No such blocked well set: {bwname}")

    wnames = bwset.get_well_names()

    logger.debug("Lognames are %s", lnames)

    bwlist = []
    logger.debug("Loading wells ...")
    for wname in wnames:
        logger.debug("Loading well %s", wname)
        bwtmp = blockedwell_from_roxar(
            rox.project,
            gname,
            bwname,
            wname,
            lognames=lnames,
            ijk=ijk,
            realisation=realisation,
        )
        bwlist.append(bwtmp)

    self._wells = bwlist

    if not self._wells:
        xtg.warn("No wells imported to BlockedWells")
        self._wells = None
