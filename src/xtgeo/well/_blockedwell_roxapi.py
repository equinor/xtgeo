# -*- coding: utf-8 -*-
"""Well input and output, private module for ROXAPI"""

from __future__ import print_function, absolute_import
from collections import OrderedDict

import numpy as np
import numpy.ma as npma
import pandas as pd

from xtgeo.common import XTGeoDialog
from xtgeo.roxutils import RoxUtils
from xtgeo.common.exceptions import WellNotFoundError

# try:
#     import roxar
# except ImportError:
#     pass

xtg = XTGeoDialog()
logger = xtg.functionlogger(__name__)


# Import / export via ROX api


def import_bwell_roxapi(
    self, project, gname, bwname, wname, lognames=None, ijk=True, realisation=0
):
    """Private function for loading project and ROXAPI blockwell import"""

    logger.info("Opening RMS project ...")
    rox = RoxUtils(project, readonly=True)

    _roxapi_import_bwell(self, rox, gname, bwname, wname, lognames, ijk, realisation)

    rox.safe_close()


def export_bwell_roxapi(self, project, gname, bwname, wname, realisation=0):
    """Private function for loading project and ROXAPI blockwell import"""

    logger.info("Opening RMS project ...")
    rox = RoxUtils(project, readonly=True)

    _roxapi_export_bwell(self, rox, gname, bwname, wname, realisation)

    rox.safe_close()


def _roxapi_import_bwell(
    self, rox, gname, bwname, wname, lognames, ijk, realisation
):  # pylint: disable=too-many-statements
    """Private function for ROXAPI well import (get well from Roxar)"""

    if gname in rox.project.grid_models:
        gmodel = rox.project.grid_models[gname]
        logger.info("RMS grid model <%s> OK", gname)
    else:
        raise ValueError("No such grid name present: {}".format(gname))

    if bwname in gmodel.blocked_wells_set:
        bwset = gmodel.blocked_wells_set[bwname]
        logger.info("Blocked well set <%s> OK", bwname)
    else:
        raise ValueError("No such blocked well set: {}".format(bwname))

    if wname in bwset.get_well_names():
        self._wname = wname
    else:
        raise WellNotFoundError("No such well in blocked well set: {}".format(wname))

    # pylint: disable=unnecessary-comprehension
    bwprops = [item for item in bwset.properties]
    bwnames = [item.name for item in bwset.properties]

    bw_cellindices = bwset.get_cell_numbers()
    dind = bwset.get_data_indices([wname])

    cind = bw_cellindices[dind]
    xyz = np.transpose(gmodel.get_grid().get_cell_centers(cind))

    logs = OrderedDict()
    logs["X_UTME"] = xyz[0].astype(np.float64)
    logs["Y_UTMN"] = xyz[1].astype(np.float64)
    logs["Z_TVDSS"] = xyz[2].astype(np.float64)
    if ijk:
        ijk = np.transpose(gmodel.get_grid().grid_indexer.get_indices(cind))
        logs["I_INDEX"] = ijk[0].astype(np.float64)
        logs["J_INDEX"] = ijk[1].astype(np.float64)
        logs["K_INDEX"] = ijk[2].astype(np.float64)

    usenames = []
    if lognames and lognames == "all":
        usenames = bwnames
    elif lognames:
        usenames = lognames

    for bwprop in bwprops:
        lname = bwprop.name
        if lname not in usenames:
            continue
        propvalues = bwprop.get_values(realisation=realisation)
        tmplog = propvalues[dind].astype(np.float64)
        tmplog = npma.filled(tmplog, fill_value=np.nan)
        tmplog[tmplog == -999] = np.nan
        if "discrete" in str(bwprop.type):
            self._wlogtype[lname] = "DISC"
            self._wlogrecord[lname] = bwprop.code_names
        else:
            self._wlogtype[lname] = "CONT"
            self._wlogrecord[lname] = None

        logs[lname] = tmplog

    self._df = pd.DataFrame.from_dict(logs)
    self._gname = gname
    self._filesrc = None

    # finally get some other metadata like RKB and topside X Y; as they
    # seem to miss for the BW in RMS, try and get them from the
    # well itself...

    if wname in rox.project.wells:
        self._rkb = rox.project.wells[wname].rkb
        self._xpos, self._ypos = rox.project.wells[wname].wellhead
    else:
        self._rkb = None
        self._xpos, self._ypos = self._df["X_UTME"][0], self._df["Y_UTMN"][0]


def _roxapi_export_bwell(
    self, rox, gname, bwname, wname, realisation
):  # pylint: disable=too-many-statements
    """Private function for ROXAPI well export (set well with updated logs to Roxar)"""

    raise NotImplementedError("Later!")

    # if gname in rox.project.grid_models:
    #     gmodel = rox.project.grid_models[gname]
    #     logger.info("RMS grid model <%s> OK", gname)
    # else:
    #     raise ValueError("No such grid name present: {}".format(gname))

    # if bwname in gmodel.blocked_wells_set:
    #     bwset = gmodel.blocked_wells_set[bwname]
    #     logger.info("Blocked well set <%s> OK", bwname)
    # else:
    #     raise ValueError("No such blocked well set: {}".format(bwname))

    # if wname in bwset.get_well_names():
    #     self._wname = wname
    # else:
    #     raise WellNotFoundError("No such well in blocked well set: {}".format(wname))

    # bwprops = [item for item in bwset.properties]
    # bwnames = [item.name for item in bwset.properties]

    # # get the current indices for the well
    # dind = bwset.get_data_indices([self.wname])

    # for lname in self.lognames:
    #     if lname not in bwnames:
    #         if self._wlogtype[lname] == "CONT":
    #             bwlog = bwset.properties.create(lname,
    #                                             roxar.GridPropertyType.continuous,
    #                                             np.float32)
    #             bwprop = bwset.generate_values(discrete=False)
    #         else:
    #             bwlog = bwset.properties.create(lname,
    #                                             roxar.GridPropertyType.discrete,
    #                                             np.uint16)
    #             bwprop = bwset.generate_values(discrete=True)

    #     bwlog =  bwprops[lname]
    #     bwprop = bwlog.get_values(realisation=realisation)
    #     bwsbwprop[dind]

    #     # COFFEE!
    #     bwprop.set_values(self.dataframe[lname].values)
