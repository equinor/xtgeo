# -*- coding: utf-8 -*-
"""Well input and output, private module for ROXAPI"""


from collections import OrderedDict

import numpy as np
import numpy.ma as npma
import pandas as pd

from xtgeo.common import XTGeoDialog
from xtgeo.roxutils import RoxUtils
from xtgeo.common.exceptions import WellNotFoundError

try:
    import roxar
except ImportError:
    pass

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


def export_bwell_roxapi(
    self, project, gname, bwname, wname, lognames="all", ijk=False, realisation=0
):
    """Private function for blockwell export (store in RMS) from XTGeo to RoxarAPI"""

    logger.info("Opening RMS project ...")
    rox = RoxUtils(project, readonly=False)

    _roxapi_export_bwell(self, rox, gname, bwname, wname, lognames, ijk, realisation)

    if rox._roxexternal:
        rox.project.save()

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
    xyz = np.transpose(gmodel.get_grid(realisation=realisation).get_cell_centers(cind))

    logs = OrderedDict()
    logs["X_UTME"] = xyz[0].astype(np.float64)
    logs["Y_UTMN"] = xyz[1].astype(np.float64)
    logs["Z_TVDSS"] = xyz[2].astype(np.float64)
    if ijk:
        ijk = np.transpose(
            gmodel.get_grid(realisation=realisation).grid_indexer.get_indices(cind)
        )
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
            self._wlogtypes[lname] = "DISC"
            self._wlogrecords[lname] = bwprop.code_names
        else:
            self._wlogtypes[lname] = "CONT"
            self._wlogrecords[lname] = None

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
    self, rox, gname, bwname, wname, lognames, ijk, realisation
):  # pylint: disable=too-many-statements
    """Private function for ROXAPI well export (set well with updated logs to Roxar)"""

    if gname in rox.project.grid_models:
        gmodel = rox.project.grid_models[gname]
        logger.info("RMS grid model <%s> OK", gname)
    else:
        raise ValueError("No such grid name present: {}".format(gname))

    if bwname in gmodel.blocked_wells_set:
        bwset = gmodel.blocked_wells_set[bwname]
        logger.info("Blocked well set <%s> OK", bwname)
        bwprops = bwset.properties
    else:
        raise ValueError("No such blocked well set: {}".format(bwname))

    if wname in bwset.get_well_names():
        self._wname = wname
    else:
        raise WellNotFoundError("No such well in blocked well set: {}".format(wname))

    bwnames = [item.name for item in bwset.properties]

    # get the current indices for the well
    dind = bwset.get_data_indices([self._wname])

    for lname in self.lognames:
        if not ijk and "_INDEX" in lname:
            continue

        if lognames != "all" and lname not in lognames:
            continue

        if lname not in bwnames:
            if self._wlogtypes[lname] == "CONT":
                print("Create CONT", lname, "for", wname)
                bwlog = bwset.properties.create(
                    lname, roxar.GridPropertyType.continuous, np.float32
                )
                bwprop = bwset.generate_values(discrete=False, fill_value=0.0)
            else:
                print("Create DISK", lname, "for", wname)
                bwlog = bwset.properties.create(
                    lname, roxar.GridPropertyType.discrete, np.int32
                )
                bwprop = bwset.generate_values(discrete=True, fill_value=0)

        else:
            bwlog = bwprops[lname]
            bwprop = bwlog.get_values(realisation=realisation)

        usedtype = bwprop.dtype
        dind = bwset.get_data_indices([self._wname])

        if self.dataframe[lname].values.size != dind.size:
            raise ValueError(
                "Dataframe is of wrong size, changing numbers of rows is not possible"
            )

        maskedvalues = np.ma.masked_invalid(self.dataframe[lname].values).astype(
            usedtype
        )

        bwprop[dind] = maskedvalues
        bwlog.set_values(bwprop)
