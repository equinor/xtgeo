"""Blocked Well input and output, private module for ROXAPI"""

import numpy as np
import numpy.ma as npma
import pandas as pd

from xtgeo.common._xyz_enum import _AttrName, _AttrType
from xtgeo.common.constants import INT_MIN
from xtgeo.common.exceptions import WellNotFoundError
from xtgeo.common.log import null_logger
from xtgeo.roxutils import RoxUtils
from xtgeo.roxutils._roxar_loader import roxar

logger = null_logger(__name__)


# Import / export via ROX/RMS api


def import_bwell_roxapi(
    self, project, gname, bwname, wname, lognames=None, ijk=True, realisation=0
):
    """Private function for loading project and ROXAPI blockwell import"""

    logger.debug("Opening RMS project ...")
    rox = RoxUtils(project, readonly=True)

    _roxapi_import_bwell(self, rox, gname, bwname, wname, lognames, ijk, realisation)

    rox.safe_close()


def export_bwell_roxapi(
    self, project, gname, bwname, wname, lognames="all", ijk=False, realisation=0
):
    """Private function for blockwell export (store in RMS) from XTGeo to RoxarAPI"""

    logger.debug("Opening RMS project ...")
    rox = RoxUtils(project, readonly=False)

    _roxapi_export_bwell(self, rox, gname, bwname, wname, lognames, ijk, realisation)

    if rox._roxexternal:
        rox.project.save()

    rox.safe_close()


def _roxapi_import_bwell(
    self,
    rox,
    gname,
    bwname,
    wname,
    lognames,
    ijk,
    realisation,
):
    """Private function for ROXAPI well import (get well from Roxar)"""

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

    if wname in bwset.get_well_names(realisation=realisation):
        self._wname = wname
    else:
        raise WellNotFoundError(f"No such well in blocked well set: {wname}")

    bwprops = list(bwset.properties)
    bwnames = [item.name for item in bwset.properties]

    bw_cellindices = bwset.get_cell_numbers(realisation=realisation)
    dind = bwset.get_data_indices([wname], realisation=realisation)

    cind = bw_cellindices[dind]
    xyz = np.transpose(gmodel.get_grid(realisation=realisation).get_cell_centers(cind))

    logs = {}
    logs[_AttrName.XNAME.value] = xyz[0].astype(np.float64)
    logs[_AttrName.YNAME.value] = xyz[1].astype(np.float64)
    logs[_AttrName.ZNAME.value] = xyz[2].astype(np.float64)

    if ijk:
        ijk = np.transpose(
            gmodel.get_grid(realisation=realisation).grid_indexer.get_indices(cind)
        )
        logs[_AttrName.I_INDEX.value] = ijk[0].astype(np.float64)
        logs[_AttrName.J_INDEX.value] = ijk[1].astype(np.float64)
        logs[_AttrName.K_INDEX.value] = ijk[2].astype(np.float64)

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
            self.create_log(
                lname, logtype=_AttrType.DISC.value, logrecord=bwprop.code_names
            )
        else:
            self.create_log(lname, logtype=_AttrType.CONT.value)

        logs[lname] = tmplog

    self.set_dataframe(pd.DataFrame.from_dict(logs))
    self._gridname = gname
    self._filesrc = None

    # finally get some other metadata like RKB and topside X Y; as they
    # seem to miss for the BW in RMS, try and get them from the
    # well itself...

    if wname in rox.project.wells:
        self._rkb = rox.project.wells[wname].rkb
        self._xpos, self._ypos = rox.project.wells[wname].wellhead
    else:
        self._rkb = None
        self._xpos, self._ypos = (
            self._df[_AttrName.XNAME.value][0],
            self._df[_AttrName.YNAME.value][0],
        )


def _roxapi_export_bwell(self, rox, gname, bwname, wname, lognames, ijk, realisation):
    """Private function for ROXAPI well export (set well with updated logs to Roxar)"""

    if gname in rox.project.grid_models:
        gmodel = rox.project.grid_models[gname]
        logger.debug("RMS grid model <%s> OK", gname)
    else:
        raise ValueError(f"No such grid name present: {gname}")

    if bwname in gmodel.blocked_wells_set:
        bwset = gmodel.blocked_wells_set[bwname]
        logger.debug("Blocked well set <%s> OK", bwname)
        bwprops = bwset.properties
    else:
        raise ValueError(f"No such blocked well set: {bwname}")

    if wname in bwset.get_well_names(realisation=realisation):
        self._wname = wname
    else:
        raise WellNotFoundError(f"No such well in blocked well set: {wname}")

    bwnames = [item.name for item in bwset.properties]  # updated list

    # get the current indices for the well
    dind = bwset.get_data_indices([self._wname], realisation=realisation)

    for lname in self.lognames:
        if not ijk and lname in [
            _AttrName.I_INDEX.value,
            _AttrName.J_INDEX.value,
            _AttrName.K_INDEX.value,
        ]:
            continue

        if lognames != "all" and lname not in lognames:
            continue

        if lname not in bwnames:
            if self.wlogtypes[lname] == _AttrType.CONT.value:
                bwlog = bwset.properties.create(
                    lname, roxar.GridPropertyType.continuous, np.float32
                )
                bwprop = bwset.generate_values(discrete=False, fill_value=0.0)
            else:
                bwlog = bwset.properties.create(
                    lname, roxar.GridPropertyType.discrete, np.int32
                )
                bwprop = bwset.generate_values(discrete=True, fill_value=0)

        else:
            bwlog = bwprops[lname]
            bwprop = bwlog.get_values(realisation=realisation)

        usedtype = bwprop.dtype
        dind = bwset.get_data_indices([self._wname], realisation=realisation)

        if self.get_dataframe(copy=False)[lname].values.size != dind.size:
            raise ValueError(
                "Dataframe is of wrong size, changing numbers of rows is not possible"
            )
        maskedvalues = np.ma.masked_invalid(
            self.get_dataframe(copy=False)[lname].values
        ).astype(usedtype)

        # there are cases where the RMS API complains on the actual value of the masked
        # array being outside range; hence remedy is to set 'data' value to a usedtype
        # dependent 'undef'. Hpwever still some flaky stuff here...
        undef = INT_MIN if "int" in str(usedtype) else np.nan
        maskedvalues.data[maskedvalues.mask] = undef

        bwprop[dind] = maskedvalues
        try:
            bwlog.set_values(bwprop, realisation=realisation)
        except ValueError as err:
            msg = (
                f"\nFailed setting blocked log for {lname} in RMS.\n"
                f"Details:\n{maskedvalues.data}\n{maskedvalues.mask}\n{usedtype}"
            )
            logger.critical(msg)
            raise err
