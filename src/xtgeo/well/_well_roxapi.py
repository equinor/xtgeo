# -*- coding: utf-8 -*-
"""Well input and output, private module for ROXAPI"""

from __future__ import print_function, absolute_import
from collections import OrderedDict

import numpy as np
import numpy.ma as npma
import pandas as pd

import xtgeo
from xtgeo.roxutils import RoxUtils
from xtgeo.common import XTGeoDialog

xtg = XTGeoDialog()
logger = xtg.functionlogger(__name__)


# Import from ROX api
# --------------------------------------------------------------------------------------
def import_well_roxapi(
    self,
    project,
    wname,
    trajectory="Drilled trajectory",
    logrun="log",
    lognames="all",
    lognames_strict=False,
    inclmd=False,
    inclsurvey=False,
):  # pragma: no cover
    """Private function for loading project and ROXAPI well import"""

    rox = RoxUtils(project, readonly=True)

    _roxapi_import_well(
        self,
        rox,
        wname,
        trajectory,
        logrun,
        lognames,
        lognames_strict,
        inclmd,
        inclsurvey,
    )

    rox.safe_close()


def _roxapi_import_well(
    self, rox, wname, traj, lrun, lognames, lognames_strict, inclmd, inclsurvey
):  # pragma: no cover
    """Private function for ROXAPI well import"""

    if wname in rox.project.wells:
        roxwell = rox.project.wells[wname]
    else:
        raise ValueError("No such well name present: {}".format(wname))

    if traj in roxwell.wellbore.trajectories:
        roxtraj = roxwell.wellbore.trajectories[traj]
    else:
        raise ValueError("No such well traj present for {}: {}".format(wname, traj))

    if lrun in roxtraj.log_runs:
        roxlrun = roxtraj.log_runs[lrun]
    else:
        raise ValueError("No such logrun present for {}: {}".format(wname, lrun))

    # get logs repr trajecetry
    logs = _roxapi_traj(self, roxtraj, roxlrun, inclmd, inclsurvey)

    if lognames and lognames == "all":
        for logcurv in roxlrun.log_curves:
            lname = logcurv.name
            logs[lname] = _get_roxlog(self, roxlrun, lname)
    elif lognames:
        for lname in lognames:
            if lname in roxlrun.log_curves:
                logs[lname] = _get_roxlog(self, roxlrun, lname)
            else:
                if lognames_strict:
                    validlogs = [logname.name for logname in roxlrun.log_curves]
                    raise ValueError(
                        "Could not get log name {}, validlogs are {}".format(
                            lname, validlogs
                        )
                    )

    self._rkb = roxwell.rkb
    self._xpos, self._ypos = roxwell.wellhead
    self._wname = wname
    self._df = pd.DataFrame.from_dict(logs)


def _roxapi_traj(self, roxtraj, roxlrun, inclmd, inclsurvey):  # pragma: no cover
    """Get trajectory in ROXAPI"""
    # compute trajectory

    surveyset = roxtraj.survey_point_series
    measured_depths = roxlrun.get_measured_depths()

    mds = measured_depths.tolist()

    geo_array_shape = (len(measured_depths), 6)
    geo_array = np.empty(geo_array_shape)

    for ino, mdv in enumerate(mds):
        try:
            geo_array[ino] = surveyset.interpolate_survey_point(mdv)
        except ValueError:
            logger.warning("MD is %s, surveyinterpolation fails, " "CHECK RESULT!", mdv)
            geo_array[ino] = geo_array[ino - 1]

    self._wlogtype = dict()
    self._wlogrecord = dict()
    logs = OrderedDict()

    logs["X_UTME"] = geo_array[:, 3]
    logs["Y_UTMN"] = geo_array[:, 4]
    logs["Z_TVDSS"] = geo_array[:, 5]
    if inclmd or inclsurvey:
        logs["M_MDEPTH"] = geo_array[:, 0]
        self._mdlogname = "M_MDEPTH"
    if inclsurvey:
        logs["M_INCL"] = geo_array[:, 1]
        logs["M_AZI"] = geo_array[:, 2]

    return logs


def _get_roxlog(self, roxlrun, lname):  # pragma: no cover
    roxcurve = roxlrun.log_curves[lname]
    tmplog = roxcurve.get_values().astype(np.float64)
    tmplog = npma.filled(tmplog, fill_value=np.nan)
    tmplog[tmplog == -999] = np.nan
    if roxcurve.is_discrete:
        self._wlogtype[lname] = "DISC"
        self._wlogrecord[lname] = roxcurve.get_code_names()
    else:
        self._wlogtype[lname] = "CONT"
        self._wlogrecord[lname] = None

    return tmplog


def export_well_roxapi(
    self,
    project,
    wname,
    lognames="all",
    logrun="log",
    trajectory="Drilled trajectory",
    realisation=0,
):
    """Private function for well export (store in RMS) from XTGeo to RoxarAPI"""

    logger.info("Opening RMS project ...")
    rox = RoxUtils(project, readonly=False)

    _roxapi_export_well(self, rox, wname, lognames, logrun, trajectory, realisation)

    if rox._roxexternal:
        rox.project.save()

    rox.safe_close()


def _roxapi_export_well(self, rox, wname, lognames, logrun, trajectory, realisation):

    if wname in rox.project.wells:
        _roxapi_update_well(self, rox, wname, lognames, logrun, trajectory, realisation)
    else:
        _roxapi_create_well(self, rox, wname, lognames, logrun, trajectory, realisation)


def _roxapi_update_well(self, rox, wname, lognames, logrun, trajectory, realisation):
    """Assume well is to updated only with logs, new or changed.

    Also, the length of arrays should not change, at least not for now.

    """

    logger.info("Not in use: %s", realisation)

    well = rox.project.wells[wname]
    traj = well.wellbore.trajectories[trajectory]
    lrun = traj.log_runs[logrun]

    lrun.log_curves.clear()

    if lognames == "all":
        uselognames = self.lognames
    else:
        uselognames = lognames

    for lname in uselognames:

        isdiscrete = False
        xtglimit = xtgeo.UNDEF_LIMIT
        if self._wlogtype[lname] == "DISC":
            isdiscrete = True
            xtglimit = xtgeo.UNDEF_INT_LIMIT

        if isdiscrete:
            thelog = lrun.log_curves.create_discrete(name=lname)
        else:
            thelog = lrun.log_curves.create(name=lname)

        values = thelog.generate_values()

        if values.size != self.dataframe[lname].values.size:
            raise ValueError("New logs have different sampling or size, not possible")

        usedtype = values.dtype

        vals = np.ma.masked_invalid(self.dataframe[lname].values)
        vals = np.ma.masked_greater(vals, xtglimit)
        vals = vals.astype(usedtype)
        thelog.set_values(vals)

        if isdiscrete:
            # roxarapi requires keys to int, while xtgeo can accept any, e.g. strings
            codedict = {
                int(key): str(value) for key, value in self._wlogrecord[lname].items()
            }
            thelog.set_code_names(codedict)


def _roxapi_create_well(self, rox, wname, lognames, logrun, trajectory, realisation):

    logger.info("Not in use: %s, %s, %s, %s, %s", self, rox, wname, lognames, logrun)
    logger.info("Not in use: %s, %s ", trajectory, realisation)

    raise NotImplementedError("In prep")
