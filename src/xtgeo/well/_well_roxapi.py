# -*- coding: utf-8 -*-
"""Well input and output, private module for ROXAPI."""

import numpy as np
import numpy.ma as npma
import pandas as pd

from xtgeo.common import XTGeoDialog, logger
from xtgeo.common._xyz_enum import _AttrName, _AttrType
from xtgeo.common.constants import UNDEF_INT_LIMIT, UNDEF_LIMIT
from xtgeo.roxutils import RoxUtils

# Well() instance: self


# Import from ROX api
# --------------------------------------------------------------------------------------
def import_well_roxapi(
    project,
    wname,
    trajectory="Drilled trajectory",
    logrun="log",
    lognames="all",
    lognames_strict=False,
    inclmd=False,
    inclsurvey=False,
):  # pragma: no cover
    """Private function for loading project and ROXAPI well import."""
    rox = RoxUtils(project, readonly=True)

    result = _roxapi_import_well(
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
    return result


def _roxapi_import_well(
    rox, wname, traj, lrun, lognames, lognames_strict, inclmd, inclsurvey
):  # pragma: no cover
    """Private function for ROXAPI well import."""
    if wname in rox.project.wells:
        roxwell = rox.project.wells[wname]
    else:
        raise ValueError(f"No such well name present: {wname}")

    if traj in roxwell.wellbore.trajectories:
        roxtraj = roxwell.wellbore.trajectories[traj]
    else:
        raise ValueError(f"No such well traj present for {wname}: {traj}")

    if lrun in roxtraj.log_runs:
        roxlrun = roxtraj.log_runs[lrun]
    else:
        raise ValueError(f"No such logrun present for {wname}: {lrun}")

    wlogtypes = dict()
    wlogrecords = dict()

    # get logs repr trajecetry
    mdlogname, logs = _roxapi_traj(roxtraj, roxlrun, inclmd, inclsurvey)

    if lognames and lognames == "all":
        for logcurv in roxlrun.log_curves:
            lname = logcurv.name
            logs[lname] = _get_roxlog(wlogtypes, wlogrecords, roxlrun, lname)
    elif lognames:
        for lname in lognames:
            if lname in roxlrun.log_curves:
                logs[lname] = _get_roxlog(wlogtypes, wlogrecords, roxlrun, lname)
            else:
                if lognames_strict:
                    validlogs = [logname.name for logname in roxlrun.log_curves]
                    raise ValueError(
                        f"Could not get log name {lname}, validlogs are {validlogs}"
                    )

    return {
        "rkb": roxwell.rkb,
        "xpos": roxwell.wellhead[0],
        "ypos": roxwell.wellhead[1],
        "wname": wname,
        "wlogtypes": wlogtypes,
        "wlogrecords": wlogrecords,
        "mdlogname": mdlogname,
        "df": pd.DataFrame.from_dict(logs),
    }


def _roxapi_traj(roxtraj, roxlrun, inclmd, inclsurvey):  # pragma: no cover
    """Get trajectory in ROXAPI."""

    surveyset = roxtraj.survey_point_series
    measured_depths = roxlrun.get_measured_depths()

    mds = measured_depths.tolist()

    geo_array_shape = (len(measured_depths), 6)
    geo_array = np.empty(geo_array_shape)

    for ino, mdv in enumerate(mds):
        try:
            geo_array[ino] = surveyset.interpolate_survey_point(mdv)
        except ValueError:
            logger.warning("MD is %s, surveyinterpolation fails, CHECK RESULT!", mdv)
            geo_array[ino] = geo_array[ino - 1]

    logs = dict()
    mdlogname = None

    logs[_AttrName.XNAME.value] = geo_array[:, 3]
    logs[_AttrName.YNAME.value] = geo_array[:, 4]
    logs[_AttrName.ZNAME.value] = geo_array[:, 5]
    if inclmd or inclsurvey:
        logs[_AttrName.M_MD_NAME.value] = geo_array[:, 0]
        mdlogname = _AttrName.M_MD_NAME.value
    if inclsurvey:
        logs[_AttrName.M_INCL_NAME.value] = geo_array[:, 1]
        logs[_AttrName.M_AZI_NAME.value] = geo_array[:, 2]

    return mdlogname, logs


def _get_roxlog(wlogtypes, wlogrecords, roxlrun, lname):  # pragma: no cover
    roxcurve = roxlrun.log_curves[lname]
    tmplog = roxcurve.get_values().astype(np.float64)
    tmplog = npma.filled(tmplog, fill_value=np.nan)
    tmplog[tmplog == -999] = np.nan
    if roxcurve.is_discrete:
        wlogtypes[lname] = _AttrType.DISC.value
        wlogrecords[lname] = roxcurve.get_code_names()
    else:
        wlogtypes[lname] = _AttrType.CONT.value
        wlogrecords[lname] = None

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
    """Private function for well export (store in RMS) from XTGeo to RoxarAPI."""
    logger.debug("Opening RMS project ...")

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
    logger.debug("Key realisation not in use: %s", realisation)

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
        xtglimit = UNDEF_LIMIT
        if self.wlogtypes[lname] == _AttrType.DISC.value:
            isdiscrete = True
            xtglimit = UNDEF_INT_LIMIT

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
            if vals.mask.all():
                codedict = {0: "unset"}
            else:
                codedict = {
                    int(key): str(value)
                    for key, value in self._wlogrecords[lname].items()
                }
            thelog.set_code_names(codedict)


def _roxapi_create_well(self, rox, wname, lognames, logrun, trajectory, realisation):
    """Save Well() instance to a new well in RMS.

    From version 2.15.
    """
    logger.debug("Key realisation is not supported: %s", realisation)

    roxwell = rox.project.wells.create(wname)
    roxwell.rkb = self.rkb
    roxwell.wellhead = (self.xpos, self.ypos)

    traj = roxwell.wellbore.trajectories.create(trajectory)

    series = traj.survey_point_series
    east = self.dataframe[self.xname].values
    north = self.dataframe[self.yname].values
    tvd = self.dataframe[self.zname].values
    values = np.array([east, north, tvd]).transpose()
    series.set_points(values)

    md = series.get_measured_depths_and_points()[:, 0]

    lrun = traj.log_runs.create(logrun)
    lrun.set_measured_depths(md)

    # Add log curves
    for curvename, curveprop in self.get_wlogs().items():
        if curvename not in self.lognames:
            continue  # skip X_UTME .. Z_TVDSS
        if lognames and lognames != "all" and curvename not in lognames:
            continue
        if not lognames:
            continue

        cname = curvename
        if curvename == "MD":
            cname = "MD_imported"
            xtg.warn(f"Logname MD is renamed to {cname}")

        if curveprop[0] == _AttrType.DISC.value:
            lcurve = lrun.log_curves.create_discrete(cname)
            cc = np.ma.masked_invalid(self.dataframe[curvename].values)
            lcurve.set_values(cc.astype(np.int32))
            codedict = {int(key): str(value) for key, value in curveprop[1].items()}
            lcurve.set_code_names(codedict)
        else:
            lcurve = lrun.log_curves.create(cname)
            lcurve.set_values(self.dataframe[curvename].values)

        logger.debug("Log curve created: %s", cname)
