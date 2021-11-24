# -*- coding: utf-8 -*-
"""Well input and output, private module for ROXAPI."""


from collections import OrderedDict

import numpy as np
import numpy.ma as npma
import pandas as pd

import xtgeo
from xtgeo.common import XTGeoDialog
from xtgeo.roxutils import RoxUtils

xtg = XTGeoDialog()
logger = xtg.functionlogger(__name__)

# Well() instance self = xwell1


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
        raise ValueError("No such well name present: {}".format(wname))

    if traj in roxwell.wellbore.trajectories:
        roxtraj = roxwell.wellbore.trajectories[traj]
    else:
        raise ValueError("No such well traj present for {}: {}".format(wname, traj))

    if lrun in roxtraj.log_runs:
        roxlrun = roxtraj.log_runs[lrun]
    else:
        raise ValueError("No such logrun present for {}: {}".format(wname, lrun))

    wlogtypes = dict()
    wlogrecords = dict()

    # get logs repr trajecetry
    mdlogname, logs = _roxapi_traj(
        wlogtypes, wlogrecords, roxtraj, roxlrun, inclmd, inclsurvey
    )

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
                        "Could not get log name {}, validlogs are {}".format(
                            lname, validlogs
                        )
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


def _roxapi_traj(
    wlogtypes, wlogrecords, roxtraj, roxlrun, inclmd, inclsurvey
):  # pragma: no cover
    """Get trajectory in ROXAPI."""
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

    logs = OrderedDict()
    mdlogname = None

    logs["X_UTME"] = geo_array[:, 3]
    logs["Y_UTMN"] = geo_array[:, 4]
    logs["Z_TVDSS"] = geo_array[:, 5]
    if inclmd or inclsurvey:
        logs["M_MDEPTH"] = geo_array[:, 0]
        mdlogname = "M_MDEPTH"
    if inclsurvey:
        logs["M_INCL"] = geo_array[:, 1]
        logs["M_AZI"] = geo_array[:, 2]

    return mdlogname, logs


def _get_roxlog(wlogtypes, wlogrecords, roxlrun, lname):  # pragma: no cover
    roxcurve = roxlrun.log_curves[lname]
    tmplog = roxcurve.get_values().astype(np.float64)
    tmplog = npma.filled(tmplog, fill_value=np.nan)
    tmplog[tmplog == -999] = np.nan
    if roxcurve.is_discrete:
        wlogtypes[lname] = "DISC"
        wlogrecords[lname] = roxcurve.get_code_names()
    else:
        wlogtypes[lname] = "CONT"
        wlogrecords[lname] = None

    return tmplog


def export_well_roxapi(
    xwell1,
    project,
    wname,
    lognames="all",
    logrun="log",
    trajectory="Drilled trajectory",
    realisation=0,
):
    """Private function for well export (store in RMS) from XTGeo to RoxarAPI."""
    logger.info("Opening RMS project ...")

    rox = RoxUtils(project, readonly=False)

    _roxapi_export_well(xwell1, rox, wname, lognames, logrun, trajectory, realisation)

    if rox._roxexternal:
        rox.project.save()

    rox.safe_close()


def _roxapi_export_well(xwell1, rox, wname, lognames, logrun, trajectory, realisation):

    if wname in rox.project.wells:
        _roxapi_update_well(
            xwell1, rox, wname, lognames, logrun, trajectory, realisation
        )
    else:
        _roxapi_create_well(
            xwell1, rox, wname, lognames, logrun, trajectory, realisation
        )


def _roxapi_update_well(xwell1, rox, wname, lognames, logrun, trajectory, realisation):
    """Assume well is to updated only with logs, new or changed.

    Also, the length of arrays should not change, at least not for now.

    """
    logger.info("Key realisation not in use: %s", realisation)

    well = rox.project.wells[wname]
    traj = well.wellbore.trajectories[trajectory]
    lrun = traj.log_runs[logrun]

    lrun.log_curves.clear()

    if lognames == "all":
        uselognames = xwell1.lognames
    else:
        uselognames = lognames

    for lname in uselognames:

        isdiscrete = False
        xtglimit = xtgeo.UNDEF_LIMIT
        if xwell1._wlogtypes[lname] == "DISC":
            isdiscrete = True
            xtglimit = xtgeo.UNDEF_INT_LIMIT

        if isdiscrete:
            thelog = lrun.log_curves.create_discrete(name=lname)
        else:
            thelog = lrun.log_curves.create(name=lname)

        values = thelog.generate_values()

        if values.size != xwell1.dataframe[lname].values.size:
            raise ValueError("New logs have different sampling or size, not possible")

        usedtype = values.dtype

        vals = np.ma.masked_invalid(xwell1.dataframe[lname].values)
        vals = np.ma.masked_greater(vals, xtglimit)
        vals = vals.astype(usedtype)
        thelog.set_values(vals)

        if isdiscrete:
            # roxarapi requires keys to int, while xtgeo can accept any, e.g. strings
            codedict = {
                int(key): str(value)
                for key, value in xwell1._wlogrecords[lname].items()
            }
            thelog.set_code_names(codedict)


def _roxapi_create_well(xwell1, rox, wname, lognames, logrun, trajectory, realisation):
    """Save Well() instance to a new well in RMS.

    From version 2.15.
    """
    logger.debug("Key realisation is not supported: %s", realisation)

    roxwell = rox.project.wells.create(wname)
    roxwell.rkb = xwell1.rkb
    roxwell.wellhead = (xwell1.xpos, xwell1.ypos)

    traj = roxwell.wellbore.trajectories.create(trajectory)

    series = traj.survey_point_series
    east = xwell1.dataframe["X_UTME"].values
    north = xwell1.dataframe["Y_UTMN"].values
    tvd = xwell1.dataframe["Z_TVDSS"].values
    values = np.array([east, north, tvd]).transpose()
    series.set_points(values)

    md = series.get_measured_depths_and_points()[:, 0]

    lrun = traj.log_runs.create(logrun)
    lrun.set_measured_depths(md)

    # Add log curves
    for curvename, curveprop in xwell1.get_wlogs().items():
        if curvename not in xwell1.lognames:
            continue  # skip X_UTME .. Z_TVDSS
        if lognames and lognames != "all" and curvename not in lognames:
            continue
        if not lognames:
            continue

        cname = curvename
        if curvename == "MD":
            cname = "MD_imported"
            xtg.warn(f"Logname MD is renamed to {cname}")

        if curveprop[0] == "DISC":
            lcurve = lrun.log_curves.create_discrete(cname)
            cc = np.ma.masked_invalid(xwell1.dataframe[curvename].values)
            lcurve.set_values(cc.astype(np.int32))
            codedict = {int(key): str(value) for key, value in curveprop[1].items()}
            lcurve.set_code_names(codedict)
        else:
            lcurve = lrun.log_curves.create(cname)
            lcurve.set_values(xwell1.dataframe[curvename].values)

        logger.info("Log curve created: %s", cname)
