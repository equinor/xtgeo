"""Well input and output, private module for ROXAPI."""

from __future__ import annotations

import contextlib
import functools
from datetime import datetime
from typing import Any, Literal, Optional

import numpy as np
import numpy.ma as npma
import pandas as pd

import xtgeo
from xtgeo.common import XTGeoDialog, null_logger
from xtgeo.common._xyz_enum import _AttrName, _AttrType
from xtgeo.common.constants import UNDEF_INT_LIMIT, UNDEF_LIMIT
from xtgeo.common.pandas_extensions import LazyArray
from xtgeo.roxutils import RoxUtils

xtg = XTGeoDialog()
logger = null_logger(__name__)


@contextlib.contextmanager
def timer():
    enter = datetime.now()
    done = None
    try:
        yield lambda: ((done or datetime.now()) - enter).total_seconds()
    finally:
        done = datetime.now()


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

    wlogtypes = {}
    wlogrecords = {}

    # get logs repr trajecetry
    mdlogname, nele, logs = _roxapi_traj(roxtraj, roxlrun, inclmd, inclsurvey)
    if lognames and lognames == "all":
        for logcurv in roxlrun.log_curves:
            # logs[lname] = _get_roxlog(wlogtypes, wlogrecords, roxlrun, lname)
            logs[logcurv.name] = functools.partial(
                _get_roxlog,
                wlogtypes,
                wlogrecords,
                roxlrun,
                logcurv.name,
            )
    elif lognames:
        for lname in lognames:
            if lname in roxlrun.log_curves:
                # logs[lname] = _get_roxlog(wlogtypes, wlogrecords, roxlrun, lname)
                logs[lname] = functools.partial(
                    _get_roxlog,
                    wlogtypes,
                    wlogrecords,
                    roxlrun,
                    lname,
                )
            else:
                if lognames_strict:
                    validlogs = [logname.name for logname in roxlrun.log_curves]
                    raise ValueError(
                        f"Could not get log name {lname}, validlogs are {validlogs}"
                    )
    # for logcurv in roxlrun.log_curves:
    # logs[logcurv.name] = functools.partial(
    #     _get_roxlog,
    #     wlogtypes,
    #     wlogrecords,
    #     roxlrun,
    #     logcurv.name,
    # )

    return {
        "rkb": roxwell.rkb,
        "xpos": roxwell.wellhead[0],
        "ypos": roxwell.wellhead[1],
        "wname": wname,
        "wlogtypes": wlogtypes,
        "wlogrecords": wlogrecords,
        "mdlogname": mdlogname,
        "df": pd.DataFrame.from_dict({k: LazyArray(v, nele) for k, v in logs.items()}),
    }


def _roxapi_traj(roxtraj, roxlrun, inclmd: bool, inclsurvey: bool):  # pragma: no cover
    """Get trajectory in ROXAPI."""

    surveyset_ = None
    measured_depths_ = None
    shape = next(roxlrun.log_curves.values()).shape[0]
    interpolated_survey_points_ = np.zeros((shape, 6))

    def interpolate_survey_point(idx: int):
        nonlocal surveyset_, measured_depths_
        if surveyset_ is None and measured_depths_ is None:
            surveyset_ = roxtraj.survey_point_series
            measured_depths_ = roxlrun.get_measured_depths()
            for i, p in enumerate(measured_depths_):
                interpolated_survey_points_[i] = surveyset_.interpolate_survey_point(p)
        return interpolated_survey_points_[:, idx]

    logs = {}

    # Callabole key/values(callabole)
    # XYZ must be first, see _ensure_consistency_attr_types(...)
    logs[_AttrName.XNAME.value] = lambda: interpolate_survey_point(3)
    logs[_AttrName.YNAME.value] = lambda: interpolate_survey_point(4)
    logs[_AttrName.ZNAME.value] = lambda: interpolate_survey_point(5)

    mdlogname = None

    if inclmd or inclsurvey:
        logs[_AttrName.M_MD_NAME.value] = lambda: interpolate_survey_point(0)
        mdlogname = _AttrName.M_MD_NAME.value
    if inclsurvey:
        logs[_AttrName.M_INCL_NAME.value] = lambda: interpolate_survey_point(1)
        logs[_AttrName.M_AZI_NAME.value] = lambda: interpolate_survey_point(2)

    return shape, mdlogname, logs


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
    self: xtgeo.Well,
    project,
    wname,
    lognames: str | list[str] = "all",
    logrun: str = "log",
    trajectory: str = "Drilled trajectory",
    realisation: int = 0,
    update_option: Optional[Literal["overwrite", "append"]] = None,
):
    """Private function for well export (i.e. store in RMS) from XTGeo to RoxarAPI."""
    logger.debug("Opening RMS project ...")

    rox = RoxUtils(project, readonly=False)

    if wname in rox.project.wells:
        _roxapi_update_well(
            self, rox, wname, lognames, logrun, trajectory, realisation, update_option
        )
    else:
        _roxapi_create_well(self, rox, wname, lognames, logrun, trajectory, realisation)

    if rox._roxexternal:
        rox.project.save()

    rox.safe_close()


def _store_log_in_roxapi(self, lrun: Any, logname: str) -> None:
    """Store a single log in RMS / Roxar API for a well"""
    if logname in (self.xname, self.yname, self.zname):
        return

    isdiscrete = False
    xtglimit = UNDEF_LIMIT
    if self.wlogtypes[logname] == _AttrType.DISC.value:
        isdiscrete = True
        xtglimit = UNDEF_INT_LIMIT

    store_logname = logname

    # the MD name is applied as default in RMS for measured depth; hence it can be wise
    # to avoid duplicate names here, since the measured depth log is crucial.
    if logname == "MD":
        store_logname = "MD_imported"
        xtg.warn(f"Logname MD is stored as {store_logname}")

    if isdiscrete:
        thelog = lrun.log_curves.create_discrete(name=store_logname)
    else:
        thelog = lrun.log_curves.create(name=store_logname)

    values = thelog.generate_values()

    if values.size != self.get_dataframe(copy=False)[logname].values.size:
        raise ValueError("New logs have different sampling or size, not possible")

    usedtype = values.dtype

    vals = np.ma.masked_invalid(self.get_dataframe(copy=False)[logname].values)
    vals = np.ma.masked_greater(vals, xtglimit)
    vals = vals.astype(usedtype)
    thelog.set_values(vals)

    if isdiscrete:
        # roxarapi requires keys to be ints, while xtgeo can accept any, e.g. strings
        if vals.mask.all():
            codedict = {0: "unset"}
        else:
            codedict = {
                int(key): str(value) for key, value in self.wlogrecords[logname].items()
            }
        thelog.set_code_names(codedict)


def _roxapi_update_well(
    self: xtgeo.Well,
    rox: Any,
    wname: str,
    lognames: str | list[str],
    logrun: str,
    trajectory: str,
    realisation: int,
    update_option: Optional[Literal["overwrite", "append"]] = None,
):
    """Assume well is to updated only with logs, new only are appended

    Also, the length of arrays are not allowed not change (at least not for now).

    """
    logger.debug("Key realisation not in use: %s", realisation)
    if update_option not in (None, "overwrite", "append"):
        raise ValueError(
            f"The update_option <{update_option}> is invalid, valid "
            "options are: None | overwrite | append"
        )

    lrun = rox.project.wells[wname].wellbore.trajectories[trajectory].log_runs[logrun]

    # find existing lognames in target
    current_logs = [lname.name for lname in lrun.log_curves]

    uselognames = self.lognames if lognames == "all" else lognames

    if update_option is None:
        lrun.log_curves.clear()  # clear existing logs; all will be replaced

    for lname in uselognames:
        if update_option == "append" and lname in current_logs:
            continue
        _store_log_in_roxapi(self, lrun, lname)


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

    dataframe = self.get_dataframe(copy=False)
    east = dataframe[self.xname].values
    north = dataframe[self.yname].values
    tvd = dataframe[self.zname].values
    values = np.array([east, north, tvd]).transpose()
    series.set_points(values)

    md = series.get_measured_depths_and_points()[:, 0]

    lrun = traj.log_runs.create(logrun)
    lrun.set_measured_depths(md)

    uselognames = self.lognames if lognames == "all" else lognames
    for lname in uselognames:
        _store_log_in_roxapi(self, lrun, lname)
