# -*- coding: utf-8 -*-
"""Well input and output, private module for ROXAPI"""

from __future__ import print_function, absolute_import
from collections import OrderedDict

import numpy as np
import numpy.ma as npma
import pandas as pd

from xtgeo.roxutils import RoxUtils
from xtgeo.common import XTGeoDialog

xtg = XTGeoDialog()
logger = xtg.functionlogger(__name__)


# Import from ROX api
# -------------------------------------------------------------------------
def import_well_roxapi(self, project, wname, trajectory='Drilled trajectory',
                       logrun='log', lognames=None, inclmd=False,
                       inclsurvey=False):
    """Private function for loading project and ROXAPI well import"""

    rox = RoxUtils(project, readonly=True)

    _roxapi_import_well(self, rox, wname, trajectory, logrun,
                        lognames, inclmd, inclsurvey)

    rox.safe_close()


def _roxapi_import_well(self, rox, wname, traj, lrun, lognames,
                        inclmd, inclsurvey):
    """Private function for ROXAPI well import"""

    # pylint: disable=too-many-locals, too-many-branches, too-many-statements

    if wname in rox.project.wells:
        roxwell = rox.project.wells[wname]
    else:
        raise ValueError('No such well name present: {}'.format(wname))

    if traj in roxwell.wellbore.trajectories:
        roxtraj = roxwell.wellbore.trajectories[traj]
    else:
        raise ValueError('No such well traj present for {}: {}'
                         .format(wname, traj))

    if lrun in roxtraj.log_runs:
        roxlrun = roxtraj.log_runs[lrun]
    else:
        raise ValueError('No such logrun present for {}: {}'
                         .format(wname, lrun))

    self._wlogtype = dict()  # 'DISC' or 'CONT'
    self._wlogrecord = dict()  # None for CONT, a dict for DISC

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
            logger.warning('MD is %s, surveyinterpolation fails, '
                           'CHECK RESULT!', mdv)
            geo_array[ino] = geo_array[ino - 1]

    self._wlogtype = dict()
    self._wlogrecord = dict()
    logs = OrderedDict()
    logs['X_UTME'] = geo_array[:, 3]
    logs['Y_UTMN'] = geo_array[:, 4]
    logs['Z_TVDSS'] = geo_array[:, 5]
    if inclmd or inclsurvey:
        logs['M_MDEPTH'] = geo_array[:, 0]
        self._mdlogname = 'M_MDEPTH'
    if inclsurvey:
        logs['M_INCL'] = geo_array[:, 1]
        logs['M_AZI'] = geo_array[:, 2]

    if lognames and lognames == 'all':
        for logcurv in roxlrun.log_curves:
            lname = logcurv.name
            logs[lname] = _get_roxlog(self, roxlrun, lname)
    elif lognames:
        for lname in lognames:
            if lname in roxlrun.log_curves:
                logs[lname] = _get_roxlog(self, roxlrun, lname)
            else:
                validlogs = [logname.name for logname in roxlrun.log_curves]
                raise ValueError('Could not get log name {}, validlogs are {}'
                                 .format(lname, validlogs))

    self._rkb = roxwell.rkb
    self._xpos, self._ypos = roxwell.wellhead
    self._wname = wname
    self._df = pd.DataFrame.from_dict(logs)


def _get_roxlog(self, roxlrun, lname):
    roxcurve = roxlrun.log_curves[lname]
    tmplog = roxcurve.get_values().astype(np.float64)
    tmplog = npma.filled(tmplog, fill_value=np.nan)
    tmplog[tmplog == -999] = np.nan
    if roxcurve.is_discrete:
        self._wlogtype[lname] = 'DISC'
        self._wlogrecord[lname] = roxcurve.get_code_names()
    else:
        self._wlogtype[lname] = 'CONT'
        self._wlogrecord[lname] = None

    return tmplog
