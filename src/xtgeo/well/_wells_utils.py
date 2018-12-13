# -*- coding: utf-8 -*-
"""Utilities for Wells class"""

from __future__ import print_function, absolute_import

import logging
import numpy as np
import pandas as pd
import shapely.geometry as sg

from xtgeo.common import XTGeoDialog

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

xtg = XTGeoDialog()


def wellintersections(self, tvdrange=(None, None), sampling=None):
    """Get intersections between wells, return as dataframe table.

    This routine is using shapely functions!
    """

    # collect dataframes
    dfdict = {}

    if sampling is None:
        for well in self._wells:
            wdfr = well.dataframe.copy()
            if tvdrange[0] is not None:
                wdfr = wdfr[wdfr.Z_TVDSS > tvdrange[0]]
            if tvdrange[1] is not None:
                wdfr = wdfr[wdfr.Z_TVDSS < tvdrange[1]]
            dfdict[well.name] = wdfr
    else:
        for well in self._wells:
            poly = well.get_fence_polyline(sampling=sampling, extend=0,
                                           tvdmin=tvdrange[0], asnumpy=False)
            if not poly:
                continue
            wdfr = poly.dataframe

            if tvdrange[1] is not None:
                wdfr = wdfr[wdfr.Z_TVDSS < tvdrange[1]]
            dfdict[well.name] = wdfr

    xpoints = []
    for wname, dfr in dfdict.items():

        xcor = dfr['X_UTME'].values
        ycor = dfr['Y_UTMN'].values

        thisline = sg.LineString(np.stack([xcor, ycor], axis=1))

        for cname, cdfr in dfdict.items():

            if cname == wname:
                continue
            # get crossing line
            xcor = cdfr['X_UTME'].values
            ycor = cdfr['Y_UTMN'].values
            zcor = cdfr['Z_TVDSS'].values

            otherline = sg.LineString(np.stack([xcor, ycor, zcor], axis=1))

            if not otherline.crosses(thisline):
                continue

            ixx = otherline.intersection(thisline)

            if ixx.is_empty:
                continue

            if isinstance(ixx, sg.Point):
                xcor, ycor, zcor = ixx.coords[0]
                xpoints.append([wname, cname, xcor, ycor, zcor])

            elif isinstance(ixx, sg.MultiPoint):
                for pxx in list(ixx):
                    xcor, ycor, zcor = pxx.coords[0]
                    xpoints.append([wname, cname, xcor, ycor, zcor])

            elif isinstance(ixx, sg.GeometryCollection):
                for gxx in list(ixx):
                    if isinstance(gxx, sg.Point):
                        xcor, ycor, zcor = gxx.coords[0]
                        xpoints.append([wname, cname, xcor, ycor, zcor])

    dfr = pd.DataFrame(xpoints, columns=['WELL', 'CWELL', 'UTMX',
                                         'UTMY', 'TVD'])
    return dfr
