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


def wellintersections(self, tvdrange=(None, None), fencesampling=None,
                      wfilter=None, mdepth=False):
    """Get intersections between wells, return as dataframe table.

    This routine is using shapely functions!

    If mdepth is True, then the self._mdlogname is also included,
    but only if fencesampling is None!
    """

    # collect dataframes and objects
    dfdict = {}
    dodict = {}

    if mdepth and fencesampling is not None:
        raise ValueError('mdepth cannot be True when fencesampling is applied')

    if mdepth and fencesampling is None:
        for wll in self.wells:
            if not wll._mdlogname:
                wll.geometrics()  # will make the Q_MDEPTH unless exists

    if fencesampling is None:
        for well in self._wells:
            wdfr = well.dataframe.copy()
            if tvdrange[0] is not None:
                wdfr = wdfr[wdfr.Z_TVDSS > tvdrange[0]]
            if tvdrange[1] is not None:
                wdfr = wdfr[wdfr.Z_TVDSS < tvdrange[1]]
            dfdict[well.name] = wdfr
            dodict[well.name] = well
    else:
        logger.info('Resample well fences to %s', fencesampling)
        for well in self._wells:
            poly = well.get_fence_polyline(sampling=fencesampling, extend=0,
                                           tvdmin=tvdrange[0], asnumpy=False)
            if not poly:
                continue
            wdfr = poly.dataframe

            if tvdrange[1] is not None:
                wdfr = wdfr[wdfr.Z_TVDSS < tvdrange[1]]
            dfdict[well.name] = wdfr

    xpoints = []
    for wname, dfr in dfdict.items():

        logger.info('Work with %s', wname)
        xcor = dfr['X_UTME'].values
        ycor = dfr['Y_UTMN'].values
        if mdepth:
            mdcor = dfr[dodict[wname].mdlogname].values

        if xcor.size < 2:
            continue

        thisline = sg.LineString(np.stack([xcor, ycor], axis=1))
        if mdepth:
            thisline2 = sg.LineString(np.stack([xcor, ycor, mdcor], axis=1))

        for cname, cdfr in dfdict.items():

            if cname == wname:
                continue
            # get crossing line
            xcorc = cdfr['X_UTME'].values
            ycorc = cdfr['Y_UTMN'].values
            zcorc = cdfr['Z_TVDSS'].values

            if xcorc.size < 2:
                continue

            otherline = sg.LineString(np.stack([xcorc, ycorc, zcorc], axis=1))

            if not thisline.crosses(otherline):
                continue

            ixx = thisline.intersection(otherline)

            if ixx.is_empty:
                continue

            # need to repeat if mdepth is True; need this trick to get mdepth
            if mdepth:
                other2 = sg.LineString(np.stack([xcorc, ycorc], axis=1))
                ixx2 = thisline2.intersection(other2)

            logger.debug('==> Intersects with %s', cname)
            if isinstance(ixx, sg.Point):
                xcor, ycor, zcor = ixx.coords[0]
                if mdepth:
                    _x, _y, mcor = ixx2.coords[0]
                    xpoints.append([wname, mcor, cname, xcor, ycor, zcor])
                else:
                    xpoints.append([wname, cname, xcor, ycor, zcor])

            elif isinstance(ixx, sg.MultiPoint):
                if mdepth:
                    pxx2 = list(ixx2)
                for ino, pxx in enumerate(list(ixx)):
                    xcor, ycor, zcor = pxx.coords[0]
                    if mdepth:
                        _x, _y, mcor = pxx2[ino].coords[0]
                        xpoints.append([wname, mcor, cname, xcor, ycor, zcor])
                    else:
                        xpoints.append([wname, cname, xcor, ycor, zcor])

            elif isinstance(ixx, sg.GeometryCollection):
                if mdepth:
                    gxx2 = list(ixx2)
                for ino, gxx in enumerate(list(ixx)):
                    if isinstance(gxx, sg.Point):
                        xcor, ycor, zcor = gxx.coords[0]
                        if mdepth:
                            _x, _y, mcor = gxx2[ino].coords[0]
                            xpoints.append([wname, mcor, cname, xcor, ycor,
                                            zcor])
                        else:
                            xpoints.append([wname, cname, xcor, ycor, zcor])
    if mdepth:
        dfr = pd.DataFrame(xpoints, columns=['WELL', 'MDEPTH', 'CWELL',
                                             'X_UTME', 'Y_UTMN', 'Z_TVDSS'])
    else:
        dfr = pd.DataFrame(xpoints, columns=['WELL', 'CWELL', 'X_UTME',
                                             'Y_UTMN', 'Z_TVDSS'])

    if wfilter:
        # filter away that CWELL points closer than wfilter (keep only
        # one or some)

        logger.info('Doing filtering...')

        dfrw = dfr.groupby('WELL')

        fdata = []
        for well, dfrwc in dfrw:
            logger.info('==> %s ...', well)
            dfrc = dfrwc.groupby('CWELL')

            for cwell, dfrcc in dfrc:
                if (dfrcc.X_UTME.values.size > 2):
                    dfx = dfrcc.copy()
                    dfx['DUTMX'] = dfx.X_UTME.diff()
                    dfx['DUTMY'] = dfx.Y_UTMN.diff()
                    dfx['DTVD'] = dfx.Z_TVDSS.diff()

                    dfx = dfx[(abs(dfx.DUTMX) > wfilter) |
                              (abs(dfx.DUTMY) > wfilter) |
                              (abs(dfx.DTVD) > wfilter)]

                    del dfx['DUTMX']
                    del dfx['DUTMY']
                    del dfx['DTVD']

                    fdata.append(dfx)
                else:
                    fdata.append(dfrcc)
            dfr = pd.concat(fdata, ignore_index=True).reset_index(drop=True)

    logger.info('All intersections found!')
    return dfr
