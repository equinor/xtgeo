# -*- coding: utf-8 -*-
"""Utilities for Wells class"""


import logging
import numpy as np
import pandas as pd
import shapely.geometry as sg

from xtgeo.common import XTGeoDialog
from xtgeo.common import XTGShowProgress

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

xtg = XTGeoDialog()


def wellintersections(
    self, wfilter=None, showprogress=False
):  # pylint: disable=too-many-locals, too-many-branches, too-many-statements
    """Get intersections between wells, return as dataframe table.

    This routine is using "shapely" functions!

    Some actions are done in order to filter away the part of the trajectories
    that are paralell.

    """

    xpoints = []

    # make a dict if nocrossings
    nox = {}

    wlen = len(self.wells)

    progress = XTGShowProgress(wlen, show=showprogress, leadtext="progress: ", skip=5)

    for iwell, well in enumerate(self.wells):

        progress.flush(iwell)

        gstatus = well.geometrics()

        logger.info("Work with %s", well.name)
        if not gstatus:
            logger.info("Skip %s (cannot compute geometrics)", well.name)
            continue

        welldfr = well.dataframe.copy()

        xcor = welldfr["X_UTME"].values
        ycor = welldfr["Y_UTMN"].values
        zcor = welldfr["Z_TVDSS"].values
        mcor = welldfr[well.mdlogname].values
        logger.info("The mdlogname property is: %s", well.mdlogname)

        if xcor.size < 2:
            continue

        thisline1 = sg.LineString(np.stack([xcor, ycor], axis=1))
        thisline2 = sg.LineString(np.stack([xcor, ycor, mcor], axis=1))

        nox[well.name] = list()
        # loop over other wells
        for other in self.wells:

            if other.name == well.name:
                continue  # same well

            if not well.may_overlap(other):
                nox[well.name].append(other.name)
                continue  # a quick check; no chance for overlap

            logger.info("Consider crossing with %s ...", other.name)

            # try to be smart to skip entries that earlier have beenn tested
            # for crossing. If other does not cross well, then well does not
            # cross other...
            if other.name in nox.keys() and well.name in nox[other.name]:
                continue

            # truncate away the paralell part on a copy
            owell = other.copy()

            # wfilter = None
            if wfilter is not None and "parallel" in wfilter:
                xtol = wfilter["parallel"].get("xtol")
                ytol = wfilter["parallel"].get("ytol")
                ztol = wfilter["parallel"].get("ztol")
                itol = wfilter["parallel"].get("itol")
                atol = wfilter["parallel"].get("atol")
                owell.truncate_parallel_path(
                    well, xtol=xtol, ytol=ytol, ztol=ztol, itol=itol, atol=atol
                )

            xcorc = owell.dataframe["X_UTME"].values
            ycorc = owell.dataframe["Y_UTMN"].values
            zcorc = owell.dataframe["Z_TVDSS"].values

            if xcorc.size < 2:
                continue

            otherline = sg.LineString(np.stack([xcorc, ycorc, zcorc], axis=1))

            if not thisline1.crosses(otherline):
                nox[well.name].append(other.name)
                continue

            ixx = thisline1.intersection(otherline)

            if ixx.is_empty:
                nox[well.name].append(other.name)
                continue

            # need this trick to get mdepth
            other2 = sg.LineString(np.stack([xcorc, ycorc], axis=1))
            ixx2 = thisline2.intersection(other2)

            logger.debug("==> Intersects with %s", other.name)

            if isinstance(ixx, sg.Point):
                xcor, ycor, zcor = ixx.coords[0]
                _x, _y, mcor = ixx2.coords[0]
                xpoints.append([well.name, mcor, other.name, xcor, ycor, zcor])

            elif isinstance(ixx, sg.MultiPoint):
                pxx2 = list(ixx2)
                for ino, pxx in enumerate(list(ixx)):
                    xcor, ycor, zcor = pxx.coords[0]
                    _x, _y, mcor = pxx2[ino].coords[0]
                    xpoints.append([well.name, mcor, other.name, xcor, ycor, zcor])

            elif isinstance(ixx, sg.GeometryCollection):
                gxx2 = list(ixx2)
                for ino, gxx in enumerate(list(ixx)):
                    if isinstance(gxx, sg.Point):
                        xcor, ycor, zcor = gxx.coords[0]
                        _x, _y, mcor = gxx2[ino].coords[0]
                        xpoints.append([well.name, mcor, other.name, xcor, ycor, zcor])

    dfr = pd.DataFrame(
        xpoints, columns=["WELL", "MDEPTH", "CWELL", "X_UTME", "Y_UTMN", "Z_TVDSS"]
    )

    progress.finished()

    logger.info("All intersections found!")
    return dfr
