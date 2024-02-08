"""Utilities for Wells class"""

import logging

import numpy as np
import pandas as pd
import shapely.geometry as sg

from xtgeo.common.xtgeo_dialog import XTGeoDialog, XTGShowProgress

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

xtg = XTGeoDialog()

# self == Wells instance (plural wells)


def wellintersections(
    self,
    wfilter=None,
    showprogress=False,
):
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

        logger.debug("Work with %s", well.name)
        try:
            well.geometrics()
        except ValueError:
            logger.debug("Skip %s (cannot compute geometrics)", well.name)
            continue

        welldfr = well.get_dataframe()

        xcor = welldfr[well.xname].values
        ycor = welldfr[well.yname].values
        mcor = welldfr[well.mdlogname].values
        logger.debug("The mdlogname property is: %s", well.mdlogname)

        if xcor.size < 2:
            continue

        thisline1 = sg.LineString(np.stack([xcor, ycor], axis=1))
        thisline2 = sg.LineString(np.stack([xcor, ycor, mcor], axis=1))

        nox[well.name] = []
        # loop over other wells
        for other in self.wells:
            if other.name == well.name:
                continue  # same well

            if not well.may_overlap(other):
                nox[well.name].append(other.name)
                continue  # a quick check; no chance for overlap

            logger.debug("Consider crossing with %s ...", other.name)

            # try to be smart to skip entries that earlier have beenn tested
            # for crossing. If other does not cross well, then well does not
            # cross other...
            if other.name in nox and well.name in nox[other.name]:
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

            other_dframe = owell.get_dataframe()
            xcorc = other_dframe[well.xname].values
            ycorc = other_dframe[well.yname].values
            zcorc = other_dframe[well.zname].values

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
        xpoints,
        columns=[
            "WELL",
            "MDEPTH",
            "CWELL",
            self._wells[0].xname,
            self._wells[0].yname,
            self._wells[0].zname,
        ],
    )

    progress.finished()

    logger.debug("All intersections found!")
    return dfr
