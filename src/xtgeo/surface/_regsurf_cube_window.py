# -*- coding: utf-8 -*-
"""Regular surface vs Cube, slice a window interval"""


import numpy as np
import numpy.ma as ma
from xtgeo.common import XTGeoDialog
from xtgeo.common import XTGShowProgress
from . import _regsurf_cube_window_v2 as cwv

xtg = XTGeoDialog()

logger = xtg.functionlogger(__name__)


ALLATTRS = [
    "max",
    "min",
    "rms",
    "mean",
    "var",
    "maxpos",
    "maxneg",
    "maxabs",
    "sumpos",
    "sumneg",
    "sumabs",
    "meanabs",
    "meanpos",
    "meanneg",
]


def slice_cube_window(
    self,
    cube,
    zsurf=None,
    other=None,
    other_position="below",
    sampling="nearest",
    mask=True,
    zrange=10,
    ndiv=None,
    attribute="max",
    maskthreshold=0.1,
    snapxy=False,
    showprogress=False,
    deadtraces=True,
    algorithm=1,
):
    if algorithm == 1:

        if sampling == "cube":
            sampling = "nearest"

        attrs = _slice_cube_window_v1(
            self,
            cube,
            zsurf=zsurf,
            other=other,
            other_position=other_position,
            sampling=sampling,
            mask=mask,
            zrange=zrange,
            ndiv=ndiv,
            attribute=attribute,
            maskthreshold=maskthreshold,
            snapxy=snapxy,
            showprogress=showprogress,
            deadtraces=deadtraces,
        )

    else:
        attrs = cwv.slice_cube_window(
            self,
            cube,
            zsurf=zsurf,
            other=other,
            other_position=other_position,
            sampling=sampling,
            mask=mask,
            zrange=zrange,
            ndiv=ndiv,
            attribute=attribute,
            maskthreshold=maskthreshold,
            snapxy=snapxy,
            showprogress=showprogress,
            deadtraces=deadtraces,
        )
    return attrs


def _slice_cube_window_v1(
    self,
    cube,
    zsurf=None,
    other=None,
    other_position="below",
    sampling="nearest",
    mask=True,
    zrange=10,
    ndiv=None,
    attribute="max",
    maskthreshold=0.1,
    snapxy=False,
    showprogress=False,
    deadtraces=True,
):

    """Slice Cube with a window and extract attribute(s)

    This is legacy version algorithm 1 and will removed later.

    The zrange is one-sided (on order to secure a centered input); hence
    of zrange is 5 than the fill window is 10.

    The maskthreshold is only valid for surfaces; if isochore is less than
    given value then the result will be masked.

    Note: attribute may be a scalar or a list. If a list, then a dict of
    surfaces are returned.
    """
    logger.info("Slice cube window method")

    qattr_is_string = True
    if not isinstance(attribute, list):
        if attribute == "all":
            attrlist = ALLATTRS
            qattr_is_string = False
        else:
            attrlist = [attribute]
    else:
        attrlist = attribute
        qattr_is_string = False

    if zsurf is not None:
        this = zsurf
    else:
        this = self.copy()

    if other is not None:
        zdelta = np.absolute(this.values - other.values)
        zrange = zdelta.max()

    ndivmode = "user setting"
    if ndiv is None:
        ndivmode = "auto"
        ndiv = int(2 * zrange / cube.zinc)
        if ndiv < 1:
            ndiv = 1
            logger.warning("NDIV < 1; reset to 1")

    logger.info("ZRANGE is %s", zrange)
    logger.info("NDIV is set to %s (%s)", ndiv, ndivmode)

    # This will run slice in a loop within a window. Then, numpy methods
    # are applied to get the attributes

    if other is None:
        attvalues = _slice_constant_window(
            this,
            cube,
            sampling,
            zrange,
            ndiv,
            mask,
            attrlist,
            snapxy,
            showprogress=showprogress,
            deadtraces=deadtraces,
        )
    else:
        attvalues = _slice_between_surfaces(
            this,
            cube,
            sampling,
            other,
            other_position,
            zrange,
            ndiv,
            mask,
            attrlist,
            maskthreshold,
            snapxy,
            showprogress=showprogress,
            deadtraces=deadtraces,
        )

    results = dict()

    for attr in attrlist:
        scopy = self.copy()
        scopy.values = attvalues[attr]
        results[attr] = scopy

    # for backward compatibility
    if qattr_is_string:
        self.values = attvalues[attrlist[0]]
        return None

    return results


def _slice_constant_window(
    this,
    cube,
    sampling,
    zrange,
    ndiv,
    mask,
    attrlist,
    snapxy,
    showprogress=False,
    deadtraces=True,
):
    """Slice a window, (constant in vertical extent)."""
    npcollect = []
    zcenter = this.copy()

    logger.info("Mean W of depth no MIDDLE slice is %s", zcenter.values.mean())
    zcenter.slice_cube(
        cube, sampling=sampling, mask=mask, snapxy=snapxy, deadtraces=deadtraces
    )
    logger.info("Mean of cube slice is %s", zcenter.values.mean())

    npcollect.append(zcenter.values)

    zincr = zrange / float(ndiv)

    logger.info("ZINCR is %s", zincr)

    # collect above the original surface
    progress = XTGShowProgress(
        ndiv * 2, show=showprogress, leadtext="progress: ", skip=1
    )
    for idv in range(ndiv):
        progress.flush(idv)
        ztmp = this.copy()
        ztmp.values -= zincr * (idv + 1)
        ztmp.slice_cube(
            cube, sampling=sampling, mask=mask, snapxy=snapxy, deadtraces=deadtraces
        )
        npcollect.append(ztmp.values)
    # collect below the original surface
    for idv in range(ndiv):
        progress.flush(ndiv + idv)
        ztmp = this.copy()
        ztmp.values += zincr * (idv + 1)
        ztmp.slice_cube(
            cube, sampling=sampling, mask=mask, snapxy=snapxy, deadtraces=deadtraces
        )
        npcollect.append(ztmp.values)

    logger.info("Make a stack of the maps...")
    stacked = ma.dstack(npcollect)
    del npcollect

    attvalues = dict()
    for attr in attrlist:
        logger.info("Running attribute %s", attr)
        attvalues[attr] = _attvalues(attr, stacked)

    progress.finished()
    return attvalues  # this is dict with numpies, one per attribute


def _slice_between_surfaces(
    this,
    cube,
    sampling,
    other,
    other_position,
    zrange,
    ndiv,
    mask,
    attrlist,
    mthreshold,
    snapxy,
    showprogress=False,
    deadtraces=True,
):

    """Slice and find values between two surfaces."""

    npcollect = []
    zincr = zrange / float(ndiv)

    zcenter = this.copy()
    zcenter.slice_cube(
        cube, sampling=sampling, mask=mask, snapxy=snapxy, deadtraces=deadtraces
    )
    npcollect.append(zcenter.values)

    # collect below or above the original surface
    if other_position == "above":
        mul = -1
    else:
        mul = 1

    # collect above the original surface
    progress = XTGShowProgress(ndiv, show=showprogress, leadtext="progress: ")
    for idv in range(ndiv):
        progress.flush(idv)
        ztmp = this.copy()
        ztmp.values += zincr * (idv + 1) * mul
        zvalues = ztmp.values.copy()

        ztmp.slice_cube(
            cube, sampling=sampling, mask=mask, snapxy=snapxy, deadtraces=deadtraces
        )

        diff = mul * (other.values - zvalues)

        values = ztmp.values
        values = ma.masked_where(diff < 0.0, values)

        npcollect.append(values)

    stacked = ma.dstack(npcollect)

    del npcollect

    # for cases with erosion, the two surfaces are equal
    isovalues = mul * (other.values - this.values)

    attvalues = dict()
    for attr in attrlist:
        attvaluestmp = _attvalues(attr, stacked)
        attvalues[attr] = ma.masked_where(isovalues < mthreshold, attvaluestmp)

    progress.finished()

    return attvalues  # this is dict with numpies, one per attribute


def _attvalues(attribute, stacked):  # pylint: disable=too-many-branches
    """Attribute values computed in numpy.ma stack."""
    if attribute == "max":
        attvalues = ma.max(stacked, axis=2)
    elif attribute == "min":
        attvalues = ma.min(stacked, axis=2)
    elif attribute == "rms":
        attvalues = np.sqrt(ma.mean(np.square(stacked), axis=2))
    elif attribute == "var":
        attvalues = ma.var(stacked, axis=2)
    elif attribute == "mean":
        attvalues = ma.mean(stacked, axis=2)
    elif attribute == "maxpos":
        stacked = ma.masked_less(stacked, 0.0, copy=True)
        attvalues = ma.max(stacked, axis=2)
    elif attribute == "maxneg":  # ~ minimum of negative values?
        stacked = ma.masked_greater_equal(stacked, 0.0, copy=True)
        attvalues = ma.min(stacked, axis=2)
    elif attribute == "maxabs":
        attvalues = ma.max(abs(stacked), axis=2)
    elif attribute == "sumpos":
        stacked = ma.masked_less(stacked, 0.0, copy=True)
        attvalues = ma.sum(stacked, axis=2)
    elif attribute == "sumneg":
        stacked = ma.masked_greater_equal(stacked, 0.0, copy=True)
        attvalues = ma.sum(stacked, axis=2)
    elif attribute == "sumabs":
        attvalues = ma.sum(abs(stacked), axis=2)
    elif attribute == "meanabs":
        attvalues = ma.mean(abs(stacked), axis=2)
    elif attribute == "meanpos":
        stacked = ma.masked_less(stacked, 0.0, copy=True)
        attvalues = ma.mean(stacked, axis=2)
    elif attribute == "meanneg":
        stacked = ma.masked_greater_equal(stacked, 0.0, copy=True)
        attvalues = ma.mean(stacked, axis=2)
    else:
        etxt = "Invalid attribute applied: {}".format(attribute)
        raise ValueError(etxt)

    if not attvalues.flags["C_CONTIGUOUS"]:
        mask = ma.getmaskarray(attvalues)
        mask = np.asanyarray(mask, order="C")
        attvalues = np.asanyarray(attvalues, order="C")
        attvalues = ma.array(attvalues, mask=mask, order="C")

    return attvalues
