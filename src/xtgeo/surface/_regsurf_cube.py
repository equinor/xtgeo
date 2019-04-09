# -*- coding: utf-8 -*-
"""Regular surface vs Cube"""
from __future__ import division, absolute_import
from __future__ import print_function

import numpy as np
import numpy.ma as ma

import xtgeo.cxtgeo.cxtgeo as _cxtgeo
from xtgeo.common import XTGeoDialog
from xtgeo.common import XTGShowProgress

xtg = XTGeoDialog()

logger = xtg.functionlogger(__name__)
_cxtgeo.xtg_verbose_file("NONE")

XTGDEBUG = xtg.get_syslevel()

ALLATTRS = [
    "max",
    "min",
    "rms",
    "var",
    "mean",
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

# self = RegularSurface instance!
# pylint: disable=too-many-locals, too-many-branches


def slice_cube(
    self, cube, zsurf=None, sampling="nearest", mask=True, snapxy=False, deadtraces=True
):
    """Private function for the Cube slicing."""

    if zsurf is not None:
        other = zsurf
    else:
        logger.info('The current surface is copied as "other"')
        other = self.copy()

    if not self.compare_topology(other, strict=False):
        raise RuntimeError("Topology of maps differ. Stop!")

    if mask:
        opt2 = 0
    else:
        opt2 = 1

    if deadtraces:
        # set dead traces to cxtgeo UNDEF -> special treatment in the C code
        olddead = cube.values_dead_traces(_cxtgeo.UNDEF)

    cubeval1d = np.ravel(cube.values, order="C")

    nsurf = self.ncol * self.nrow

    usesampling = 0
    if sampling == "trilinear":
        usesampling = 1
        if snapxy:
            usesampling = 2

    logger.debug("Running method from C... (using typemaps for numpies!:")
    istat, v1d = _cxtgeo.surf_slice_cube(
        cube.ncol,
        cube.nrow,
        cube.nlay,
        cube.xori,
        cube.xinc,
        cube.yori,
        cube.yinc,
        cube.zori,
        cube.zinc,
        cube.rotation,
        cube.yflip,
        cubeval1d,
        self.ncol,
        self.nrow,
        self.xori,
        self.xinc,
        self.yori,
        self.yinc,
        self.yflip,
        self.rotation,
        other.get_values1d(),
        nsurf,
        usesampling,
        opt2,
        XTGDEBUG,
    )

    if istat != 0:
        logger.warning("Problem, ISTAT = %s", istat)

    self.set_values1d(v1d)

    if deadtraces:
        cube.values_dead_traces(olddead)  # reset value for dead traces

    return istat


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
    deletecube=False,
    algorithm=1,
):

    """Slice Cube with a window and extract attribute(s)

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

    if other is None and algorithm == 1:
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
            deletecube=deletecube,
        )
    elif other is None and algorithm == 2:
        attvalues = _slice_constant_window2(
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
            deletecube=deletecube,
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
            deletecube=deletecube,
        )

    results = dict()

    if algorithm != 2:
        for attr in attrlist:
            scopy = self.copy()
            scopy.values = attvalues[attr]
            results[attr] = scopy

        # for backward compatibility
        if qattr_is_string:
            self.values = attvalues[attrlist[0]]
            return None

        return results

    return None


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
    deletecube=False,
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
    if deletecube:
        del cube

    attvalues = dict()
    for attr in attrlist:
        logger.info("Running attribute %s", attr)
        attvalues[attr] = _attvalues(attr, stacked)

    progress.finished()
    return attvalues  # this is dict with numpies, one per attribute


# NOT FINISHED:
def _slice_constant_window2(  # pylint: disable=unused-argument
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
    deletecube=False,
):
    """Slice a window, (constant in vertical extent); faster and better
    algorithm (algorithm2).
    """

    zincr = zrange / float(ndiv)
    ztmp = this.copy()
    ztmp.values = ztmp.values - zincr * (ndiv + 1)

    if mask:
        opt2 = 0
    else:
        opt2 = 1

    if deadtraces:
        # set dead traces to cxtgeo UNDEF -> special treatment in the C code
        olddead = cube.values_dead_traces(_cxtgeo.UNDEF)

    cubeval1d = np.ravel(cube.values, order="C")

    usesampling = 0
    if sampling == "trilinear":
        usesampling = 1
        if snapxy:
            usesampling = 2

    # allocate the attribute maps
    nattr = 5  # predefined attributes
    nsurf = ztmp.ncol * ztmp.nrow * nattr

    istat, _attrmaps = _cxtgeo.surf_slice_cube_window(
        cube.ncol,
        cube.nrow,
        cube.nlay,
        cube.xori,
        cube.xinc,
        cube.yori,
        cube.yinc,
        cube.zori,
        cube.zinc,
        cube.rotation,
        cube.yflip,
        cubeval1d,
        ztmp.ncol,
        ztmp.nrow,
        ztmp.xori,
        ztmp.xinc,
        ztmp.yori,
        ztmp.yinc,
        ztmp.yflip,
        ztmp.rotation,
        ztmp.get_values1d(),
        zincr,
        ndiv * 2,
        nsurf,
        nattr,
        usesampling,
        opt2,
        XTGDEBUG,
    )

    if istat != 0:
        logger.warning("Problem, ISTAT = %s", istat)

    if deadtraces:
        cube.values_dead_traces(olddead)  # reset value for dead traces

    if deletecube:
        del cube

    return istat


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
    deletecube=False,
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

    if deletecube:
        del cube

    # for cases with erosion, the two surfaces are equal
    isovalues = mul * (other.values - this.values)

    attvalues = dict()
    for attr in attrlist:
        attvaluestmp = _attvalues(attr, stacked)
        attvalues[attr] = ma.masked_where(isovalues < mthreshold, attvaluestmp)

    progress.finished()

    return attvalues  # this is dict with numpies, one per attribute


def _attvalues(attribute, stacked):
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
