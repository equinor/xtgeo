# -*- coding: utf-8 -*-
"""Regular surface vs Cube, slice a window interval v2"""
from __future__ import division, absolute_import
from __future__ import print_function

import numpy as np
import xtgeo
import xtgeo.cxtgeo._cxtgeo as _cxtgeo
from xtgeo.common import XTGeoDialog

xtg = XTGeoDialog()

logger = xtg.functionlogger(__name__)


ALLATTRS = [
    "min",
    "max",
    "mean",
    "var",
    "rms",
    "maxpos",
    "maxneg",
    "maxabs",
    "meanabs",
    "meanpos",
    "meanneg",
    "sumpos",
    "sumneg",
    "sumabs",
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
):

    if not snapxy:
        attrs = _slice_cube_window_resample(
            self,
            cube,
            zsurf,
            other,
            other_position,
            sampling,
            mask,
            zrange,
            ndiv,
            attribute,
            maskthreshold,
            showprogress,
            deadtraces,
        )

    else:

        attrs = _slice_cube_window(
            self,
            cube,
            zsurf,
            other,
            other_position,
            sampling,
            mask,
            zrange,
            ndiv,
            attribute,
            maskthreshold,
            showprogress,
            deadtraces,
        )

    if isinstance(attrs, dict) and len(attrs) == 1:

        if isinstance(attribute, list):
            myattr = attribute[0]
        else:
            myattr = attribute

        self.values = attrs[myattr].values
        return None

    return attrs


def _slice_cube_window(
    self,
    cube,
    zsurf,
    other,
    other_position,
    sampling,
    mask,
    zrange,
    ndiv,
    attribute,
    maskthreshold,
    showprogress,
    deadtraces,
):  # pylint: disable=too-many-branches, too-many-statements

    """Slice Cube between surfaces to find attributes

    New from May 2020, to provide a much faster algorithm and correct some issues
    found in previous version
    """

    logger.info("Slice cube window method v2")

    if deadtraces:
        olddead = cube.values_dead_traces(xtgeo.UNDEF)

    optprogress = 0
    if showprogress:
        optprogress = 1

    if not isinstance(attribute, list):
        if attribute == "all":
            attrlist = ALLATTRS
        else:
            attrlist = [attribute]
    else:
        attrlist = attribute

    if zsurf is not None:
        this = zsurf
    else:
        this = self.copy()

    if other is not None:
        zdelta = np.absolute(this.values - other.values)
        zdiameter = zdelta.max()
        if other_position.lower() == "below":
            surf1 = this
            surf2 = other
        else:
            surf1 = other  # BEWARE self
            surf2 = this
    else:
        surf1 = this.copy()
        surf2 = this.copy()
        surf1.values -= zrange
        surf2.values += zrange
        zdiameter = 2 * zrange

    if ndiv is None:
        ndiv = int(2 * zdiameter / cube.zinc)
        if ndiv < 1:
            ndiv = 1
            logger.warning("NDIV < 1; reset to 1")

    # force/overrule a coarse sampling for sampling "cube"
    ndivdisc = int((zdiameter) * 1.0001 / cube.zinc)
    if sampling == "cube":
        ndiv = ndivdisc

    zrinc = zdiameter / ndiv
    logger.debug("zdiameter and cube zinc: %s %s", zdiameter, cube.zinc)
    logger.debug("zrinc and ndiv: %s %s", zrinc, ndiv)

    optsum = 0
    if any("sum" in word for word in attrlist):
        optsum = 1

    results = _attributes_betw_surfaces(
        self,
        cube,
        surf1,
        surf2,
        sampling,
        mask,
        zrinc,
        ndiv,
        ndivdisc,
        optprogress,
        maskthreshold,
        optsum,
    )

    if deadtraces:
        cube.values_dead_traces(olddead)  # reset value for dead traces

    # build the returning result
    alist = {}
    for att in attrlist:
        num = ALLATTRS.index(att)
        alist[att] = self.copy()
        alist[att].values = results[num, :]

    return alist


def _attributes_betw_surfaces(
    self,
    cube,
    surf1,
    surf2,
    sampling,
    maskopt,
    zrinc,
    ndiv,
    ndivdisc,
    optprogress,
    maskthreshold,
    optsum,
):
    """This is the actual lowlevel engine communicating with C code"""

    logger.info("Attributes between surfaces")

    results = np.zeros((len(ALLATTRS) * self.ncol * self.nrow), dtype=np.float64)

    optnearest = 0
    if sampling in ["nearest", "cube"]:
        optnearest = 1

    _cxtgeo.surf_cube_attr_intv(
        cube.ncol,
        cube.nrow,
        cube.nlay,
        cube.zori,
        cube.zinc,
        cube.values,
        surf1.values.data,
        surf2.values.data,
        surf1.values.mask,
        surf2.values.mask,
        zrinc,
        ndiv,
        ndivdisc,
        results,
        optnearest,
        maskopt,
        optprogress,
        maskthreshold,
        optsum,
    )

    logger.info("Results updated, with size %s", results.shape)

    results = results.reshape((len(ALLATTRS), self.ncol * self.nrow), order="C")

    return results


def _slice_cube_window_resample(
    self,
    cube,
    zsurf,
    other,
    other_position,
    sampling,
    mask,
    zrange,
    ndiv,
    attribute,
    maskthreshold,
    showprogress,
    deadtraces,
):
    """Makes a resample from origonal surfaces first to fit cube topology"""

    logger.info("Attributes between surfaces, resampling version")

    scube = xtgeo.surface_from_cube(cube, 0.0)

    scube.resample(self)

    szsurf = None
    if zsurf:
        szsurf = scube.copy()
        szsurf.resample(zsurf)

    sother = None
    if other:
        sother = scube.copy()
        sother.resample(other)

    attrs = _slice_cube_window(
        scube,
        cube,
        szsurf,
        sother,
        other_position,
        sampling,
        mask,
        zrange,
        ndiv,
        attribute,
        maskthreshold,
        showprogress,
        deadtraces,
    )

    # now resample attrs back to a copy of self
    zelf = self.copy()
    for key, _val in attrs.items():
        zelf.resample(attrs[key])
        attrs[key] = zelf.copy()

    return attrs
