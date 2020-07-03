# -*- coding: utf-8 -*-
"""Do gridding from 3D parameters"""

from __future__ import division, absolute_import
from __future__ import print_function

import warnings

import numpy as np
import numpy.ma as ma
import scipy.interpolate
import scipy.ndimage

import xtgeo

xtg = xtgeo.common.XTGeoDialog()

logger = xtg.functionlogger(__name__)

# Note: 'self' is an instance of RegularSurface
# pylint: disable=too-many-branches, too-many-statements, too-many-locals


def points_gridding(self, points, method="linear", coarsen=1):
    """Do gridding from a points data set."""

    xiv, yiv = self.get_xy_values()

    dfra = points.dataframe

    xcv = dfra[points.xname].values
    ycv = dfra[points.yname].values
    zcv = dfra[points.zname].values

    if coarsen > 1:
        xcv = xcv[::coarsen]
        ycv = ycv[::coarsen]
        zcv = zcv[::coarsen]

    validmethods = ["linear", "nearest", "cubic"]
    if method not in set(validmethods):
        raise ValueError(
            "Invalid method for gridding: {}, valid "
            "options are {}".format(method, validmethods)
        )

    try:
        znew = scipy.interpolate.griddata(
            (xcv, ycv), zcv, (xiv, yiv), method=method, fill_value=np.nan
        )
    except ValueError as verr:
        raise RuntimeError("Could not do gridding: {}".format(verr))

    logger.info("Gridding point ... DONE")

    self._ensure_correct_values(znew)


def avgsum_from_3dprops_gridding(
    self,
    summing=False,
    xprop=None,
    yprop=None,
    mprop=None,
    dzprop=None,
    truncate_le=None,
    zoneprop=None,
    zone_minmax=None,
    coarsen=1,
    zone_avg=False,
    mask_outside=False,
):

    """Get surface average from a 3D grid prop."""
    # NOTE:
    # This do _either_ averaging _or_ sum gridding (if summing is True)
    # - Inputs shall be pure 3D numpies, not masked!
    # - Xprop and yprop must be made for all cells
    # - Also dzprop for all cells, and dzprop = 0 for inactive cells!

    logger.info("Avgsum calculation %s", __name__)

    if zone_minmax is None:
        raise ValueError("zone_minmax is required")

    if dzprop is None:
        raise ValueError("DZ property is required")

    xprop, yprop, zoneprop, mprop, dzprop = _zone_averaging(
        xprop,
        yprop,
        zoneprop,
        zone_minmax,
        coarsen,
        zone_avg,
        dzprop,
        mprop,
        summing=summing,
    )

    gnlay = xprop.shape[2]

    # avoid artifacts from inactive cells that slips through somehow...(?)
    if dzprop.max() > xtgeo.UNDEF_LIMIT:
        raise RuntimeError("Bug: DZ with unphysical values present")

    trimbydz = False
    if not summing:
        trimbydz = True

    if summing and mask_outside:
        trimbydz = True

    xiv, yiv = self.get_xy_values()

    # weight are needed if zoneprop is not follow layers, but rather regions
    weights = dzprop.copy() * 0.0 + 1.0
    weights[zoneprop < zone_minmax[0]] = 0.0
    weights[zoneprop > zone_minmax[1]] = 0.0

    # this operation is needed if zoneprop is aka a region ("irregular zone")
    zoneprop = ma.masked_less(zoneprop, zone_minmax[0])
    zoneprop = ma.masked_greater(zoneprop, zone_minmax[1])

    for klay0 in range(gnlay):

        k1lay = klay0 + 1

        if k1lay == 1:
            msum = np.zeros((self.ncol, self.nrow), order="C")
            dzsum = np.zeros((self.ncol, self.nrow), order="C")

        numz = zoneprop[::, ::, klay0].mean()
        if isinstance(numz, float):
            numz = int(round(zoneprop[::, ::, klay0].mean()))
            if numz < zone_minmax[0] or numz > zone_minmax[1]:
                continue
        else:
            continue

        qmcompute = True
        if summing:
            propsum = mprop[:, :, klay0].sum()
            if abs(propsum) < 1e-12:
                logger.info("Too little HC, skip layer K = %s", k1lay)
                qmcompute = False
            else:
                logger.debug("Z property sum is %s", propsum)

        logger.info("Mapping for layer or zone %s ....", k1lay)

        xcv = xprop[::, ::, klay0].ravel(order="C")
        ycv = yprop[::, ::, klay0].ravel(order="C")
        mvv = mprop[::, ::, klay0].ravel(order="C")
        dzv = dzprop[::, ::, klay0].ravel(order="C")
        wei = weights[::, ::, klay0].ravel(order="C")

        # this is done to avoid problems if undef values still remains
        # in the coordinates (assume Y undef where X undef):
        xcc = xcv.copy()
        xcv = xcv[xcc < 1e20]
        ycv = ycv[xcc < 1e20]
        mvv = mvv[xcc < 1e20]
        dzv = dzv[xcc < 1e20]
        wei = wei[xcc < 1e20]

        if summing:
            mvdz = mvv * wei
        else:
            mvdz = mvv * dzv * wei

        if qmcompute:
            try:
                mvdzi = scipy.interpolate.griddata(
                    (xcv, ycv), mvdz, (xiv, yiv), method="linear", fill_value=0.0
                )
            except ValueError:
                warnings.warn("Some problems in gridding ... will contue", UserWarning)
                continue

            msum = msum + mvdzi

        if trimbydz:
            try:
                dzi = scipy.interpolate.griddata(
                    (xcv, ycv), dzv, (xiv, yiv), method="linear", fill_value=0.0
                )
            except ValueError:
                continue

            dzsum = dzsum + dzi

    if not summing:
        dzsum[dzsum == 0.0] = 1e-20
        vvz = msum / dzsum
        vvz = ma.masked_invalid(vvz)
    else:
        vvz = msum

    if trimbydz:
        vvz = ma.masked_where(dzsum < 1.1e-20, vvz)
    else:
        vvz = ma.array(vvz)  # so the result becomes a ma array

    if truncate_le:
        vvz = ma.masked_less(vvz, truncate_le)

    self.values = vvz
    logger.info("Avgsum calculation done! %s", __name__)

    return True


def _zone_averaging(
    xprop, yprop, zoneprop, zone_minmax, coarsen, zone_avg, dzprop, mprop, summing=False
):

    # General preprocessing, and...
    # Change the 3D numpy array so they get layers by
    # averaging across zones. This may speed up a lot,
    # but will reduce the resolution.
    # The x y coordinates shall be averaged (ideally
    # with thickness weighting...) while e.g. hcpfzprop
    # must be summed.
    # Somewhat different processing whether this is a hc thickness
    # or an average.

    xpr = xprop
    ypr = yprop
    zpr = zoneprop
    dpr = dzprop

    mpr = mprop

    if coarsen > 1:
        xpr = xprop[::coarsen, ::coarsen, ::].copy(order="C")
        ypr = yprop[::coarsen, ::coarsen, ::].copy(order="C")
        zpr = zoneprop[::coarsen, ::coarsen, ::].copy(order="C")
        dpr = dzprop[::coarsen, ::coarsen, ::].copy(order="C")
        mpr = mprop[::coarsen, ::coarsen, ::].copy(order="C")
        zpr.astype(np.int32)

    if zone_avg:
        zmin = int(zone_minmax[0])
        zmax = int(zone_minmax[1])
        if zpr.min() > zmin:
            zmin = zpr.min()
        if zpr.max() < zmax:
            zmax = zpr.max()

        newx = []
        newy = []
        newz = []
        newm = []
        newd = []

        for izv in range(zmin, zmax + 1):
            logger.info("Averaging for zone %s ...", izv)
            xpr2 = ma.masked_where(zpr != izv, xpr)
            ypr2 = ma.masked_where(zpr != izv, ypr)
            zpr2 = ma.masked_where(zpr != izv, zpr)
            dpr2 = ma.masked_where(zpr != izv, dpr)
            mpr2 = ma.masked_where(zpr != izv, mpr)

            # get the thickness and normalize along axis 2 (vertical)
            # to get normalized thickness weights
            lay_sums = dpr2.sum(axis=2)
            normed_dz = dpr2 / lay_sums[:, :, np.newaxis]

            # assume that coordinates have equal weights within a zone
            xpr2 = ma.average(xpr2, axis=2)
            ypr2 = ma.average(ypr2, axis=2)
            zpr2 = ma.average(zpr2, axis=2)  # avg zone

            dpr2 = ma.sum(dpr2, axis=2)

            if summing:
                mpr2 = ma.sum(mpr2, axis=2)
            else:
                mpr2 = ma.average(mpr2, weights=normed_dz, axis=2)  # avg zone

            newx.append(xpr2)
            newy.append(ypr2)
            newz.append(zpr2)
            newd.append(dpr2)
            newm.append(mpr2)

        xpr = ma.dstack(newx)
        ypr = ma.dstack(newy)
        zpr = ma.dstack(newz)
        dpr = ma.dstack(newd)
        mpr = ma.dstack(newm)
        zpr.astype(np.int32)

    xpr = ma.filled(xpr, fill_value=xtgeo.UNDEF)
    ypr = ma.filled(ypr, fill_value=xtgeo.UNDEF)
    zpr = ma.filled(zpr, fill_value=0)
    dpr = ma.filled(dpr, fill_value=0.0)

    mpr = ma.filled(mpr, fill_value=0.0)

    return xpr, ypr, zpr, mpr, dpr


def surf_fill(self, fill_value=None):
    """Replace the value of invalid 'data' cells (indicated by 'invalid')
    by the value of the nearest valid data cell or a constant.

    This is a quite fast method to fill undefined areas of the map.
    The surface values are updated 'in-place'

    .. versionadded:: 2.1.0
    .. versionchanged:: 2.6.0 Added fill_value
    """
    logger.info("Do fill...")

    if fill_value is not None:
        if np.isscalar(fill_value) and not isinstance(fill_value, str):
            self.values = ma.filled(self.values, fill_value=float(fill_value))
        else:
            raise ValueError("Keyword fill_value must be int or float")
    else:

        invalid = ma.getmaskarray(self.values)

        ind = scipy.ndimage.distance_transform_edt(
            invalid, return_distances=False, return_indices=True
        )
        self._values = self._values[tuple(ind)]
        logger.info("Do fill... DONE")


def smooth_median(self, iterations=1, width=1):
    """Smooth a surface using a median filter.

    .. versionadded:: 2.1.0
    """

    mask = ma.getmaskarray(self.values)
    tmpv = ma.filled(self.values, fill_value=np.nan)

    for _itr in range(iterations):
        tmpv = scipy.ndimage.median_filter(tmpv, width)

    tmpv = ma.masked_invalid(tmpv)

    # seems that false areas of invalids (masked) may be made; combat that:
    self.values = tmpv
    self.fill()
    self.values = ma.array(self.values, mask=mask)
