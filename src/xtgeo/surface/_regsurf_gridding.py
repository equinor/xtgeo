# -*- coding: utf-8 -*-
"""Do gridding from 3D parameters"""

from __future__ import division, absolute_import
from __future__ import print_function

import warnings

import numpy as np
import numpy.ma as ma
import logging
import scipy.interpolate

import xtgeo.cxtgeo.cxtgeo as _cxtgeo
import xtgeo

xtg = xtgeo.common.XTGeoDialog()

logger = xtg.functionlogger(__name__)

# Note: 'self' is an instance of RegularSurface


def points_gridding(self, points, method='linear', coarsen=1):
    """Do gridding from a points data set."""

    xiv, yiv = self.get_xy_values()

    dfra = points.dataframe

    xcv = dfra['X_UTME'].values
    ycv = dfra['Y_UTMN'].values
    zcv = dfra['Z_TVDSS'].values

    if coarsen > 1:
        xcv = xcv[::coarsen]
        ycv = ycv[::coarsen]
        zcv = zcv[::coarsen]

    logger.info('Length of xcv array: {}'.format(xcv.size))

    validmethods = ['linear', 'nearest', 'cubic']
    if method not in set(validmethods):
        raise ValueError('Invalid method for gridding: {}, valid '
                         'options are {}'. format(method, validmethods))

    try:
        znew = scipy.interpolate.griddata((xcv, ycv), zcv, (xiv, yiv),
                                          method=method, fill_value=np.nan)
    except ValueError as verr:
        raise RuntimeError('Could not do gridding: {}'.format(verr))

    logger.info('Gridding point ... DONE')

    znew = self.ensure_correct_values(self.ncol, self.nrow, znew)

    self.values = znew


def avgsum_from_3dprops_gridding(self, summing=False, xprop=None,
                                 yprop=None, mprop=None, dzprop=None,
                                 truncate_le=None, zoneprop=None,
                                 zone_minmax=None,
                                 coarsen=1, zone_avg=False,
                                 mask_outside=False):

    # NOTE:
    # This do _either_ averaging _or_ sum gridding (if summing is True)
    # - Inputs shall be pure 3D numpies, not masked!
    # - Xprop and yprop must be made for all cells
    # - Also dzprop for all cells, and dzprop = 0 for inactive cells!

    qlog = logging.getLogger().isEnabledFor(logging.INFO)

    if zone_minmax is None:
        raise ValueError('zone_minmax is required')

    if dzprop is None:
        raise ValueError('DZ property is required')

    xprop, yprop, zoneprop, mprop, dzprop = _zone_averaging(
        xprop,
        yprop,
        zoneprop,
        zone_minmax,
        coarsen,
        zone_avg,
        dzprop,
        mprop,
        summing=summing)

    gnlay = xprop.shape[2]

    uprops = {'xprop': xprop, 'yprop': yprop, 'zoneprop': zoneprop,
              'dzprop': dzprop, 'mprop': mprop}

    # some sanity checks
    for name, ppx in uprops.items():
        minpp = ppx.min()
        maxpp = ppx.max()
        logger.info('Min Max for {} is {} .. {}'.format(name, minpp, maxpp))

    # avoid artifacts from inactive cells that slips through somehow...(?)
    if dzprop.max() > _cxtgeo.UNDEF_LIMIT:
        raise RuntimeError('Bug: DZ with unphysical values present')

    trimbydz = False
    if not summing:
        trimbydz = True

    if summing and mask_outside:
        trimbydz = True

    xiv, yiv = self.get_xy_values()

    for klay0 in range(gnlay):

        k1lay = klay0 + 1

        if k1lay == 1:
            logger.info('Initialize zsum ...')
            msum = np.zeros((self._ncol, self._nrow), order='C')
            dzsum = np.zeros((self._ncol, self._nrow), order='C')

        numz = int(round(zoneprop[::, ::, klay0].mean()))
        if numz < zone_minmax[0] or numz > zone_minmax[1]:
            continue

        qmcompute = True
        if summing:
            propsum = mprop[:, :, klay0].sum()
            if (abs(propsum) < 1e-12):
                logger.info('Too little HC, skip layer K = {}'.format(k1lay))
                qmcompute = False
            else:
                logger.debug('Z property sum is {}'.format(propsum))

        logger.info('Mapping for layer or zone ' + str(k1lay) + '...')

        xcv = xprop[::, ::, klay0].ravel(order='C')
        ycv = yprop[::, ::, klay0].ravel(order='C')
        mv = mprop[::, ::, klay0].ravel(order='C')
        dz = dzprop[::, ::, klay0].ravel(order='C')

        # this is done to avoid problems if undef values still remains
        # in the coordinates (assume Y undef where X undef):
        xcc = xcv.copy()
        xcv = xcv[xcc < 1e20]
        ycv = ycv[xcc < 1e20]
        mv = mv[xcc < 1e20]
        dz = dz[xcc < 1e20]

        # some sanity checks
        if qlog:
            uprops = {'xcv': xcv, 'ycv': ycv, 'mv': mv, 'dz': dz}
            for name, ppx in uprops.items():
                minpp = ppx.min()
                maxpp = ppx.max()
                logger.info('Min Max {} is {} - {}'.format(name, minpp, maxpp))

        if summing:
            mvdz = mv
        else:
            mvdz = mv * dz

        if qmcompute:
            try:
                mvdzi = scipy.interpolate.griddata((xcv, ycv),
                                                   mvdz,
                                                   (xiv, yiv),
                                                   method='linear',
                                                   fill_value=0.0)
            except ValueError:
                warnings.warn('Some problems in gridding ... will contue',
                              UserWarning)
                continue

            logger.debug(mvdzi.shape)

            msum = msum + mvdzi

        if trimbydz:
            try:
                dzi = scipy.interpolate.griddata((xcv, ycv),
                                                 dz,
                                                 (xiv, yiv),
                                                 method='linear',
                                                 fill_value=0.0)
            except ValueError:
                continue

            dzsum = dzsum + dzi

    if not summing:
        dzsum[dzsum == 0.0] = 1e-20
        vv = msum / dzsum
        vv = ma.masked_invalid(vv)
    else:
        vv = msum

    if trimbydz:
        vv = ma.masked_where(dzsum < 1.1e-20, vv)
    else:
        vv = ma.array(vv)  # so the result becomes a ma array

    if truncate_le:
        vv = ma.masked_less(vv, truncate_le)

    self.values = vv

    return True


def _zone_averaging(xprop, yprop, zoneprop, zone_minmax, coarsen,
                    zone_avg, dzprop, mprop, summing=False):

    # General preprocessing, and...
    # Change the 3D numpy array so they get layers by
    # averaging across zones. This may speed up a lot,
    # but will reduce the resolution.
    # The x y coordinates shall be averaged (ideally
    # with thickness weighting...) while e.g. hcpfzprop
    # must be summed.
    # Somewhat different processing whether this is a hc thickness
    # or an average.

    uprops = {'xprop': xprop, 'yprop': yprop, 'zoneprop': zoneprop,
              'dzprop': dzprop, 'mprop': mprop}

    # some sanity checks
    for name, ppx in uprops.items():
        minpp = ppx.min()
        maxpp = ppx.max()
        logger.info('Min Max for {} is {} .. {}'.format(name, minpp, maxpp))

    xpr = xprop
    ypr = yprop
    zpr = zoneprop
    dpr = dzprop

    mpr = mprop

    if coarsen > 1:
        xpr = xprop[::coarsen, ::coarsen, ::].copy(order='C')
        ypr = yprop[::coarsen, ::coarsen, ::].copy(order='C')
        zpr = zoneprop[::coarsen, ::coarsen, ::].copy(order='C')
        dpr = dzprop[::coarsen, ::coarsen, ::].copy(order='C')
        mpr = mprop[::coarsen, ::coarsen, ::].copy(order='C')
        zpr.astype(np.int32)

        logger.info('Coarsen is {}'.format(coarsen))

    if zone_avg:
        logger.info('Tuning zone_avg is {}'.format(zone_avg))
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

        for iz in range(zmin, zmax + 1):
            logger.info('Averaging for zone {} ...'.format(iz))
            xpr2 = ma.masked_where(zpr != iz, xpr)
            ypr2 = ma.masked_where(zpr != iz, ypr)
            zpr2 = ma.masked_where(zpr != iz, zpr)
            dpr2 = ma.masked_where(zpr != iz, dpr)
            mpr2 = ma.masked_where(zpr != iz, mpr)

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

    xpr = ma.filled(xpr, fill_value=_cxtgeo.UNDEF)
    ypr = ma.filled(ypr, fill_value=_cxtgeo.UNDEF)
    zpr = ma.filled(zpr, fill_value=0)
    dpr = ma.filled(dpr, fill_value=0.0)

    mpr = ma.filled(mpr, fill_value=0.0)

    return xpr, ypr, zpr, mpr, dpr
