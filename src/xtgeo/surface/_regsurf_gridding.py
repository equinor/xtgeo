"""Do gridding from 3D parameters"""

import warnings

import numpy as np
import numpy.ma as ma
import scipy.interpolate

import cxtgeo.cxtgeo as _cxtgeo
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
            msum = np.zeros((self._ncol, self._nrow), order='F')
            dzsum = np.zeros((self._ncol, self._nrow), order='F')

        numz = int(round(zoneprop[::, ::, klay0].mean()))
        if numz < zone_minmax[0] or numz > zone_minmax[1]:
            continue

        if summing:
            propsum = mprop[:, :, klay0].sum()
            if (abs(propsum) < 1e-12):
                logger.info('Too little HC, skip layer K = {}'.format(k1lay))
                continue
            else:
                logger.debug('Z property sum is {}'.format(propsum))

        logger.info('Mapping for layer or zone ' + str(k1lay) + '...')

        xcv = xprop[::, ::, klay0].ravel(order='F')
        ycv = yprop[::, ::, klay0].ravel(order='F')
        mv = mprop[::, ::, klay0].ravel(order='F')
        dz = dzprop[::, ::, klay0].ravel(order='F')

        # this is done to avoid problems if undef values still remains
        # in the coordinates:
        xcc = xcv.copy()
        xcv = xcv[xcc < _cxtgeo.UNDEF_LIMIT]
        ycv = ycv[xcc < _cxtgeo.UNDEF_LIMIT]
        mv = mv[xcc < _cxtgeo.UNDEF_LIMIT]
        dz = dz[xcc < _cxtgeo.UNDEF_LIMIT]

        if summing:
            mvdz = mv
        else:
            mvdz = mv * dz

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

        if not summing or trimbydz:
            try:
                dzi = scipy.interpolate.griddata((xcv, ycv),
                                                 dz,
                                                 (xiv, yiv),
                                                 method='linear',
                                                 fill_value=0.0)
            except ValueError:
                continue

            dzi = np.asfortranarray(dzi)
            dzsum = dzsum + dzi

        logger.debug(mvdzi.shape)
        mvdzi = np.asfortranarray(mvdzi)

        msum = msum + mvdzi

    if not summing:
        dzsum[dzsum == 0.0] = 1e-20
        vv = msum / dzsum
        vv = ma.masked_invalid(vv)
    else:
        vv = msum

    if trimbydz:
        vv = ma.masked_where(dzsum < 1e-20, vv)
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

    if summing:
        hpr = mprop
    else:
        mpr = mprop

    if coarsen > 1:
        xpr = xprop[::coarsen, ::coarsen, ::].copy(order='F')
        ypr = yprop[::coarsen, ::coarsen, ::].copy(order='F')
        zpr = zoneprop[::coarsen, ::coarsen, ::].copy(order='F')
        dpr = dzprop[::coarsen, ::coarsen, ::].copy(order='F')
        if summing:
            hpr = mprop[::coarsen, ::coarsen, ::].copy(order='F')
        else:
            mpr = mprop[::coarsen, ::coarsen, ::].copy(order='F')

        logger.info('Coarsen is {}'.format(coarsen))

    if zone_avg:
        logger.info('Tuning zone_avg is {}'.format(zone_avg))
        zmin = int(zone_minmax[0])
        zmax = int(zone_minmax[1])
        newx = []
        newy = []
        newz = []
        newh = []
        newm = []
        newd = []

        for iz in range(zmin, zmax + 1):
            xpr2 = ma.masked_where(zpr != iz, xpr)
            ypr2 = ma.masked_where(zpr != iz, ypr)
            zpr2 = ma.masked_where(zpr != iz, zpr)
            dpr2 = ma.masked_where(zpr != iz, dpr)
            if summing:
                hpr2 = ma.masked_where(zpr != iz, hpr)
            else:
                mpr2 = ma.masked_where(zpr != iz, mpr)

            # get the thickness and normalize along axis 2 (vertical)
            # to get normalized thickness weights
            lay_sums = dpr2.sum(axis=2)
            normed_dz = dpr2 / lay_sums[:, :, np.newaxis]

            xpr2 = ma.average(xpr2, weights=normed_dz, axis=2)
            ypr2 = ma.average(ypr2, weights=normed_dz, axis=2)
            zpr2 = ma.average(zpr2, axis=2)  # no need for weights
            if summing:
                hpr2 = ma.sum(hpr2, axis=2)
            else:
                mpr2 = ma.average(mpr2, weights=normed_dz, axis=2)

            dpr2 = ma.sum(dpr2, axis=2)
            newx.append(xpr2)
            newy.append(ypr2)
            newz.append(zpr2)
            newd.append(dpr2)
            if summing:
                newh.append(hpr2)
            else:
                newm.append(mpr2)

        xpr = ma.dstack(newx)
        ypr = ma.dstack(newy)
        zpr = ma.dstack(newz)
        dpr = ma.dstack(newd)
        if summing:
            hpr = ma.dstack(newh)
        else:
            mpr = ma.dstack(newm)

    xpr = ma.filled(xpr, fill_value=_cxtgeo.UNDEF)
    ypr = ma.filled(ypr, fill_value=_cxtgeo.UNDEF)
    zpr = ma.filled(zpr, fill_value=0)
    dpr = ma.filled(dpr, fill_value=0.0)

    for a in [xpr, ypr, zpr]:
        logger.info('Reduced shape of ... is {}'.format(a.shape))

    if summing:
        hpr = ma.filled(hpr, fill_value=0.0)
        return xpr, ypr, zpr, hpr, dpr
    else:
        mpr = ma.filled(mpr, fill_value=0.0)
        return xpr, ypr, zpr, mpr, dpr
