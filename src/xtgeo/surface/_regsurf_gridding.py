"""Do gridding from 3D parameters"""

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

    xi, yi = self.get_xy_values()

    df = points.dataframe

    xc = df['X_UTME'].values
    yc = df['Y_UTMN'].values
    zc = df['Z_TVDSS'].values

    if coarsen > 1:
        xc = xc[::coarsen]
        yc = yc[::coarsen]
        zc = zc[::coarsen]

    logger.info('Length of xc array: {}'.format(xc.size))

    validmethods = ['linear', 'nearest', 'cubic']
    if method not in set(validmethods):
        raise ValueError('Invalid method for gridding: {}, valid '
                         'options are {}'. format(method, validmethods))

    try:
        znew = scipy.interpolate.griddata((xc, yc), zc, (xi, yi),
                                          method=method, fill_value=np.nan)
    except ValueError as verr:
        raise RuntimeError('Could not do gridding: {}'.format(verr))

    logger.info('Gridding point ... DONE')

    znew = self.ensure_correct_values(self.ncol, self.nrow, znew)

    self.values = znew


def avgsum_from_3d_prop_gridding(self, summing=False, xprop=None,
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

    xprop, yprop, zoneprop, mprop, dzprop = _zone_averaging2(
        xprop,
        yprop,
        zoneprop,
        zone_minmax,
        coarsen,
        zone_avg,
        dzprop,
        mprop,
        summing=summing)

    gnlay = xprop.nlay

    # avoid artifacts from inactive cells that slips through somehow...(?)
    if dzprop.max() > _cxtgeo.UNDEF_LIMIT:
        raise RuntimeError('Bug: DZ with unphysical values present')

    xi, yi = self.get_xy_values()

    for k0 in range(gnlay):

        k1 = k0 + 1

        if k1 == 1:
            logger.info('Initialize zsum ...')
            zsum = np.zeros((self._ncol, self._nrow), order='F')
            dzsum = np.zeros((self._ncol, self._nrow), order='F')

        numz = int(round(zoneprop[::, ::, k0].mean()))
        if numz < zone_minmax[0] or numz > zone_minmax[1]:
            continue

        if summing:
            propsum = mprop[:, :, k0].sum()
            if (abs(propsum) < 1e-12):
                logger.info('Too little HC, skip layer K = {}'.format(k1))
                continue
            else:
                logger.debug('Z property sum is {}'.format(propsum))

        logger.info('Mapping for layer or zone ' + str(k1) + '...')

        xc = xprop[::, ::, k0].ravel(order='F')
        yc = yprop[::, ::, k0].ravel(order='F')
        mv = mprop[::, ::, k0].ravel(order='F')
        dz = dzprop[::, ::, k0].ravel(order='F')

        # this is done to avoid problems if undef values still remains
        # in the coordinates:
        xcc = xc.copy()
        xc = xc[xcc < _cxtgeo.UNDEF_LIMIT]
        yc = yc[xcc < _cxtgeo.UNDEF_LIMIT]
        mv = mv[xcc < _cxtgeo.UNDEF_LIMIT]
        dz = dz[xcc < _cxtgeo.UNDEF_LIMIT]

        if summing:
            mvdz = mv
        else:
            mvdz = mv * dz

        try:
            mvdzi = scipy.interpolate.griddata((xc, yc),
                                               mvdz,
                                               (xi, yi),
                                               method='linear',
                                               fill_value=0.0)
        except ValueError:
            continue

        if (not summing) or (summing and mask_outside):
            try:
                dzi = scipy.interpolate.griddata((xc, yc),
                                                 dz,
                                                 (xi, yi),
                                                 method='linear',
                                                 fill_value=0.0)
            except ValueError:
                continue

            dzi = np.asfortranarray(dzi)
            dzsum = dzsum + dzi

        logger.debug(mvdzi.shape)
        mvdzi = np.asfortranarray(mvdzi)

        wsum = wsum + mvdzi

    if summing and mask_outside:
        zsum = ma.masked_where(dzsum < 1e-20, zsum)
    elif summing:
        zsum = ma.array(zsum)  # so the result becomes a ma array

    dzmask = dzsum.copy()

    dzsum[dzsum == 0.0] = 1e-20

    if not summing:
        vv = wsum / dzsum
        vv = ma.masked_invalid(vv)
    else:
        vv = wsum

    # apply the mask from the DZ map to mask the result
    dzmask = ma.masked_less(dzmask, 1e-20)
    dzmask = ma.getmaskarray(dzmask)

    if truncate_le:
        vv = ma.masked_less(vv, truncate_le)

    vv.mask = dzmask
    self.values = vv

    return True


def avg_from_3d_prop_gridding(self, xprop=None, yprop=None,
                              mprop=None, dzprop=None,
                              truncate_le=None, zoneprop=None,
                              zone_minmax=None,
                              coarsen=1, zone_avg=False):

    # NOTE:
    # - Inputs shall be pure 3D numpies, not masked!
    # - Xprop and yprop must be made for all cells
    # - Also dzprop for all cells, and dzprop = 0 for inactive cells!

    if zone_minmax is None:
        raise ValueError('zone_minmax is required')

    if dzprop is None:
        raise ValueError('DZ property is required')

    # preprocessing
    xprop, yprop, zoneprop, mprop, dzprop = _zone_averaging(
        xprop,
        yprop,
        zoneprop,
        zone_minmax,
        coarsen,
        zone_avg,
        dzprop,
        mprop=mprop,
        hcprop=None)

    ncol, nrow, nlay = xprop.shape

    # avoid artifacts from inactive cells that slips through somehow...(?)
    if dzprop.max() > _cxtgeo.UNDEF_LIMIT:
        raise RuntimeError('Bug: DZ with unphysical values present')

    xi, yi = self.get_xy_values()

    first = True
    for k0 in range(nlay):

        k1 = k0 + 1

        numz = int(round(zoneprop[::, ::, k0].mean()))
        if numz < zone_minmax[0] or numz > zone_minmax[1]:
            continue

        logger.info('Mapping for ' + str(k1) + '...')

        if first:
            wsum = np.zeros((self._ncol, self._nrow), order='F')
            dzsum = np.zeros((self._ncol, self._nrow), order='F')
            first = False

        xc = xprop[::, ::, k0].ravel(order='F')
        yc = yprop[::, ::, k0].ravel(order='F')
        mv = mprop[::, ::, k0].ravel(order='F')
        dz = dzprop[::, ::, k0].ravel(order='F')

        mvdz = mv * dz

        try:
            mvdzi = scipy.interpolate.griddata((xc, yc),
                                               mvdz,
                                               (xi, yi),
                                               method='linear',
                                               fill_value=0.0)
        except ValueError:
            continue

        try:
            dzi = scipy.interpolate.griddata((xc, yc),
                                             dz,
                                             (xi, yi),
                                             method='linear',
                                             fill_value=0.0)
        except ValueError:
            continue

        logger.debug(mvdzi.shape)
        mvdzi = np.asfortranarray(mvdzi)
        dzi = np.asfortranarray(dzi)

        wsum = wsum + mvdzi
        dzsum = dzsum + dzi

    dzmask = dzsum.copy()

    dzsum[dzsum == 0.0] = 1e-20
    vv = wsum / dzsum
    vv = ma.masked_invalid(vv)

    # apply the mask from the DZ map to mask the result
    dzmask = ma.masked_less(dzmask, 1e-20)
    dzmask = ma.getmaskarray(dzmask)

    if truncate_le:
        vv = ma.masked_less(vv, truncate_le)

    vv.mask = dzmask
    self.values = vv


def hc_thickness_3dprops_gridding(self, xprop=None, yprop=None,
                                  hcpfzprop=None, zoneprop=None,
                                  zone_minmax=None, dzprop=None,
                                  zone_avg=False, coarsen=1,
                                  mask_outside=False):

    # NOTE:_
    # - Inputs are pure 3D numpies, not masked!
    # - Xprop and yprop must be made for all cells

    if zone_minmax is None:
        raise ValueError('zone_minmax is required')

    if dzprop is None:
        raise ValueError('DZ property is required')

    xprop, yprop, zoneprop, hcpfzprop, dzprop = _zone_averaging(
        xprop,
        yprop,
        zoneprop,
        zone_minmax,
        coarsen,
        zone_avg,
        dzprop,
        hcprop=hcpfzprop)

    ncol, nrow, nlay = xprop.shape

    # rotation in mesh coords are OK!
    xi, yi = self.get_xy_values()

    # filter and compute per K layer (start count on 0)
    for k0 in range(nlay):

        k1 = k0 + 1   # layer counting base is 1 for k1

        logger.info('Mapping for (combined) layer no ' + str(k1) + '...')

        if k1 == 1:
            logger.info('Initialize zsum ...')
            zsum = np.zeros((self._ncol, self._nrow), order='F')
            dzsum = np.zeros((self._ncol, self._nrow), order='F')

        # check if in zone
        numz = int(round(zoneprop[:, :, k0].mean()))
        if numz < zone_minmax[0] or numz > zone_minmax[1]:
            logger.info('SKIP (not active zone) numz={}, zone_minmax is {}'
                        .format(numz, zone_minmax))
            continue

        propsum = hcpfzprop[:, :, k0].sum()
        if (abs(propsum) < 1e-12):
            logger.info('Too little HC, skip layer K = {}'.format(k1))
            continue
        else:
            logger.debug('Z property sum is {}'.format(propsum))

        xc = xprop[:, :, k0].ravel(order='F')
        yc = yprop[:, :, k0].ravel(order='F')
        zc = hcpfzprop[:, :, k0].ravel(order='F')

        xcc = xc.copy()
        xc = xc[xcc < _cxtgeo.UNDEF_LIMIT]
        yc = yc[xcc < _cxtgeo.UNDEF_LIMIT]
        zc = zc[xcc < _cxtgeo.UNDEF_LIMIT]

        if mask_outside:
            dzc = dzprop[:, :, k0].ravel(order='F')
            dzc = dzc[xcc < _cxtgeo.UNDEF_LIMIT]
            try:
                dzi = scipy.interpolate.griddata((xc, yc), dzc, (xi, yi),
                                                 method='linear',
                                                 fill_value=0.0)
            except ValueError as ve:
                logger.info('Not able to grid layer {} ({})'.format(k1, ve))
                continue

            dzi = np.asfortranarray(dzi)
            dzsum = dzsum + dzi

        try:
            zi = scipy.interpolate.griddata((xc, yc), zc, (xi, yi),
                                            method='linear',
                                            fill_value=0.0)
        except ValueError as ve:
            logger.info('Not able to grid layer {} ({})'.format(k1, ve))
            continue

        zi = np.asfortranarray(zi)
        logger.info('ZI shape is {} and flags {}'.format(zi.shape, zi.flags))

        zsum = zsum + zi
        logger.info('Sum of HCPB layer is {}'.format(zsum.mean()))

    if mask_outside:
        zsum = ma.masked_where(dzsum < 1e-20, zsum)
    else:
        zsum = ma.array(zsum)  # so the result becomes a ma array

    self.values = zsum

    logger.info('Exit from hc_thickness_from_3dprops')

    return True


def _zone_averaging(xprop, yprop, zoneprop, zone_minmax, coarsen,
                    zone_avg, dzprop, mprop=None, hcprop=None):

    # Change the 3D numpy array so they get layers by
    # averaging across zones. This may speed up a lot,
    # but will reduce the resolution.
    # The x y coordinates shall be averaged (ideally
    # with thickness weighting...) while e.g. hcpfzprop
    # must be summed.
    # Somewhat different processing whether this is a hc thickness
    # or an average.

    uprops = {'xprop': xprop, 'yprop': yprop, 'zoneprop': zoneprop,
              'dzprop': dzprop}

    # some sanity checks
    for name, pp in uprops.items():
        minpp = pp.min()
        maxpp = pp.max()
        logger.info('Min Max for {} is {} .. {}'.format(name, minpp, maxpp))

    qhc = False
    if hcprop is not None:
        qhc = True
        hpr = hcprop

    if mprop is not None:
        mpr = mprop

    if coarsen > 1:
        xpr = xprop[::coarsen, ::coarsen, ::].copy(order='F')
        ypr = yprop[::coarsen, ::coarsen, ::].copy(order='F')
        zpr = zoneprop[::coarsen, ::coarsen, ::].copy(order='F')
        dpr = dzprop[::coarsen, ::coarsen, ::].copy(order='F')
        if qhc:
            hpr = hcprop[::coarsen, ::coarsen, ::].copy(order='F')
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
            if qhc:
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
            if qhc:
                hpr2 = ma.sum(hpr2, axis=2)
            else:
                mpr2 = ma.average(mpr2, weights=normed_dz, axis=2)

            dpr2 = ma.sum(dpr2, axis=2)
            newx.append(xpr2)
            newy.append(ypr2)
            newz.append(zpr2)
            newd.append(dpr2)
            if qhc:
                newh.append(hpr2)
            else:
                newm.append(mpr2)

        xpr = ma.dstack(newx)
        ypr = ma.dstack(newy)
        zpr = ma.dstack(newz)
        dpr = ma.dstack(newd)
        if qhc:
            hpr = ma.dstack(newh)
        else:
            mpr = ma.dstack(newm)

    xpr = ma.filled(xpr, fill_value=_cxtgeo.UNDEF)
    ypr = ma.filled(ypr, fill_value=_cxtgeo.UNDEF)
    zpr = ma.filled(zpr, fill_value=0)
    dpr = ma.filled(dpr, fill_value=0.0)

    for a in [xpr, ypr, zpr]:
        logger.info('Reduced shape of ... is {}'.format(a.shape))

    if qhc:
        hpr = ma.filled(hpr, fill_value=0.0)
        return xpr, ypr, zpr, hpr, dpr
    else:
        mpr = ma.filled(mpr, fill_value=0.0)
        return xpr, ypr, zpr, mpr, dpr


def _zone_averaging2(xprop, yprop, zoneprop, zone_minmax, coarsen,
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
    for name, pp in uprops.items():
        minpp = pp.min()
        maxpp = pp.max()
        logger.info('Min Max for {} is {} .. {}'.format(name, minpp, maxpp))

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
