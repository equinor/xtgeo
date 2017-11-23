"""Do gridding from 3D parameters"""

import numpy as np
import numpy.ma as ma
import scipy.interpolate

import cxtgeo.cxtgeo as _cxtgeo
from xtgeo.common import XTGeoDialog
xtg = XTGeoDialog()

logger = xtg.functionlogger(__name__)


def avg_from_3d_prop_gridding(surf, xprop=None, yprop=None,
                              mprop=None, dzprop=None, layer_minmax=None,
                              truncate_le=None, zoneprop=None,
                              zone_minmax=None,
                              sampling=1):

    ncol, nrow, nlay = xprop.shape

    if layer_minmax is None:
        layer_minmax = (1, 99999)

    if zone_minmax is None:
        zone_minmax = (1, 99999)

    usezoneprop = True
    if zoneprop is None:
        usezoneprop = False

    # avoid artifacts from inactive cells that slips through somehow...
    dzprop[mprop > _cxtgeo.UNDEF_LIMIT] = 0.0

    logger.info('Layer from: {}'.format(layer_minmax[0]))
    logger.info('Layer to: {}'.format(layer_minmax[1]))
    logger.debug('Layout is {} {} {}'.format(ncol, nrow, nlay))

    logger.info('Zone from: {}'.format(zone_minmax[0]))
    logger.info('Zone to: {}'.format(zone_minmax[1]))
    logger.info('Zone is :')
    logger.info(zoneprop)

    # do not allow rotation...
    if surf._rotation < -0.1 or surf._rotation > 0.1:
        logger.error('Cannut use rotated maps. Return')
        return

    xmax = surf._xori + surf._xinc * surf._ncol
    ymax = surf._yori + surf._yinc * surf._nrow
    xi = np.linspace(surf._xori, xmax, surf._ncol)
    yi = np.linspace(surf._yori, ymax, surf._nrow)

    xi, yi = np.meshgrid(xi, yi, indexing='ij')

    sf = sampling

    logger.debug('ZONEPROP:')
    logger.debug(zoneprop)
    # compute per K layer (start on count 1)

    first = True
    for k in range(1, nlay + 1):

        if k < layer_minmax[0] or k > layer_minmax[1]:
            logger.info('SKIP LAYER {}'.format(k))
            continue
        else:
            logger.info('USE LAYER {}'.format(k))

        if usezoneprop:
            zonecopy = ma.copy(zoneprop[::sf, ::sf, k - 1:k])

            zzz = int(round(zonecopy.mean()))
            if zzz < zone_minmax[0] or zzz > zone_minmax[1]:
                continue

        logger.info('Mapping for ' + str(k) + '...')

        xcopy = np.copy(xprop[::, ::, k - 1:k])
        ycopy = np.copy(yprop[::, ::, k - 1:k])
        zcopy = np.copy(mprop[::, ::, k - 1:k])
        dzcopy = np.copy(dzprop[::, ::, k - 1:k])

        if first:
            wsum = np.zeros((surf._ncol, surf._nrow))
            dzsum = np.zeros((surf._ncol, surf._nrow))
            first = False

        logger.debug(zcopy)

        xc = np.reshape(xcopy, -1, order='F')
        yc = np.reshape(ycopy, -1, order='F')
        zv = np.reshape(zcopy, -1, order='F')
        dz = np.reshape(dzcopy, -1, order='F')

        zvdz = zv * dz

        try:
            zvdzi = scipy.interpolate.griddata((xc[::sf], yc[::sf]),
                                               zvdz[::sf],
                                               (xi, yi),
                                               method='linear',
                                               fill_value=0.0)
        except ValueError:
            continue

        try:
            dzi = scipy.interpolate.griddata((xc[::sf], yc[::sf]),
                                             dz[::sf],
                                             (xi, yi),
                                             method='linear',
                                             fill_value=0.0)
        except ValueError:
            continue

        logger.debug(zvdzi.shape)

        wsum = wsum + zvdzi
        dzsum = dzsum + dzi

        logger.debug(wsum[0:20, 0:20])

    dzsum[dzsum == 0.0] = 1e-20
    vv = wsum / dzsum
    vv = ma.masked_invalid(vv)
    if truncate_le:
        vv = ma.masked_less(vv, truncate_le)

    surf.values = vv


def hc_thickness_3dprops_gridding(surf, xprop=None, yprop=None,
                                  hcpfzprop=None, zoneprop=None,
                                  zone_minmax=None, layer_minmax=None):

    for a in [hcpfzprop, xprop, yprop, zoneprop]:
        logger.debug('{} MIN MAX MEAN {} {} {}'.
                     format(str(a), a.min(), a.max(), a.mean()))

    ncol, nrow, nlay = xprop.shape

    if layer_minmax is None:
        layer_minmax = (1, nlay)
    else:
        minmax = list(layer_minmax)
        if minmax[0] < 1:
            minmax[0] = 1
        if minmax[1] > nlay:
            minmax[1] = nlay
        layer_minmax = tuple(minmax)

    if zone_minmax is None:
        zone_minmax = (1, 99999)

    logger.debug('Grid layout is {} {} {}'.format(ncol, nrow, nlay))

    # do not allow rotation...
    if surf._rotation < -0.1 or surf._rotation > 0.1:
        logger.error('Cannot use rotated maps. Return')
        return False

    xmax = surf._xori + surf._xinc * surf._ncol
    ymax = surf._yori + surf._yinc * surf._nrow
    xi = np.linspace(surf._xori, xmax, surf._ncol)
    yi = np.linspace(surf._yori, ymax, surf._nrow)

    xi, yi = np.meshgrid(xi, yi, indexing='ij')

    # filter and compute per K layer (start count on 0)
    for k0 in range(layer_minmax[0] - 1, layer_minmax[1]):

        k1 = k0 + 1   # layer counting base is 1 for k1

        logger.info('Mapping for layer ' + str(k1) + '...')
        logger.debug('K0 counter is {}'.format(k0))

        if k1 == layer_minmax[0]:
            logger.info('Initialize zsum ...')
            zsum = np.zeros((surf._ncol, surf._nrow))

        # this should actually never happen...
        if k1 < layer_minmax[0] or k1 > layer_minmax[1]:
            logger.info('SKIP (layer_minmax)')
            continue

        zonecopy = np.copy(zoneprop[:, :, k0])

        logger.debug('Zone MEAN is {}'.format(zonecopy.mean()))

        actz = zonecopy.mean()
        if actz < zone_minmax[0] or actz > zone_minmax[1]:
            logger.info('SKIP (not active zone)')
            continue

        # get slices per layer of relevant props
        xcopy = np.copy(xprop[:, :, k0])
        ycopy = np.copy(yprop[:, :, k0])
        zcopy = np.copy(hcpfzprop[:, :, k0])

        propsum = zcopy.sum()
        if (abs(propsum) < 1e-12):
            logger.debug('Z property sum is {}'.format(propsum))
            logger.info('Too little HC, skip layer K = {}'.format(k1))
            continue
        else:
            logger.debug('Z property sum is {}'.format(propsum))

        # debugging info...
        logger.debug(xi.shape)
        logger.debug(yi.shape)
        logger.debug('XI min and max {} {}'.format(xi.min(),
                                                   xi.max()))
        logger.debug('YI min and max {} {}'.format(yi.min(),
                                                   yi.max()))
        logger.debug('XPROP min and max {} {}'.format(xprop.min(),
                                                      xprop.max()))
        logger.debug('YPROP min and max {} {}'.format(yprop.min(),
                                                      yprop.max()))
        logger.debug('HCPROP min and max {} {}'
                     .format(hcpfzprop.min(), hcpfzprop.max()))

        # need to make arrays 1D
        logger.debug('Reshape and filter ...')
        x = np.reshape(xcopy, -1, order='F')
        y = np.reshape(ycopy, -1, order='F')
        z = np.reshape(zcopy, -1, order='F')

        xc = np.copy(x)

        x = x[xc < surf._undef_limit]
        y = y[xc < surf._undef_limit]
        z = z[xc < surf._undef_limit]

        logger.debug('Reshape and filter ... done')

        logger.debug('Map ... layer = {}'.format(k1))

        try:
            zi = scipy.interpolate.griddata((x, y), z, (xi, yi),
                                            method='linear',
                                            fill_value=0.0)
        except ValueError as ve:
            logger.info('Not able to grid layer {} ({})'.format(k1, ve))
            continue

        logger.info('ZI shape is {}'.format(zi.shape))
        logger.debug('Map ... done')

        zsum = zsum + zi
        logger.info('Sum of HCPB layer is {}'.format(zsum.mean()))

    surf.values = zsum
    logger.debug(repr(surf._values))

    logger.debug('Exit from hc_thickness_from_3dprops')

    return True
