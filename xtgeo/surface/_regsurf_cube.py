# -*- coding: utf-8 -*-
"""Regular surface vs Cube"""

import numpy as np
import numpy.ma as ma

import cxtgeo.cxtgeo as _cxtgeo
from xtgeo.common import XTGeoDialog

xtg = XTGeoDialog()

logger = xtg.basiclogger(__name__)
_cxtgeo.xtg_verbose_file('NONE')

xtg_verbose_level = xtg.get_syslevel()


def slice_cube(rmap, cube, zsurf=None, sampling='nearest', mask=True):
    """Private funct ot do the Cube slicing."""

    if zsurf is not None:
        other = zsurf
    else:
        other = rmap.copy()

    if not rmap.compare_topology(other):
        raise Exception

    if mask:
        opt2 = 0
    else:
        opt2 = 1

    cubeval1d = np.ravel(cube.values, order='F')

    nsurf = rmap.ncol * rmap.nrow

    usesampling = 0
    if sampling == 'nearest':
        usesampling = 1

    logger.debug('Running method from C... (using typemaps for numpies!:')
    istat, v1d = _cxtgeo.surf_slice_cube(cube.ncol,
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
                                         rmap.ncol,
                                         rmap.nrow,
                                         rmap.xori,
                                         rmap.xinc,
                                         rmap.yori,
                                         rmap.yinc,
                                         rmap.rotation,
                                         other.get_zval(),
                                         nsurf,
                                         usesampling, opt2,
                                         xtg_verbose_level)

    if istat != 0:
        logger.warning('Seem to be rotten')

    rmap.set_zval(v1d)


def slice_cube_window(rmap, cube, zsurf=None, sampling='nearest', mask=True,
                      zrange=10, ndiv=None, attribute='max'):
    """Slice Cube with a window and extract attribute(s)

    The zrange is one-sided (on order to secure a centered input); hence
    of zrange is 5 than the fill window is 10.
    """
    logger.info('Slice cube window method')

    if zsurf is not None:
        other = zsurf
    else:
        other = rmap.copy()

    ndivmode = 'user setting'
    if ndiv is None:
        ndivmode = 'auto'
        ndiv = int(2 * zrange / cube.zinc)
        if ndiv < 1:
            ndiv = 1
            logger.warning('NDIV < 1; reset to 1')

    logger.info('NDIV is set to {} ({})'.format(ndiv, ndivmode))

    # This will run slice in a loop within a window. Then, numpy methods
    # are applied to get the attributes

    npcollect = []
    zcenter = other.copy()
    zcenter.slice_cube(cube, sampling=sampling, mask=mask)
    npcollect.append(zcenter.values)

    zincr = zrange / float(ndiv)

    logger.info('ZINCR is {}'.format(zincr))

    # collect above the original surface
    for i in range(ndiv):
        ztmp = other.copy()
        ztmp.values -= zincr * (i + 1)
        logger.info('Mean of depth slice is {}'.format(ztmp.values.mean()))
        ztmp.slice_cube(cube, sampling=sampling, mask=mask)
        logger.info('Mean of cube slice is {}'.format(ztmp.values.mean()))
        npcollect.append(ztmp.values)

    # collect below the original surface
    for i in range(ndiv):
        ztmp = other.copy()
        ztmp.values += zincr * (i + 1)
        logger.info('Mean of depth slice is {}'.format(ztmp.values.mean()))
        ztmp.slice_cube(cube, sampling=sampling, mask=mask)
        logger.info('Mean of cube slice is {}'.format(ztmp.values.mean()))
        npcollect.append(ztmp.values)

    stacked = ma.dstack(npcollect)

    if attribute == 'max':
        attvalues = ma.max(stacked, axis=2)
    elif attribute == 'min':
        attvalues = ma.min(stacked, axis=2)
    elif attribute == 'rms':
        attvalues = np.sqrt(ma.mean(np.square(stacked), axis=2))
    elif attribute == 'var':
        attvalues = ma.var(stacked, axis=2)
    else:
        etxt = 'Invalid attribute applied: {}'.format(attribute)
        raise ValueError(etxt)

    rmap.values = attvalues
    logger.info('Mean of cube attribute is {}'.format(rmap.values.mean()))
