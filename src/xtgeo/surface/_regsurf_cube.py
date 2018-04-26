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

# self = RegularSurface instance!


def slice_cube(self, cube, zsurf=None, sampling='nearest', mask=True):
    """Private function for the Cube slicing."""

    if zsurf is not None:
        other = zsurf
    else:
        other = self.copy()

    if not self.compare_topology(other):
        raise Exception

    if mask:
        opt2 = 0
    else:
        opt2 = 1

    cubeval1d = np.ravel(cube.values, order='C')

    nsurf = self.ncol * self.nrow

    usesampling = 0
    if sampling == 'trilinear':
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
                                         self.ncol,
                                         self.nrow,
                                         self.xori,
                                         self.xinc,
                                         self.yori,
                                         self.yinc,
                                         self.rotation,
                                         other.get_values1d(),
                                         nsurf,
                                         usesampling, opt2,
                                         xtg_verbose_level)

    if istat != 0:
        logger.warning('Seem to be rotten')

    self.set_values1d(v1d)


def slice_cube_window(self, cube, zsurf=None, other=None,
                      other_position='below',
                      sampling='nearest', mask=True,
                      zrange=10, ndiv=None, attribute='max',
                      maskthreshold=0.1):
    """Slice Cube with a window and extract attribute(s)

    The zrange is one-sided (on order to secure a centered input); hence
    of zrange is 5 than the fill window is 10.

    The maskthreshold is only valid for surfaces; if isochore is less than
    given value then the result will be masked.
    """
    logger.info('Slice cube window method')

    if zsurf is not None:
        this = zsurf
    else:
        this = self.copy()

    if other is not None:
        zdelta = np.absolute(this.values - other.values)
        zrange = zdelta.max()

    ndivmode = 'user setting'
    if ndiv is None:
        ndivmode = 'auto'
        ndiv = int(2 * zrange / cube.zinc)
        if ndiv < 1:
            ndiv = 1
            logger.warning('NDIV < 1; reset to 1')

    logger.info('ZRANGE is {}'.format(zrange))
    logger.info('NDIV is set to {} ({})'.format(ndiv, ndivmode))

    # This will run slice in a loop within a window. Then, numpy methods
    # are applied to get the attributes

    if other is None:
        attvalues = _slice_constant_window(this, cube, sampling, zrange,
                                           ndiv, mask, attribute)
    else:
        attvalues = _slice_between_surfaces(this, cube, sampling, other,
                                            other_position, zrange,
                                            ndiv, mask, attribute,
                                            maskthreshold)

    self.values = attvalues
    logger.info('Mean of cube attribute is {}'.format(self.values.mean()))


def _slice_constant_window(this, cube, sampling, zrange,
                           ndiv, mask, attribute):
    """Slice a window, (contant in vertical extent)."""
    npcollect = []
    zcenter = this.copy()
    zcenter.slice_cube(cube, sampling=sampling, mask=mask)
    npcollect.append(zcenter.values)

    zincr = zrange / float(ndiv)

    logger.info('ZINCR is {}'.format(zincr))

    # collect above the original surface
    for i in range(ndiv):
        ztmp = this.copy()
        ztmp.values -= zincr * (i + 1)
        logger.info('Mean of depth slice is {}'.format(ztmp.values.mean()))
        ztmp.slice_cube(cube, sampling=sampling, mask=mask)
        logger.info('Mean of cube slice is {}'.format(ztmp.values.mean()))
        npcollect.append(ztmp.values)

    # collect below the original surface
    for i in range(ndiv):
        ztmp = this.copy()
        ztmp.values += zincr * (i + 1)
        logger.info('Mean of depth slice is {}'.format(ztmp.values.mean()))
        ztmp.slice_cube(cube, sampling=sampling, mask=mask)
        logger.info('Mean of cube slice is {}'.format(ztmp.values.mean()))
        npcollect.append(ztmp.values)

    stacked = ma.dstack(npcollect)

    attvalues = _attvalues(attribute, stacked)
    return attvalues


def _slice_between_surfaces(this, cube, sampling, other, other_position,
                            zrange, ndiv, mask, attribute, mthreshold):
    """Slice and find values bewteen two surface"""

    npcollect = []
    zincr = zrange / float(ndiv)

    zcenter = this.copy()
    zcenter.slice_cube(cube, sampling=sampling, mask=mask)
    npcollect.append(zcenter.values)

    # collect below or above the original surface
    if other_position == 'above':
        mul = -1
    else:
        mul = 1

    for i in range(ndiv):
        ztmp = this.copy()
        ztmp.values += zincr * (i + 1) * mul
        zvalues = ztmp.values.copy()
        logger.info('Mean of depth slice is {}'.format(ztmp.values.mean()))
        ztmp.slice_cube(cube, sampling=sampling, mask=mask)

        diff = mul * (other.values - zvalues)

        values = ztmp.values
        values = ma.masked_where(diff < 0.0, values)

        logger.info('Diff min and max {} {}'.format(diff.min(), diff.max()))
        logger.info('Mean of cube slice is {}'.format(values.mean()))
        logger.info('Number of nonmasked elements {}'.format(values.count()))

        npcollect.append(values)

    stacked = ma.dstack(npcollect)

    attvalues = _attvalues(attribute, stacked)

    # for cases with erosion, the two surfaces are equal
    isovalues = mul * (other.values - this.values)
    attvalues = ma.masked_where(isovalues < mthreshold, attvalues)

    return attvalues


def _attvalues(attribute, stacked):
    """Attibute values computed in numpy.ma stack."""
    if attribute == 'max':
        attvalues = ma.max(stacked, axis=2)
    elif attribute == 'min':
        attvalues = ma.min(stacked, axis=2)
    elif attribute == 'rms':
        attvalues = np.sqrt(ma.mean(np.square(stacked), axis=2))
    elif attribute == 'var':
        attvalues = ma.var(stacked, axis=2)
    elif attribute == 'mean':
        attvalues = ma.mean(stacked, axis=2)
    else:
        etxt = 'Invalid attribute applied: {}'.format(attribute)
        raise ValueError(etxt)

    if not attvalues.flags['C_CONTIGUOUS']:
        mask = ma.getmaskarray(attvalues)
        mask = np.asanyarray(mask, order='C')
        attvalues = np.asanyarray(attvalues, order='C')
        attvalues = ma.array(attvalues, mask=mask, order='C')

    return attvalues
