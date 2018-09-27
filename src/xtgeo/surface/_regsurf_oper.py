# coding: utf-8
"""Various operations"""
from __future__ import print_function, absolute_import

import numpy.ma as ma

import xtgeo.cxtgeo.cxtgeo as _cxtgeo
from xtgeo.common import XTGeoDialog

xtg = XTGeoDialog()

logger = xtg.functionlogger(__name__)

xtg_verbose_level = xtg.get_syslevel()
if xtg_verbose_level < 0:
    xtg_verbose_level = 0

_cxtgeo.xtg_verbose_file('NONE')


def operations_two(self, other, oper='add'):
    """General operations between two maps"""

    okstatus = self.compare_topology(other)

    if not okstatus:
        other.resample(self)

    if oper == 'add':
        self.values = self.values + other.values

    if oper == 'sub':
        self.values = self.values - other.values

    if oper == 'mul':
        self.values = self.values * other.values

    if oper == 'div':
        self.values = self.values * other.values


def resample(surf, other):
    """Resample from other surface object to this surf"""

    logger.info('Resampling...')

    svalues = surf.get_values1d()

    ier = _cxtgeo.surf_resample(other._ncol, other._nrow,
                                other._xori, other._xinc,
                                other._yori, other._yinc,
                                other._yflip, other._rotation,
                                other.get_values1d(),
                                surf._ncol, surf._nrow,
                                surf._xori, surf._xinc,
                                surf._yori, surf._yinc,
                                surf._yflip, surf._rotation,
                                svalues,
                                0,
                                xtg_verbose_level)

    if ier != 0:
        raise RuntimeError('Resampling went wrong, '
                           'code is {}'.format(ier))

    surf.set_values1d(svalues)


def distance_from_point(surf, point=(0, 0), azimuth=0.0):

    x, y = point

    svalues = surf.get_values1d()

    # call C routine
    ier = _cxtgeo.surf_get_dist_values(
        surf._xori, surf._xinc, surf._yori, surf._yinc, surf._ncol,
        surf._nrow, surf._rotation, x, y, azimuth, svalues, 0,
        xtg_verbose_level)

    if ier != 0:
        surf.logger.error('Something went wrong...')
        raise RuntimeError('Something went wrong in {}'.format(__name__))

    surf.set_values1d(svalues)


def get_value_from_xy(surf, point=(0.0, 0.0)):

    xcoord, ycoord = point

    zcoord = _cxtgeo.surf_get_z_from_xy(float(xcoord), float(ycoord),
                                        surf.ncol, surf.nrow,
                                        surf.xori, surf.yori, surf.xinc,
                                        surf.yinc, surf.yflip,
                                        surf.rotation,
                                        surf.get_values1d(), xtg_verbose_level)

    if zcoord > surf._undef_limit:
        return None

    return zcoord


def get_xy_value_from_ij(surf, iloc, jloc, zvalues=None):

    if zvalues is None:
        zvalues = surf.get_values1d()

    if 1 <= iloc <= surf.ncol and 1 <= jloc <= surf.nrow:

        ier, xval, yval, value = (
            _cxtgeo.surf_xyz_from_ij(iloc, jloc,
                                     surf.xori, surf.xinc,
                                     surf.yori, surf.yinc,
                                     surf.ncol, surf.nrow, surf._yflip,
                                     surf.rotation, zvalues,
                                     0, xtg_verbose_level))
        if ier != 0:
            surf.logger.critical('Error code {}, contact the author'.
                                 format(ier))
            raise SystemExit('Error code {}'.format(ier))

    else:
        raise ValueError('Index i and/or j out of bounds')

    if value > surf.undef_limit:
        value = None

    return xval, yval, value


def get_xy_values(surf):

    nn = surf.ncol * surf.nrow

    ier, xvals, yvals = (
        _cxtgeo.surf_xy_as_values(surf.xori, surf.xinc,
                                  surf.yori, surf.yinc * surf.yflip,
                                  surf.ncol, surf.nrow,
                                  surf.rotation, nn, nn,
                                  0, xtg_verbose_level))
    if ier != 0:
        surf.logger.critical('Error code {}, contact the author'.
                             format(ier))

    # reshape, then mask using the current Z values mask
    xvals = xvals.reshape((surf.ncol, surf.nrow))
    yvals = yvals.reshape((surf.ncol, surf.nrow))

    mask = ma.getmaskarray(surf.values)
    xvals = ma.array(xvals, mask=mask)
    yvals = ma.array(yvals, mask=mask)

    return xvals, yvals


def get_fence(surf, xyfence):

    cxarr = xyfence[:, 0]
    cyarr = xyfence[:, 1]
    czarr = xyfence[:, 2].copy()

    # czarr will be updated "inplace":
    istat = _cxtgeo.surf_get_zv_from_xyv(cxarr, cyarr, czarr,
                                         surf.ncol, surf.nrow, surf.xori,
                                         surf.yori, surf.xinc, surf.yinc,
                                         surf.yflip, surf.rotation,
                                         surf.get_values1d(),
                                         xtg_verbose_level)

    if istat != 0:
        surf.logger.warning('Seem to be rotten')

    xyfence[:, 2] = czarr
    xyfence = ma.masked_greater(xyfence, surf._undef_limit)
    xyfence = ma.mask_rows(xyfence)

    return xyfence
