# -*- coding: utf-8 -*-
"""Regular surface vs Grid3D"""
from __future__ import division, absolute_import
from __future__ import print_function

import xtgeo.cxtgeo.cxtgeo as _cxtgeo
from xtgeo.common import XTGeoDialog
from xtgeo.grid3d import _gridprop_lowlevel

xtg = XTGeoDialog()

logger = xtg.basiclogger(__name__)
_cxtgeo.xtg_verbose_file('NONE')

xtg_verbose_level = xtg.get_syslevel()

# self = RegularSurface instance!


def slice_grid3d(self, prop, zsurf=None, sbuffer=1):
    """Private function for the Grid3D slicing."""

    if zsurf is not None:
        other = zsurf
    else:
        logger.info('The current surface is copied as "other"')
        other = self.copy()
    if not self.compare_topology(other, strict=False):
        raise RuntimeError('Topology of maps differ. Stop!')

    grid = prop.grid

    zslice = other.copy()

    nsurf = self.ncol * self.nrow

    p_prop = _gridprop_lowlevel.update_carray(prop, discrete=False)

    istat, updatedval = _cxtgeo.surf_slice_grd3d(
        self.ncol,
        self.nrow,
        self.xori,
        self.xinc,
        self.yori,
        self.yinc,
        self.rotation,
        self.yflip,
        zslice.get_values1d(),
        nsurf,
        grid.ncol,
        grid.nrow,
        grid.nlay,
        grid._p_coord_v,
        grid._p_zcorn_v,
        grid._p_actnum_v,
        p_prop,
        sbuffer,
        0,
        xtg_verbose_level)

    if istat != 0:
        logger.warning('Problem, ISTAT = {}'.format(istat))

    self.set_values1d(updatedval)

    return istat
