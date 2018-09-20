# -*- coding: utf-8 -*-
"""Regular surface vs Grid3D"""
from __future__ import division, absolute_import
from __future__ import print_function

import cxtgeo.cxtgeo as _cxtgeo
from xtgeo.common import XTGeoDialog
from xtgeo.grid3d import _gridprop_lowlevel

xtg = XTGeoDialog()

logger = xtg.basiclogger(__name__)
_cxtgeo.xtg_verbose_file('NONE')

xtg_verbose_level = xtg.get_syslevel()

# self = RegularSurface instance!


def slice_grid3d(self, prop):
    """Private function for the Grid3D slicing."""

    grid = prop.grid

    zslice = self.copy()

    nsurf = self.ncol * self.nrow

    p_prop = _gridprop_lowlevel.update_carray(prop)

    print('XXXX', p_prop, grid._p_actnum_v)

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
        0,
        1)

    if istat != 0:
        logger.warning('Problem, ISTAT = {}'.format(istat))

    self.set_values1d(updatedval)

    return istat
