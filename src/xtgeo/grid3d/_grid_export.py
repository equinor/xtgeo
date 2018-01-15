# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import
import logging
from xtgeo.common import XTGeoDialog
import cxtgeo.cxtgeo as _cxtgeo

xtg = XTGeoDialog()

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def export_roff(grid, gfile, option):
    """Export grid to ROFF format (binary)"""

    logger.debug('Export to ROFF...')
    _cxtgeo.xtg_verbose_file('NONE')
    xtg_verbose_level = grid._xtg.syslevel

    if grid._nsubs == 0 and not hasattr(grid, '_p_subgrd_v'):
        logger.debug('Create a pointer for _p_subgrd_v ...')
        grid._p_subgrd_v = _cxtgeo.new_intpointer()

    # get the geometrics list to find the xshift, etc
    gx = grid.get_geometrics()

    _cxtgeo.grd3d_export_roff_grid(option, grid._ncol, grid._nrow, grid._nlay,
                                   grid._nsubs, 0, gx[3], gx[5], gx[7],
                                   grid._p_coord_v, grid._p_zcorn_v,
                                   grid._p_actnum_v, grid._p_subgrd_v,
                                   gfile, xtg_verbose_level)

    # skip parameters for now (cf Perl code)

    # end tag
    _cxtgeo.grd3d_export_roff_end(option, gfile, xtg_verbose_level)


def export_grdecl(grid, gfile):
    """Export grid to Eclipse GRDECL format (ascii)"""

    logger.debug('Export to GRDECL...')
    _cxtgeo.xtg_verbose_file('NONE')
    xtg_verbose_level = grid._xtg.syslevel

    _cxtgeo.grd3d_export_grdecl(grid._ncol, grid._nrow, grid._nlay,
                                grid._p_coord_v, grid._p_zcorn_v,
                                grid._p_actnum_v, gfile, xtg_verbose_level)
