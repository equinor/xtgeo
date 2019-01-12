# -*- coding: utf-8 -*-
"""Roxar API functions for XTGeo Grid Geometry"""
import numpy as np
from pkg_resources import parse_version as pver

try:
    import roxar
    from roxar import __version__ as ROXVER
    roxmsg = 'ROXVER: {}'.format(ROXVER)
except ImportError as roxmsg:
    pass

import xtgeo.cxtgeo.cxtgeo as _cxtgeo

from xtgeo.common import XTGeoDialog

xtg = XTGeoDialog()

logger = xtg.functionlogger(__name__)

# logger.info(roxmsg)

# self is Grid() instance
_cxtgeo.xtg_verbose_file('NONE')
xtg_verbose_level = xtg.syslevel


def import_grid_roxapi(self, projectname, gname, realisation, dimonly, info):
    """Import a Grid via ROXAR API spec."""

    def _load_grid(proj):
        """Inner function to load a grid"""

        logger.info('Loading grid ...')
        try:
            if gname not in proj.grid_models:
                raise KeyError('No such gridmodel: {}'.format(gname))

            logger.info('Get roxgrid...')
            roxgrid = proj.grid_models[gname].get_grid()

            if dimonly:
                corners = None
            else:
                logger.info('Get corners...')
                corners = roxgrid.get_cell_corners_by_index()

            if info:
                _display_roxapi_grid_info(self, proj, roxgrid, corners)

            logger.info('Convert to XTGeo internals...')
            _convert_to_xtgeo_grid(self, roxgrid, corners)

        except KeyError as keyerror:
            raise RuntimeError(keyerror)

    logger.info('Opening a RMS project ...')
    if projectname is not None and isinstance(projectname, str):
        # outside a RMS project
        with roxar.Project.open(projectname, readonly=True) as proj:
            _load_grid(proj)

    else:
        # inside a RMS project
        _load_grid(projectname)


def _display_roxapi_grid_info(self, proj, roxgrid, corners):
    # in prep!
    """Push info to screen (mostly for debugging)"""
    cpgeom = False
    if pver(ROXVER) >= pver('1.3'):
        cpgeom = True

    indexer = roxgrid.grid_indexer
    ncol, nrow, nlay = indexer.dimensions

    if cpgeom:
        xtg.say('ROXAPI with support for CornerPointGeometry')
        geom = roxgrid.get_geometry()
        defined_cells = geom.get_defined_cells()
        xtg.say('Defined cells \n{}'.format(defined_cells))

        xtg.say('IJK handedness: {}'.format(geom.ijk_handedness))
        tops, bottoms, depth = geom.get_pillar_data(0, 0)
        xtg.say('For pillar 0, 0\n')
        xtg.say('Tops\n{}'.format(tops))
        xtg.say('Bots\n{}'.format(bottoms))
        xtg.say('Depths\n{}'.format(depth))


def _convert_to_xtgeo_grid(self, roxgrid, corners):
    """Convert from RMS API to XTGeo API"""

    logger.info('Call the ROXAPI grid indexer')
    indexer = roxgrid.grid_indexer

    ncol, nrow, nlay = indexer.dimensions
    ntot = ncol * nrow * nlay

    # update other attributes
    self._ncol = ncol
    self._nrow = nrow
    self._nlay = nlay

    if corners is None:
        logger.info('Asked for dimensions_only: No geometry read!')
        return

    logger.info('Get active cells')
    mybuffer = np.ndarray(indexer.dimensions, dtype=np.int32)

    mybuffer.fill(0)

    logger.info('Get cell numbers')
    cellno = indexer.get_cell_numbers_in_range((0, 0, 0), indexer.dimensions)

    logger.info('Reorder...')
    ijk = indexer.get_indices(cellno)

    iind = ijk[:, 0]
    jind = ijk[:, 1]
    kind = ijk[:, 2]

    pvalues = np.ones(len(cellno))
    pvalues[cellno] = 1
    mybuffer[iind, jind, kind] = pvalues[cellno]

    actnum = mybuffer

    if pver(ROXVER) < pver('1.3'):
        logger.info('Handedness %s', indexer.handedness)
    else:
        logger.info('Handedness %s', indexer.ijk_handedness)

    corners = corners.ravel(order='K')
    actnum = actnum.ravel(order='K')

    logger.info('Convert to C pointers...')

    nnum = ncol * nrow * nlay * 24
    ccorners = _cxtgeo.new_doublearray(nnum)
    ntot = ncol * nrow * nlay
    cactnum = _cxtgeo.new_intarray(ntot)
    ncoord = (ncol + 1) * (nrow + 1) * 2 * 3
    nzcorn = ncol * nrow * (nlay + 1) * 4

    self._p_coord_v = _cxtgeo.new_doublearray(ncoord)
    self._p_zcorn_v = _cxtgeo.new_doublearray(nzcorn)
    self._p_actnum_v = _cxtgeo.new_intarray(ntot)

    _cxtgeo.swig_numpy_to_carr_1d(corners, ccorners)
    _cxtgeo.swig_numpy_to_carr_i1d(actnum, cactnum)

    logger.info('Calling C function...')

    # next task is to convert geometry to cxtgeo internal format
    _cxtgeo.grd3d_conv_roxapi_grid(ncol, nrow, nlay, ntot, cactnum, ccorners,
                                   self._p_coord_v, self._p_zcorn_v,
                                   self._p_actnum_v, xtg_verbose_level)

    logger.info('Calling C function done...')

    _cxtgeo.delete_doublearray(ccorners)
    _cxtgeo.delete_intarray(cactnum)


def export_grid_roxapi(self, projectname, gname, realisation):
    """Export (i.e. store in RMS) via ROXAR API spec.

    This is possible from version ROXAPI ver 1.3, where the CornerPointGeometry
    class is defined.
    """

    if pver(ROXVER) < pver('1.3'):
        raise NotImplementedError('Export is not implemented for ROXAPI '
                                  'version {}. SKIP!'.format(ROXVER))
