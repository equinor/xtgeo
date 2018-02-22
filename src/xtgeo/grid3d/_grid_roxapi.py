# coding: utf-8
"""Roxar API functions for XTGeo Grid Geometry"""

import cxtgeo.cxtgeo as _cxtgeo
import numpy as np

from xtgeo.common import XTGeoDialog

xtg = XTGeoDialog()

logger = xtg.functionlogger(__name__)

# self is Grid() instance


def import_grid_roxapi(self, projectname, gname, realisation):
    """Import a Grid via ROXAR API spec."""
    import roxar

    logger.info('Opening RMS project ...')
    if projectname is not None and isinstance(projectname, str):
        # outside a RMS project
        with roxar.Project.open(projectname, readonly=True) as proj:

            # Note that values must be extracted within the "with"
            # scope here, as e.g. prop._roxgrid.properties[pname]
            # will lose its reference as soon as we are outside
            # the project

            try:
                if gname not in proj.grid_models:
                    raise KeyError('No such gridmodel: {}'.format(gname))

                roxgrid = proj.grid_models[gname].get_grid()
                corners = roxgrid.get_cell_corners_by_index()

                _convert_to_xtgeo_grid(self, roxgrid, corners)

            except KeyError as keyerror:
                raise RuntimeError(keyerror)

    else:
        # inside a RMS project
        try:
            roxgrid = projectname.grid_models[gname].get_grid()
            corners = roxgrid.get_cell_corners_by_index()

            _convert_to_xtgeo_grid(self, roxgrid, corners)

        except KeyError as keyerror:
            raise RuntimeError(keyerror)


def _convert_to_xtgeo_grid(self, roxgrid, corners):

    _cxtgeo.xtg_verbose_file('NONE')
    xtg_verbose_level = self._xtg.syslevel

    indexer = roxgrid.grid_indexer

    ncol, nrow, nlay = indexer.dimensions
    ntot = ncol * nrow * nlay

    # get the active cell numbers
    mybuffer = np.ndarray(indexer.dimensions, dtype=np.int32)

    mybuffer.fill(0)

    cellno = indexer.get_cell_numbers_in_range((0, 0, 0), indexer.dimensions)

    ijk = indexer.get_indices(cellno)

    iind = ijk[:, 0]
    jind = ijk[:, 1]
    kind = ijk[:, 2]

    pvalues = np.ones(len(cellno))
    pvalues[cellno] = 1
    mybuffer[iind, jind, kind] = pvalues[cellno]

    actnum = mybuffer

    logger.info(indexer.handedness)

    corners = corners.ravel(order='K')
    actnum = actnum.ravel(order='K')

    # convert to C pointer
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

    # update other attributes
    self._ncol = ncol
    self._nrow = nrow
    self._nlay = nlay
