from __future__ import print_function, absolute_import

import logging

from xtgeo.common import XTGeoDialog

import cxtgeo.cxtgeo as _cxtgeo

xtg = XTGeoDialog()

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def refine_vertically(grid, rfactor):
    """Refine vertically, proportionally

    The rfactor can be a scalar or (todo:) a dictionary

    Input:
        grid (object): A grid XTGeo object
        rfactor (scalar or dict): Refinement factor
    """

    # rfac is an array with length nlay, and has refinement per
    # layer
    rfac = _cxtgeo.new_intarray(grid.nlay)

    if isinstance(rfactor, dict):
        pass  # later
    else:
        for i in range(grid.nlay):
            _cxtgeo.intarray_setitem(rfac, i, rfactor)

    xtg_verbose_level = xtg.syslevel

    newnlay = grid.nlay * rfactor  # scalar case

    ref_num_act = _cxtgeo.new_intpointer()
    ref_p_zcorn_v = _cxtgeo.new_doublearray(grid.ncol * grid.nrow *
                                            (newnlay + 1) * 4)
    ref_p_actnum_v = _cxtgeo.new_intarray(grid.ncol * grid.nrow * newnlay)

    ier = _cxtgeo.grd3d_refine_vert(grid.ncol,
                                    grid.nrow,
                                    grid.nlay,
                                    grid._p_coord_v,
                                    grid._p_zcorn_v,
                                    grid._p_actnum_v,
                                    newnlay,
                                    ref_p_zcorn_v,
                                    ref_p_actnum_v,
                                    ref_num_act,
                                    rfac,
                                    0,
                                    xtg_verbose_level)

    if ier != 0:
        raise RuntimeError('An error occured in the C routine '
                           'grd3d_refine_vert, code {}'.format(ier))

    # update instance:
    grid._nlay = newnlay
    grid._nactive = _cxtgeo.intpointer_value(ref_num_act)
    grid._p_zcorn_v = ref_p_zcorn_v
    grid._p_actnum_v = ref_p_actnum_v

    return grid
