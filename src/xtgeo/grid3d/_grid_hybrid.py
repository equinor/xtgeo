from __future__ import print_function, absolute_import

import logging

from xtgeo.common import XTGeoDialog

import xtgeo.cxtgeo.cxtgeo as _cxtgeo
from xtgeo.grid3d import _gridprop_lowlevel

xtg = XTGeoDialog()

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def make_hybridgrid(grid, **kwargs):
    """Make hybrid grid.

    It changes the grid geometry status of the object.

    Input:
        grid (object): A grid object
        TODO region (object): A region parameter (property object)
        etc...
    """

    nhdiv = kwargs.get("nhdiv")
    toplevel = kwargs.get("toplevel")
    bottomlevel = kwargs.get("bottomlevel")
    region = kwargs.get("region", None)
    region_number = kwargs.get("region_number", None)

    logger.debug("nhdiv: %s", nhdiv)
    logger.debug("toplevel: %s", toplevel)
    logger.debug("bottomlevel: %s", bottomlevel)
    logger.debug("region: %s", region)
    logger.debug("region_number: %s", region_number)

    xtg_verbose_level = xtg.syslevel

    newnlay = grid.nlay * 2 + nhdiv

    hyb_num_act = _cxtgeo.new_intpointer()
    hyb_p_zcorn_v = _cxtgeo.new_doublearray(grid.ncol * grid.nrow * (newnlay + 1) * 4)
    hyb_p_actnum_v = _cxtgeo.new_intarray(grid.ncol * grid.nrow * newnlay)

    if region is None:
        _cxtgeo.grd3d_convert_hybrid(
            grid.ncol,
            grid.nrow,
            grid.nlay,
            grid._p_coord_v,
            grid._p_zcorn_v,
            grid._p_actnum_v,
            newnlay,
            hyb_p_zcorn_v,
            hyb_p_actnum_v,
            hyb_num_act,
            toplevel,
            bottomlevel,
            nhdiv,
            xtg_verbose_level,
        )
    else:

        region.discrete_to_continuous()

        carray_reg = _gridprop_lowlevel.update_carray(region)

        _cxtgeo.grd3d_convert_hybrid2(
            grid.ncol,
            grid.nrow,
            grid.nlay,
            grid._p_coord_v,
            grid._p_zcorn_v,
            grid._p_actnum_v,
            newnlay,
            hyb_p_zcorn_v,
            hyb_p_actnum_v,
            hyb_num_act,
            toplevel,
            bottomlevel,
            nhdiv,
            carray_reg,
            region_number,
            xtg_verbose_level,
        )

        _gridprop_lowlevel.delete_carray(region, carray_reg)

    grid._nlay = newnlay
    grid._p_zcorn_v = hyb_p_zcorn_v
    grid._p_actnum_v = hyb_p_actnum_v

    return grid
