from __future__ import print_function, absolute_import

import logging

from xtgeo.common import XTGeoDialog

import cxtgeo.cxtgeo as _cxtgeo

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

    nhdiv = kwargs.get('nhdiv')
    toplevel = kwargs.get('toplevel')
    bottomlevel = kwargs.get('bottomlevel')
    region = kwargs.get('region', None)
    region_number = kwargs.get('region_number', None)

    logger.debug('nhdiv: {}'.format(nhdiv))
    logger.debug('toplevel: {}'.format(toplevel))
    logger.debug('bottomlevel: {}'.format(bottomlevel))
    logger.debug('region: {}'.format(region))
    logger.debug('region_number: {}'.format(region_number))

    xtg_verbose_level = xtg.syslevel

    newnlay = grid.nlay * 2 + nhdiv

    hyb_num_act = _cxtgeo.new_intpointer()
    hyb_p_zcorn_v = _cxtgeo.new_doublearray(grid.ncol * grid.nrow *
                                            (newnlay + 1) * 4)
    hyb_p_actnum_v = _cxtgeo.new_intarray(grid.ncol * grid.nrow * newnlay)

    if region is None:
        print('No region...')
        _cxtgeo.grd3d_convert_hybrid(grid.ncol,
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
                                     xtg_verbose_level)
    else:

        region.discrete_to_continuous()

        print(region.values3d[40:50, 40:50, :])

        _cxtgeo.grd3d_convert_hybrid2(grid.ncol,
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
                                      region.cvalues,
                                      region_number,
                                      xtg_verbose_level)

    grid._nlay = newnlay
    grid._nactive = _cxtgeo.intpointer_value(hyb_num_act)
    grid._p_zcorn_v = hyb_p_zcorn_v
    grid._p_actnum_v = hyb_p_actnum_v

    return grid
