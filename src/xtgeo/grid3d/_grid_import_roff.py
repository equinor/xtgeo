# coding: utf-8
"""Private module, Grid Import private functions for ROFF format."""

import xtgeo

from ._roff_grid import RoffGrid

xtg = xtgeo.common.XTGeoDialog()

logger = xtg.functionlogger(__name__)


def import_roff(gfile):
    roff_grid = RoffGrid.from_file(gfile._file)
    return {
        "actnumsv": roff_grid.xtgeo_actnum(),
        "coordsv": roff_grid.xtgeo_coord(),
        "zcornsv": roff_grid.xtgeo_zcorn(),
        "subgrids": roff_grid.xtgeo_subgrids(),
    }
