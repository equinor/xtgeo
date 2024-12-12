from os.path import join

import pytest

import xtgeo

xtg = xtgeo.common.XTGeoDialog()
logger = xtg.basiclogger(__name__)


def test_get_well_x_surf(testdata_path):
    """Getting XYZ, MD for well where crossing a surface"""

    WFILE = join(testdata_path, "wells/battle/1/WELLX.rmswell")
    SFILE = join(testdata_path, "surfaces/etc/battle_1330.gri")

    wll = xtgeo.well_from_file(WFILE, mdlogname="Q_MDEPTH")
    surf = xtgeo.surface_from_file(SFILE)
    top = wll.get_surface_picks(surf)

    assert top.get_dataframe().Q_MDEPTH[5] == pytest.approx(5209.636860, abs=0.001)
