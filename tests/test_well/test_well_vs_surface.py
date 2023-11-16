# -*- coding: utf-8 -*-


from os.path import join

import pytest

import xtgeo

xtg = xtgeo.common.XTGeoDialog()
from xtgeo.common import logger
from xtgeo.common.xtgeo_dialog import testdatafolder

TPATH = testdatafolder

WFILE = join(TPATH, "wells/battle/1/WELLX.rmswell")
SFILE = join(TPATH, "surfaces/etc/battle_1330.gri")


def test_get_well_x_surf():
    """Getting XYZ, MD for well where crossing a surface"""

    wll = xtgeo.well_from_file(WFILE, mdlogname="Q_MDEPTH")
    surf = xtgeo.surface_from_file(SFILE)
    top = wll.get_surface_picks(surf)

    assert top.dataframe.Q_MDEPTH[5] == pytest.approx(5209.636860, abs=0.001)
