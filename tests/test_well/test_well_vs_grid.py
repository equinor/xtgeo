from __future__ import annotations

import logging
import pathlib

import pytest

import xtgeo

logger = logging.getLogger(__name__)

WFILE = pathlib.Path("wells/reek/1/OP_1.w")
GFILE = pathlib.Path("3dgrids/reek/REEK.EGRID")
PFILE = pathlib.Path("3dgrids/reek/REEK.INIT")


@pytest.fixture(name="loadwell1")
def fixture_loadwell1(testdata_path):
    """Fixture for loading a well (pytest setup)"""
    logger.info("Load well 1")
    return xtgeo.well_from_file(testdata_path / WFILE)


@pytest.fixture(name="loadgrid1")
def fixture_loadgrid1(testdata_path):
    """Fixture for loading a grid (pytest setup)"""
    logger.info("Load grid 1")
    return xtgeo.grid_from_file(testdata_path / GFILE)


@pytest.fixture(name="loadporo1")
def fixture_loadporo1(loadgrid1, testdata_path):
    """Fixture for loading a grid poro values (pytest setup)"""
    logger.info("Load PORO 1")
    grd = loadgrid1
    return xtgeo.gridproperty_from_file(testdata_path / PFILE, name="PORO", grid=grd)


def test_make_ijk_grid(loadwell1, loadgrid1):
    """Import well from and grid and make I J K logs"""

    mywell = loadwell1
    mygrid = loadgrid1

    mywell.make_ijk_from_grid(mygrid)

    df = mywell.get_dataframe(copy=False)

    assert int(df.iloc[4850]["ICELL"]) == 29
    assert int(df.iloc[4850]["JCELL"]) == 28
    assert int(df.iloc[4850]["KCELL"]) == 13
    assert int(df.iloc[4847]["KCELL"]) == 12

    assert int(df.iloc[4775]["ICELL"]) == 29
    assert int(df.iloc[4775]["JCELL"]) == 28
    assert int(df.iloc[4775]["KCELL"]) == 1


def test_well_get_gridprops(tmp_path, loadwell1, loadgrid1, loadporo1):
    """Import well from and grid and make I J K logs"""

    mywell = loadwell1
    mygrid = loadgrid1
    myporo = loadporo1

    mywell.get_gridproperties(myporo, mygrid)

    myactnum = mygrid.get_actnum()
    myactnum.codes = {0: "INACTIVE", 1: "ACTIVE"}
    myactnum.describe()

    mywell.get_gridproperties(myactnum, mygrid)
    mywell.to_file(tmp_path / "w_from_gprops.w")
    assert mywell.get_dataframe().iloc[4775]["PORO_model"] == pytest.approx(
        0.2741, abs=0.001
    )
    assert mywell.get_dataframe().iloc[4775]["ACTNUM_model"] == 1
    assert mywell.isdiscrete("ACTNUM_model") is True


def test_well_gridprops_zone(loadwell1, testdata_path):
    """Test getting logrecords from discrete gridzones"""
    grid = xtgeo.grid_from_file(
        testdata_path / pathlib.Path("3dgrids/reek/reek_sim_grid.roff")
    )
    gridzones = xtgeo.gridproperty_from_file(
        testdata_path / pathlib.Path("3dgrids/reek/reek_sim_zone.roff"),
        grid=grid,
    )
    gridzones.name = "Zone"

    well = loadwell1
    well.get_gridproperties(gridzones, grid)
    well.zonelogname = "Zone_model"

    assert well.get_logrecord(well.zonelogname) == {
        1: "Below_Top_reek",
        2: "Below_Mid_reek",
        3: "Below_Low_reek",
    }
