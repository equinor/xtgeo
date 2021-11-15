from os.path import join

import pytest

import xtgeo

from .ecl_run_fixtures import *  # noqa: F401, F403


@pytest.fixture()
def reek_grid(reek_run):
    return reek_run.grid


@pytest.fixture
def reek_poly(testpath):
    return xtgeo.xyz.Polygons(join(testpath, "polygons", "reek", "1", "mypoly.pol"))


def test_grid_inactivate_inside(tmp_path, reek_grid, reek_poly):
    """Inactivate a grid inside polygons"""

    act1 = reek_grid.get_actnum().values3d
    n1 = act1[7, 55, 1]
    assert n1 == 1

    try:
        reek_grid.inactivate_inside(reek_poly, layer_range=(1, 4))
    except RuntimeError as rw:
        print(rw)

    reek_grid.to_file(tmp_path / "reek_inact_ins_pol.roff")

    act2 = reek_grid.get_actnum().values3d
    n2 = act2[7, 55, 1]
    assert n2 == 0


def test_grid_inactivate_outside(tmp_path, reek_grid, reek_poly):
    """Inactivate a grid outside polygons"""
    act1 = reek_grid.get_actnum().values3d
    n1 = act1[3, 56, 1]
    assert n1 == 1

    try:
        reek_grid.inactivate_outside(reek_poly, layer_range=(1, 4))
    except RuntimeError as rw:
        print(rw)

    reek_grid.to_file(tmp_path / "reek_inact_out_pol.roff")

    act2 = reek_grid.get_actnum().values3d
    n2 = act2[3, 56, 1]
    assert n2 == 0

    assert int(act1[20, 38, 4]) == int(act2[20, 38, 4])
