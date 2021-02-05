# coding: utf-8

import itertools
import os
from os.path import basename, join

import numpy as np
import pytest

import xtgeo
from xtgeo.common import XTGeoDialog

xtg = XTGeoDialog()
logger = xtg.basiclogger(__name__)

if not xtg.testsetup():
    raise SystemExit


@pytest.fixture
def grids_etc_path(testpath):
    return join(testpath, "3dgrids", "etc")


@pytest.fixture
def dual_poro_path(grids_etc_path):
    return join(grids_etc_path, "TEST_DP")


@pytest.fixture
def dual_poro_dual_perm_path(dual_poro_path):
    return dual_poro_path + "DK"


@pytest.fixture
def dual_poro_dual_perm_wg_path(grids_etc_path):
    # same as dual_poro_dual_perm but with water/gas
    # instead of oil/water
    return join(grids_etc_path, "TEST2_DPDK_WG")


class GridCase:
    def __init__(self, path, expected_dimensions, expected_perm):
        self.path = path
        self.expected_dimensions = expected_dimensions
        self.expected_perm = expected_perm

    @property
    def grid(self):
        return xtgeo.grid3d.Grid(self.path + ".EGRID")

    def get_property_from_init(self, name, **kwargs):
        return xtgeo.gridproperty_from_file(
            self.path + ".INIT", grid=self.grid, name=name, **kwargs
        )

    def get_property_from_restart(self, name, date, **kwargs):

        return xtgeo.gridproperty_from_file(
            self.path + ".UNRST", grid=self.grid, date=date, name=name, **kwargs
        )


@pytest.fixture
def dual_poro_case(dual_poro_path):
    return GridCase(dual_poro_path, (5, 3, 1), False)


@pytest.fixture
def dual_poro_dual_perm_case(dual_poro_dual_perm_path):
    return GridCase(dual_poro_dual_perm_path, (5, 3, 1), True)


@pytest.fixture
def dual_poro_dual_perm_wg_case(dual_poro_dual_perm_wg_path):
    return GridCase(dual_poro_dual_perm_wg_path, (5, 3, 1), True)


@pytest.fixture(
    params=[
        "dual_poro_case",
        "dual_poro_dual_perm_case",
        "dual_poro_dual_perm_wg_case",
    ]
)
def dual_cases(request):
    return request.getfixturevalue(request.param)


def test_dual_cases_general_grid(tmpdir, dual_cases):
    assert dual_cases.grid.dimensions == dual_cases.expected_dimensions
    assert dual_cases.grid.dualporo is True
    assert dual_cases.grid.dualperm is dual_cases.expected_perm

    dual_cases.grid.to_file(join(tmpdir, basename(dual_cases.path)) + ".roff")
    dual_cases.grid._dualactnum.to_file(
        join(tmpdir, basename(dual_cases.path) + "dualact.roff")
    )


@pytest.mark.parametrize(
    "date, name, fracture",
    itertools.product([20170121, 20170131], ["SGAS", "SOIL", "SWAT"], [True, False]),
)
def test_dual_cases_restart_property_to_file(tmpdir, dual_cases, date, name, fracture):
    prop = dual_cases.get_property_from_restart(name, date=date, fracture=fracture)
    prop.describe()

    if fracture:
        assert prop.name == f"{name}F_{date}"
    else:
        assert prop.name == f"{name}M_{date}"

    filename = join(tmpdir, basename(dual_cases.path) + str(date) + prop.name + ".roff")
    prop.to_file(filename)

    assert os.path.exists(filename)


@pytest.mark.parametrize(
    "name, fracture",
    itertools.product(["PORO", "PERMX", "PERMY", "PERMZ"], [True, False]),
)
def test_dual_cases_init_property_to_file(tmpdir, dual_cases, name, fracture):
    prop = dual_cases.get_property_from_init(name, fracture=fracture)
    prop.describe()

    if fracture:
        assert prop.name == f"{name}F"
    else:
        assert prop.name == f"{name}M"

    filename = join(tmpdir, basename(dual_cases.path) + prop.name + ".roff")
    prop.to_file(filename)
    assert os.path.exists(filename)


def test_dual_grid_poro_property(tmpdir, dual_cases):
    poro = dual_cases.get_property_from_init("PORO")

    assert poro.values[0, 0, 0] == pytest.approx(0.1)
    assert poro.values[1, 1, 0] == pytest.approx(0.16)
    assert poro.values[4, 2, 0] == pytest.approx(0.24)


def test_dual_grid_fractured_poro_property(tmpdir, dual_cases):
    poro = dual_cases.get_property_from_init("PORO", fracture=True)

    assert poro.values[0, 0, 0] == pytest.approx(0.25)
    assert poro.values[4, 2, 0] == pytest.approx(0.39)


def test_dualperm_fractured_poro_values(dual_poro_dual_perm_case):
    poro = dual_poro_dual_perm_case.get_property_from_init(name="PORO", fracture=True)
    assert poro.values[3, 0, 0] == pytest.approx(0.0)


def test_dual_case_swat_values(dual_poro_case):
    swat = dual_poro_case.get_property_from_restart("SWAT", date=20170121)
    assert swat.values[0, 0, 0] == pytest.approx(0.609244)


def test_dual_case_fractured_swat_values(dual_poro_case):
    swat = dual_poro_case.get_property_from_restart(
        "SWAT", date=20170121, fracture=True
    )
    assert swat.values[0, 0, 0] == pytest.approx(0.989687)


def test_dualperm_swat_property(dual_poro_dual_perm_case):
    swat = dual_poro_dual_perm_case.get_property_from_restart("SWAT", date=20170121)
    assert swat.values[3, 0, 0] == pytest.approx(0.5547487)


def test_dualperm_fractured_swat_property(dual_poro_dual_perm_case):
    swat = dual_poro_dual_perm_case.get_property_from_restart(
        "SWAT", date=20170121, fracture=True
    )
    assert swat.values[3, 0, 0] == pytest.approx(0.0)


def test_dualperm_wg_swat_property(dual_poro_dual_perm_wg_case):
    swat = dual_poro_dual_perm_wg_case.get_property_from_restart("SWAT", date=20170121)
    assert swat.values[3, 0, 0] == pytest.approx(0.933606)
    assert swat.values[0, 1, 0] == pytest.approx(0.0)
    assert swat.values[4, 2, 0] == pytest.approx(0.89304)


def test_dualperm_wg_fractured_swat_property(dual_poro_dual_perm_wg_case):
    swat = dual_poro_dual_perm_wg_case.get_property_from_restart(
        "SWAT", date=20170121, fracture=True
    )
    assert swat.values[3, 0, 0] == pytest.approx(0.0)
    assert swat.values[0, 1, 0] == pytest.approx(0.99818)
    assert swat.values[4, 2, 0] == pytest.approx(0.821589)


def test_dual_case_perm_property(tmpdir, dual_cases):
    perm = dual_cases.get_property_from_init("PERMX")

    assert perm.values[0, 0, 0] == pytest.approx(100.0)
    assert perm.values[3, 0, 0] == pytest.approx(100.0)
    assert perm.values[0, 1, 0] == pytest.approx(0.0)
    assert perm.values[4, 2, 0] == pytest.approx(100)


def test_dual_case_fractured_perm_property(tmpdir, dual_cases):
    perm = dual_cases.get_property_from_init("PERMX", fracture=True)

    assert perm.values[0, 0, 0] == pytest.approx(100.0)
    assert perm.values[0, 1, 0] == pytest.approx(100.0)
    assert perm.values[4, 2, 0] == pytest.approx(100)


def test_dualperm_perm_property(dual_poro_dual_perm_case):
    perm = dual_poro_dual_perm_case.get_property_from_init("PERMX", fracture=True)
    assert perm.values[3, 0, 0] == pytest.approx(0.0)


def test_dualperm_soil_property(dual_poro_dual_perm_case):
    soil = dual_poro_dual_perm_case.get_property_from_restart("SOIL", date=20170121)
    assert soil.values[3, 0, 0] == pytest.approx(0.4452512)
    assert soil.values[0, 1, 0] == pytest.approx(0.0)
    assert np.ma.is_masked(soil.values[1, 2, 0])
    assert soil.values[3, 2, 0] == pytest.approx(0.0)
    assert soil.values[4, 2, 0] == pytest.approx(0.4127138)


def test_dualperm_fractured_soil_property(dual_poro_dual_perm_case):
    soil = dual_poro_dual_perm_case.get_property_from_restart(
        "SOIL", date=20170121, fracture=True
    )
    assert soil.values[3, 0, 0] == pytest.approx(0.0)
    assert soil.values[0, 1, 0] == pytest.approx(0.01174145)
    assert soil.values[3, 2, 0] == pytest.approx(0.11676442)


def test_dualpermwg_soil_property(dual_poro_dual_perm_wg_case):
    soil = dual_poro_dual_perm_wg_case.get_property_from_restart("SOIL", date=20170121)
    assert soil.values[3, 0, 0] == pytest.approx(0.0)
    assert soil.values[0, 1, 0] == pytest.approx(0.0)


def test_dualpermwg_fractured_soil_property(dual_poro_dual_perm_wg_case):
    soil = dual_poro_dual_perm_wg_case.get_property_from_restart(
        "SOIL", date=20170121, fracture=True
    )
    assert soil.values[3, 0, 0] == pytest.approx(0.0)
    assert soil.values[0, 1, 0] == pytest.approx(0.0)


def test_dualperm_sgas_property(dual_poro_dual_perm_case):
    sgas = dual_poro_dual_perm_case.get_property_from_restart("SGAS", date=20170121)
    assert sgas.values[3, 0, 0] == pytest.approx(0.0)
    assert sgas.values[0, 1, 0] == pytest.approx(0.0)


def test_dualperm_fractured_sgas_property(dual_poro_dual_perm_case):
    sgas = dual_poro_dual_perm_case.get_property_from_restart(
        "SGAS", date=20170121, fracture=True
    )
    assert sgas.values[3, 0, 0] == pytest.approx(0.0)
    assert sgas.values[0, 1, 0] == pytest.approx(0.0)


def test_dualperm_wg_sgas_property(dual_poro_dual_perm_wg_case):
    sgas = dual_poro_dual_perm_wg_case.get_property_from_restart("SGAS", date=20170121)
    assert sgas.values[3, 0, 0] == pytest.approx(0.0663941)
    assert sgas.values[0, 1, 0] == pytest.approx(0.0)
    assert sgas.values[4, 2, 0] == pytest.approx(0.1069594)


def test_dualperm_wg_fractured_sgas_property(dual_poro_dual_perm_wg_case):
    sgas = dual_poro_dual_perm_wg_case.get_property_from_restart(
        "SGAS", date=20170121, fracture=True
    )
    assert sgas.values[3, 0, 0] == pytest.approx(0.0)
    assert sgas.values[0, 1, 0] == pytest.approx(0.00181985)
    assert sgas.values[4, 2, 0] == pytest.approx(0.178411)
