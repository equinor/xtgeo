import itertools
from os.path import basename, join

import numpy as np
import pytest

import xtgeo

from .eclrun_fixtures import *  # noqa: F401, F403


def test_ecl_run_all(ecl_runs):
    gg = ecl_runs.grid_with_props(
        initprops="all",
        restartdates="all",
        restartprops="all",
    )

    assert gg.dimensions == ecl_runs.expected_dimensions

    for name in ecl_runs.expected_init_props:
        if gg.dualporo:
            assert name + "M" in gg.gridprops
        else:
            assert name in gg.gridprops

    for name, date in itertools.product(
        ecl_runs.expected_restart_props, ecl_runs.expected_dates
    ):
        if gg.dualporo:
            assert f"{name}M_{date}" in gg.gridprops
        else:
            assert f"{name}_{date}" in gg.gridprops


def test_import_saturations(ecl_runs):
    gps = ecl_runs.get_restart_properties(names=["SGAS", "SWAT", "SOIL"], dates="all")

    if not ecl_runs.grid.dualporo:
        for d in gps.dates:
            np.testing.assert_allclose(
                gps["SGAS_" + str(d)].values
                + gps["SWAT_" + str(d)].values
                + gps["SOIL_" + str(d)].values,
                1.0,
            )


@pytest.mark.parametrize("fformat", ["grdecl", "roff", "bgrdecl"])
def test_roundtrip_properties(fformat, tmp_path, ecl_runs):
    for p in ecl_runs.expected_restart_props:
        prop = ecl_runs.get_property_from_restart(p, date="last")
        prop.to_file(tmp_path / f"{p}.{fformat}", name=p, fformat=fformat)

        prop2 = xtgeo.gridproperty_from_file(
            tmp_path / f"{p}.{fformat}",
            name=p,
            fformat=fformat,
            grid=ecl_runs.grid,
        )

        np.testing.assert_allclose(prop.values, prop2.values, atol=5e-3)


@pytest.mark.parametrize("fformat", ["grdecl", "roff", "bgrdecl"])
def test_roundtrip_satnum(fformat, tmp_path, ecl_runs):
    prop = ecl_runs.get_property_from_init("SATNUM")
    prop.to_file(tmp_path / f"satnum.{fformat}", name="SATNUM", fformat=fformat)

    prop2 = xtgeo.gridproperty_from_file(
        tmp_path / f"satnum.{fformat}",
        name="SATNUM",
        fformat=fformat,
        grid=ecl_runs.grid,
    )

    assert np.array_equal(prop.values.filled(999), prop2.values.filled(999))
    assert prop.codes == {r: str(r) for r in ecl_runs.region_numbers}


def test_first_and_last_dates(ecl_runs):
    po = ecl_runs.get_property_from_restart("PRESSURE", date="first")
    assert po.date == min(ecl_runs.expected_dates)

    px = ecl_runs.get_property_from_restart("PRESSURE", date="last")
    assert px.date == max(ecl_runs.expected_dates)


def test_dual_runs_general_grid(tmpdir, dual_runs):
    assert dual_runs.grid.dimensions == dual_runs.expected_dimensions
    assert dual_runs.grid.dualporo is True
    assert dual_runs.grid.dualperm is dual_runs.expected_perm

    dual_runs.grid.to_file(join(tmpdir, basename(dual_runs.path)) + ".roff")
    dual_runs.grid._dualactnum.to_file(
        join(tmpdir, basename(dual_runs.path) + "dualact.roff")
    )


@pytest.mark.parametrize(
    "date, name, fracture",
    itertools.product([20170121, 20170131], ["SGAS", "SOIL", "SWAT"], [True, False]),
)
def test_dual_runs_restart_property_to_file(dual_runs, date, name, fracture):
    prop = dual_runs.get_property_from_restart(name, date=date, fracture=fracture)

    if fracture:
        assert prop.name == f"{name}F_{date}"
    else:
        assert prop.name == f"{name}M_{date}"


@pytest.mark.parametrize(
    "name, fracture",
    itertools.product(["PORO", "PERMX", "PERMY", "PERMZ"], [True, False]),
)
def test_dual_runs_init_property_to_file(dual_runs, name, fracture):
    prop = dual_runs.get_property_from_init(name, fracture=fracture)

    if fracture:
        assert prop.name == f"{name}F"
    else:
        assert prop.name == f"{name}M"


def test_import_reek_init(reek_run):
    gps = reek_run.get_init_properties(names=["PORO", "PORV"])

    # get the object
    poro = gps.get_prop_by_name("PORO")
    porv = gps.get_prop_by_name("PORV")
    assert poro.values.mean() == pytest.approx(0.1677402, abs=0.00001)
    assert porv.values.mean() == pytest.approx(13536.2137, abs=0.0001)


def test_dual_grid_poro_property(dual_runs):
    poro = dual_runs.get_property_from_init("PORO")

    assert poro.values[0, 0, 0] == pytest.approx(0.1)
    assert poro.values[1, 1, 0] == pytest.approx(0.16)
    assert poro.values[4, 2, 0] == pytest.approx(0.24)


def test_dual_grid_fractured_poro_property(dual_runs):
    poro = dual_runs.get_property_from_init("PORO", fracture=True)

    assert poro.values[0, 0, 0] == pytest.approx(0.25)
    assert poro.values[4, 2, 0] == pytest.approx(0.39)


def test_dualperm_fractured_poro_values(dual_poro_dual_perm_run):
    poro = dual_poro_dual_perm_run.get_property_from_init(name="PORO", fracture=True)
    assert poro.values[3, 0, 0] == pytest.approx(0.0)


def test_dual_run_swat_values(dual_poro_run):
    swat = dual_poro_run.get_property_from_restart("SWAT", date=20170121)
    assert swat.values[0, 0, 0] == pytest.approx(0.609244)


def test_dual_run_fractured_swat_values(dual_poro_run):
    swat = dual_poro_run.get_property_from_restart("SWAT", date=20170121, fracture=True)
    assert swat.values[0, 0, 0] == pytest.approx(0.989687)


def test_dualperm_swat_property(dual_poro_dual_perm_run):
    swat = dual_poro_dual_perm_run.get_property_from_restart("SWAT", date=20170121)
    assert swat.values[3, 0, 0] == pytest.approx(0.5547487)


def test_dualperm_fractured_swat_property(dual_poro_dual_perm_run):
    swat = dual_poro_dual_perm_run.get_property_from_restart(
        "SWAT", date=20170121, fracture=True
    )
    assert swat.values[3, 0, 0] == pytest.approx(0.0)


def test_dualperm_wg_swat_property(dual_poro_dual_perm_wg_run):
    swat = dual_poro_dual_perm_wg_run.get_property_from_restart("SWAT", date=20170121)
    assert swat.values[3, 0, 0] == pytest.approx(0.933606)
    assert swat.values[0, 1, 0] == pytest.approx(0.0)
    assert swat.values[4, 2, 0] == pytest.approx(0.89304)


def test_dualperm_wg_fractured_swat_property(dual_poro_dual_perm_wg_run):
    swat = dual_poro_dual_perm_wg_run.get_property_from_restart(
        "SWAT", date=20170121, fracture=True
    )
    assert swat.values[3, 0, 0] == pytest.approx(0.0)
    assert swat.values[0, 1, 0] == pytest.approx(0.99818)
    assert swat.values[4, 2, 0] == pytest.approx(0.821589)


def test_dual_run_perm_property(dual_runs):
    perm = dual_runs.get_property_from_init("PERMX")

    assert perm.values[0, 0, 0] == pytest.approx(100.0)
    assert perm.values[3, 0, 0] == pytest.approx(100.0)
    assert perm.values[0, 1, 0] == pytest.approx(0.0)
    assert perm.values[4, 2, 0] == pytest.approx(100)


def test_dual_run_fractured_perm_property(dual_runs):
    perm = dual_runs.get_property_from_init("PERMX", fracture=True)

    assert perm.values[0, 0, 0] == pytest.approx(100.0)
    assert perm.values[0, 1, 0] == pytest.approx(100.0)
    assert perm.values[4, 2, 0] == pytest.approx(100)


def test_dualperm_perm_property(dual_poro_dual_perm_run):
    perm = dual_poro_dual_perm_run.get_property_from_init("PERMX", fracture=True)
    assert perm.values[3, 0, 0] == pytest.approx(0.0)


def test_dualperm_soil_property(dual_poro_dual_perm_run):
    soil = dual_poro_dual_perm_run.get_property_from_restart("SOIL", date=20170121)
    assert soil.values[3, 0, 0] == pytest.approx(0.4452512)
    assert soil.values[0, 1, 0] == pytest.approx(0.0)
    assert np.ma.is_masked(soil.values[1, 2, 0])
    assert soil.values[3, 2, 0] == pytest.approx(0.0)
    assert soil.values[4, 2, 0] == pytest.approx(0.4127138)


def test_dualperm_fractured_soil_property(dual_poro_dual_perm_run):
    soil = dual_poro_dual_perm_run.get_property_from_restart(
        "SOIL", date=20170121, fracture=True
    )
    assert soil.values[3, 0, 0] == pytest.approx(0.0)
    assert soil.values[0, 1, 0] == pytest.approx(0.01174145)
    assert soil.values[3, 2, 0] == pytest.approx(0.11676442)


def test_dualpermwg_soil_property(dual_poro_dual_perm_wg_run):
    soil = dual_poro_dual_perm_wg_run.get_property_from_restart("SOIL", date=20170121)
    assert soil.values[3, 0, 0] == pytest.approx(0.0)
    assert soil.values[0, 1, 0] == pytest.approx(0.0)


def test_dualpermwg_fractured_soil_property(dual_poro_dual_perm_wg_run):
    soil = dual_poro_dual_perm_wg_run.get_property_from_restart(
        "SOIL", date=20170121, fracture=True
    )
    assert soil.values[3, 0, 0] == pytest.approx(0.0)
    assert soil.values[0, 1, 0] == pytest.approx(0.0)


def test_dualperm_sgas_property(dual_poro_dual_perm_run):
    sgas = dual_poro_dual_perm_run.get_property_from_restart("SGAS", date=20170121)
    assert sgas.values[3, 0, 0] == pytest.approx(0.0)
    assert sgas.values[0, 1, 0] == pytest.approx(0.0)


def test_dualperm_fractured_sgas_property(dual_poro_dual_perm_run):
    sgas = dual_poro_dual_perm_run.get_property_from_restart(
        "SGAS", date=20170121, fracture=True
    )
    assert sgas.values[3, 0, 0] == pytest.approx(0.0)
    assert sgas.values[0, 1, 0] == pytest.approx(0.0)


def test_dualperm_wg_sgas_property(dual_poro_dual_perm_wg_run):
    sgas = dual_poro_dual_perm_wg_run.get_property_from_restart("SGAS", date=20170121)
    assert sgas.values[3, 0, 0] == pytest.approx(0.0663941)
    assert sgas.values[0, 1, 0] == pytest.approx(0.0)
    assert sgas.values[4, 2, 0] == pytest.approx(0.1069594)


def test_dualperm_wg_fractured_sgas_property(dual_poro_dual_perm_wg_run):
    sgas = dual_poro_dual_perm_wg_run.get_property_from_restart(
        "SGAS", date=20170121, fracture=True
    )
    assert sgas.values[3, 0, 0] == pytest.approx(0.0)
    assert sgas.values[0, 1, 0] == pytest.approx(0.00181985)
    assert sgas.values[4, 2, 0] == pytest.approx(0.178411)


def test_reverse_row_axis_dualprops(dual_props_run):
    grd = dual_props_run.grid_with_props(initprops=["PORO", "PORV"])

    poro = grd.gridprops.props[0]
    porowas = poro.copy()
    assert poro.values[1, 0, 0] == pytest.approx(0.17777, abs=0.01)
    assert grd.ijk_handedness == "left"

    grd.reverse_row_axis()
    assert poro.values[1, 2, 0] == pytest.approx(0.17777, abs=0.01)
    assert poro.values[1, 2, 0] == porowas.values[1, 0, 0]

    grd.reverse_row_axis()
    assert poro.values[1, 0, 0] == porowas.values[1, 0, 0]
    assert grd.ijk_handedness == "left"

    grd.reverse_row_axis(ijk_handedness="left")  # ie do nothing in this case
    assert poro.values[1, 0, 0] == porowas.values[1, 0, 0]
    assert grd.ijk_handedness == "left"


def test_get_xy_values_(reek_run):
    grid = reek_run.grid
    prop = reek_run.get_property_from_init(name="PORO")

    coord, _valuelist = prop.get_xy_value_lists(grid=grid)
    assert coord[0][0][0][1] == pytest.approx(5935688.22412, abs=0.001)


def test_ecl_run_dimensions(ecl_runs):
    assert ecl_runs.grid.dimensions == ecl_runs.expected_dimensions


def test_eclinit_import_reek(reek_run):
    po = reek_run.get_property_from_init(name="PORO")
    assert po.values.mean() == pytest.approx(0.1677, abs=0.0001)

    pv = reek_run.get_property_from_init(name="PORV")
    assert pv.values.mean() == pytest.approx(13536.2137, abs=0.0001)


@pytest.mark.parametrize("date", [19991201, "19991201", "1999-12-01"])
def test_eclunrst_import_reek(reek_run, date):
    press = reek_run.get_property_from_restart(name="PRESSURE", date=date)
    assert press.values.mean() == pytest.approx(334.5232, abs=0.0001)

    swat = reek_run.get_property_from_restart(name="SWAT", date=date)
    assert swat.values.mean() == pytest.approx(0.8780, abs=0.001)

    sgas = reek_run.get_property_from_restart(name="SGAS", date=date)
    np.testing.assert_allclose(sgas.values, 0.0)

    soil = reek_run.get_property_from_restart(name="SOIL", date=date)
    np.testing.assert_allclose(soil.values, 1 - swat.values)


@pytest.mark.parametrize(
    "dates",
    [[19991201, 20010101, 20030101], ["19991201", "20010101", "20030101"], "all"],
)
def test_import_reek_restart(dates, reek_run):
    gps = reek_run.get_restart_properties(names=["PRESSURE", "SWAT"], dates=dates)

    assert gps["PRESSURE_19991201"].values.mean() == pytest.approx(334.52327, abs=0.001)
    assert gps["SWAT_19991201"].values.mean() == pytest.approx(0.87, abs=0.01)
    assert gps["PRESSURE_20010101"].values.mean() == pytest.approx(304.897, abs=0.01)
    assert gps["PRESSURE_20030101"].values.mean() == pytest.approx(308.45, abs=0.001)


def test_import_should_fail(reek_run):
    with pytest.raises(ValueError, match="Requested keyword NOSUCHNAME"):
        reek_run.get_init_properties(names=["PORO", "NOSUCHNAME"])

    reek_run.get_restart_properties(
        names=["PRESSURE"],
        dates=[19991201, 19991212],
        strict=(True, False),
    )

    with pytest.raises(ValueError, match="PRESSURE 19991212 is not in"):
        reek_run.get_restart_properties(
            names=["PRESSURE"],
            dates=[19991201, 19991212],
            strict=(True, True),
        )


@pytest.mark.parametrize(
    "dates", [[19991201, 20010101], ["19991201", "20010101"], "all"]
)
def test_import_restart_namestyle(dates, reek_run):
    gps = reek_run.get_restart_properties(
        names=["PRESSURE", "SWAT"], dates=dates, namestyle=1
    )
    assert gps["PRESSURE--1999_12_01"].values.mean() == pytest.approx(
        334.52327, abs=0.001
    )
    assert gps["SWAT--1999_12_01"].values.mean() == pytest.approx(0.87, abs=0.01)
    assert gps["PRESSURE--2001_01_01"].values.mean() == pytest.approx(304.897, abs=0.01)

    assert "PRESSURE--1999_12_01" in gps
    assert "PRESSURE--1999_12_12" not in gps
    assert "SWAT--1999_12_01" in gps
    assert gps.names == [p.name for p in gps]
