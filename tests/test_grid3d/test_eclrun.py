import itertools
from os.path import basename, exists, join

import numpy as np
import pytest

import xtgeo

from .ecl_run_fixtures import *  # noqa: F401, F403


def test_ecl_run(tmp_path, reek_run):
    """Test import an eclrun with dates and export to roff after a diff."""
    dates = [19991201, 20030101]
    rprops = ["PRESSURE", "SWAT"]

    grid = reek_run.grid_with_props(restartdates=dates, restartprops=rprops)

    # get the property object:
    pres1 = grid.get_prop_by_name("PRESSURE_20030101")
    assert pres1.values.mean() == pytest.approx(308.45, abs=0.001)

    pres1.to_file(tmp_path / "pres1.roff")

    pres2 = grid.get_prop_by_name("PRESSURE_19991201")

    pres1.values = pres1.values - pres2.values
    avg = pres1.values.mean()
    assert avg == pytest.approx(-26.073, abs=0.001)

    pres1.to_file(tmp_path / "pressurediff.roff", name="PRESSUREDIFF")


def test_ecl_run_all(ecl_runs):
    """Test import an eclrun with all dates and props."""
    gg = ecl_runs.grid_with_props(
        initprops="all",
        restartdates="all",
        restartprops="all",
    )

    assert gg.dimensions == ecl_runs.expected_dimensions
    assert len(gg.gridprops.names) == ecl_runs.expected_num_restart_properties


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
def test_dual_runs_restart_property_to_file(tmpdir, dual_runs, date, name, fracture):
    prop = dual_runs.get_property_from_restart(name, date=date, fracture=fracture)
    prop.describe()

    if fracture:
        assert prop.name == f"{name}F_{date}"
    else:
        assert prop.name == f"{name}M_{date}"

    filename = join(tmpdir, basename(dual_runs.path) + str(date) + prop.name + ".roff")
    prop.to_file(filename)

    assert exists(filename)


@pytest.mark.parametrize(
    "name, fracture",
    itertools.product(["PORO", "PERMX", "PERMY", "PERMZ"], [True, False]),
)
def test_dual_runs_init_property_to_file(tmpdir, dual_runs, name, fracture):
    prop = dual_runs.get_property_from_init(name, fracture=fracture)
    prop.describe()

    if fracture:
        assert prop.name == f"{name}F"
    else:
        assert prop.name == f"{name}M"

    filename = join(tmpdir, basename(dual_runs.path) + prop.name + ".roff")
    prop.to_file(filename)
    assert exists(filename)


def test_dual_grid_poro_property(tmpdir, dual_runs):
    poro = dual_runs.get_property_from_init("PORO")

    assert poro.values[0, 0, 0] == pytest.approx(0.1)
    assert poro.values[1, 1, 0] == pytest.approx(0.16)
    assert poro.values[4, 2, 0] == pytest.approx(0.24)


def test_dual_grid_fractured_poro_property(tmpdir, dual_runs):
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


def test_dual_run_perm_property(tmpdir, dual_runs):
    perm = dual_runs.get_property_from_init("PERMX")

    assert perm.values[0, 0, 0] == pytest.approx(100.0)
    assert perm.values[3, 0, 0] == pytest.approx(100.0)
    assert perm.values[0, 1, 0] == pytest.approx(0.0)
    assert perm.values[4, 2, 0] == pytest.approx(100)


def test_dual_run_fractured_perm_property(tmpdir, dual_runs):
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


def test_randomline_fence_from_well(show_plot, testpath, reek_run):
    """Import ROFF grid with props and make fences"""

    grd = reek_run.grid_with_props(initprops=["PORO"])
    wll = xtgeo.Well(
        join(testpath, "wells", "reek", "1", "OP_1.w"), zonelogname="Zonelog"
    )

    # get the polygon for the well, limit it to 1200
    fspec = wll.get_fence_polyline(sampling=10, nextend=2, asnumpy=False, tvdmin=1200)

    assert fspec.dataframe[fspec.dhname][4] == pytest.approx(12.6335, abs=0.001)

    fspec = wll.get_fence_polyline(sampling=10, nextend=2, asnumpy=True, tvdmin=1200)

    # get the "image", which is a 2D numpy that can be plotted with e.g. imgshow
    hmin, hmax, vmin, vmax, por = grd.get_randomline(
        fspec, "PORO", zmin=1600, zmax=1700, zincrement=1.0
    )

    if show_plot:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.imshow(por, cmap="rainbow", extent=(hmin, hmax, vmax, vmin))
        plt.axis("tight")
        plt.colorbar()
        plt.show()


def test_randomline_fence_from_polygon(show_plot, testpath, reek_run):
    """Import ROFF grid with props and make fence from polygons"""

    grd = reek_run.grid_with_props(initprops=["PORO", "PERMX"])
    fence = xtgeo.Polygons(join(testpath, "polygons", "reek", "1", "fence.pol"))

    # get the polygons
    fspec = fence.get_fence(distance=10, nextend=2, asnumpy=False)
    assert fspec.dataframe[fspec.dhname][4] == pytest.approx(10, abs=1)

    fspec = fence.get_fence(distance=5, nextend=2, asnumpy=True)

    # get the "image", which is a 2D numpy that can be plotted with e.g. imgshow
    hmin, hmax, vmin, vmax, por = grd.get_randomline(
        fspec, "PORO", zmin=1680, zmax=1750, zincrement=0.5
    )

    hmin, hmax, vmin, vmax, perm = grd.get_randomline(
        fspec, "PERMX", zmin=1680, zmax=1750, zincrement=0.5
    )

    if show_plot:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.imshow(por, cmap="rainbow", extent=(hmin, hmax, vmax, vmin))
        plt.axis("tight")
        plt.colorbar()
        plt.figure()
        plt.imshow(perm, cmap="rainbow", extent=(hmin, hmax, vmax, vmin))
        plt.axis("tight")
        plt.colorbar()
        plt.show()


def test_randomline_fence_calczminzmax(testpath, reek_run):
    """Import ROFF grid with props and make fence from polygons, zmin/zmax auto"""

    grd = reek_run.grid_with_props(initprops=["PORO", "PERMX"])
    fence = xtgeo.Polygons(join(testpath, "polygons/reek/1/fence.pol"))

    fspec = fence.get_fence(distance=5, nextend=2, asnumpy=True)

    hmin, hmax, vmin, vmax, por = grd.get_randomline(
        fspec, "PORO", zmin=None, zmax=None
    )
    assert vmin == pytest.approx(1548.10098, abs=0.0001)


def test_reverse_row_axis_dualprops(dual_props_run):
    """Reverse axis for distorted but small grid with props"""

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


def test_avg02(tmpdir, generate_plot, reek_run, testpath):
    """Make average map from Reek Eclipse."""

    # get the poro
    po = reek_run.get_property_from_init(name="PORO")

    # get the dz and the coordinates
    dz = reek_run.grid.get_dz(asmasked=False)
    xc, yc, _zc = reek_run.grid.get_xyz(asmasked=False)

    # get actnum
    actnum = reek_run.grid.get_actnum()

    # convert from masked numpy to ordinary
    xcuse = np.copy(xc.values3d)
    ycuse = np.copy(yc.values3d)
    dzuse = np.copy(dz.values3d)
    pouse = np.copy(po.values3d)

    # dz must be zero for undef cells
    dzuse[actnum.values3d < 0.5] = 0.0
    pouse[actnum.values3d < 0.5] = 0.0

    # make a map... estimate from xc and yc
    zuse = np.ones((xcuse.shape))

    avgmap = xtgeo.RegularSurface(
        ncol=200,
        nrow=250,
        xinc=50,
        yinc=50,
        xori=457000,
        yori=5927000,
        values=np.zeros((200, 250)),
    )

    avgmap.avg_from_3dprop(
        xprop=xcuse,
        yprop=ycuse,
        zoneprop=zuse,
        zone_minmax=(1, 1),
        mprop=pouse,
        dzprop=dzuse,
        truncate_le=None,
    )

    # add the faults in plot
    fau = xtgeo.Polygons(
        join(testpath, "polygons/reek/1/top_upper_reek_faultpoly.zmap"),
        fformat="zmap",
    )
    fspec = {"faults": fau}

    if generate_plot:
        avgmap.quickplot(
            filename=join(tmpdir, "tmp_poro2.png"), xlabelrotation=30, faults=fspec
        )
        avgmap.to_file(join(tmpdir, "tmp.poro.gri"), fformat="irap_ascii")

    assert avgmap.values.mean() == pytest.approx(0.1653, abs=0.01)


def test_avg03(tmpdir, generate_plot, reek_run, testpath):
    """Make average map from Reek Eclipse, speed up by zone_avgrd."""
    # get the poro
    po = reek_run.get_property_from_init(name="PORO")

    # get the dz and the coordinates
    dz = reek_run.grid.get_dz(asmasked=False)
    xc, yc, _zc = reek_run.grid.get_xyz(asmasked=False)

    # get actnum
    actnum = reek_run.grid.get_actnum()
    actnum = actnum.get_npvalues3d()

    # convert from masked numpy to ordinary
    xcuse = xc.get_npvalues3d()
    ycuse = yc.get_npvalues3d()
    dzuse = dz.get_npvalues3d(fill_value=0.0)
    pouse = po.get_npvalues3d(fill_value=0.0)

    # dz must be zero for undef cells
    dzuse[actnum < 0.5] = 0.0
    pouse[actnum < 0.5] = 0.0

    # make a map... estimate from xc and yc
    zuse = np.ones((xcuse.shape))

    avgmap = xtgeo.RegularSurface(
        ncol=200,
        nrow=250,
        xinc=50,
        yinc=50,
        xori=457000,
        yori=5927000,
        values=np.zeros((200, 250)),
    )

    avgmap.avg_from_3dprop(
        xprop=xcuse,
        yprop=ycuse,
        zoneprop=zuse,
        zone_minmax=(1, 1),
        mprop=pouse,
        dzprop=dzuse,
        truncate_le=None,
        zone_avg=True,
    )

    # add the faults in plot
    fau = xtgeo.Polygons(
        join(testpath, "polygons/reek/1/top_upper_reek_faultpoly.zmap"),
        fformat="zmap",
    )
    fspec = {"faults": fau}

    if generate_plot:
        avgmap.quickplot(
            filename=join(tmpdir, "tmp_poro3.png"), xlabelrotation=30, faults=fspec
        )
    avgmap.to_file(join(tmpdir, "tmp.poro3.gri"), fformat="irap_ascii")

    assert avgmap.values.mean() == pytest.approx(0.1653, abs=0.01)


def test_get_xy_values_for_webportal_ecl(reek_run):
    """Get lists on webportal format (Eclipse input)"""

    grid = reek_run.grid
    prop = reek_run.get_property_from_init(name="PORO")

    coord, _valuelist = prop.get_xy_value_lists(grid=grid)
    assert coord[0][0][0][1] == pytest.approx(5935688.22412, abs=0.001)


def test_ecl_run_dimensions(ecl_runs):
    assert ecl_runs.grid.dimensions == ecl_runs.expected_dimensions


def test_eclinit_import_reek(reek_run):
    """Property import from Eclipse. Reek"""

    po = reek_run.get_property_from_init(name="PORO")
    assert po.values.mean() == pytest.approx(0.1677, abs=0.0001)

    pv = reek_run.get_property_from_init(name="PORV")
    assert pv.values.mean() == pytest.approx(13536.2137, abs=0.0001)


def test_eclinit_simple_importexport(tmpdir, ecl_runs):
    """Property import and export with anoother name"""

    gg = ecl_runs.grid
    po = ecl_runs.get_property_from_init(name="PORO")

    po.to_file(
        join(tmpdir, "simple.grdecl"),
        fformat="grdecl",
        name="PORO2",
        fmt="%12.5f",
    )

    p2 = xtgeo.gridproperty_from_file(
        join(tmpdir, "simple.grdecl"), grid=gg, name="PORO2"
    )
    assert p2.name == "PORO2"


def test_eclunrst_import_reek(reek_run):
    """Property UNRST import from Eclipse. Reek"""

    press = reek_run.get_property_from_restart(name="PRESSURE", date=19991201)
    assert press.values.mean() == pytest.approx(334.5232, abs=0.0001)

    swat = reek_run.get_property_from_restart(name="SWAT", date=19991201)
    assert swat.values.mean() == pytest.approx(0.8780, abs=0.001)

    sgas = reek_run.get_property_from_restart(name="SGAS", date=19991201)
    np.testing.assert_allclose(sgas.values, 0.0)

    soil = reek_run.get_property_from_restart(name="SOIL", date=19991201)
    np.testing.assert_allclose(soil.values, 1 - swat.values)


def test_io_ecl2roff_discrete(tmpdir, reek_run):
    """Import Eclipse discrete property; then export to ROFF int."""

    po = reek_run.get_property_from_init("SATNUM")

    assert po.codes == {1: "1"}
    assert po.ncodes == 1
    assert isinstance(po.codes[1], str)

    po.to_file(join(tmpdir, "ecl2roff_disc.roff"), name="SATNUM", fformat="roff")


def test_io_ecl_dates(reek_run):
    """Import Eclipse with some more flexible dates settings"""
    po = reek_run.get_property_from_restart("PRESSURE", date="first")
    assert po.date == 19991201
    px = reek_run.get_property_from_restart("PRESSURE", date="last")
    assert px.date == 20030101


def test_import_reek_init(reek_run):
    gps = reek_run.get_init_properties(names=["PORO", "PORV"])

    # get the object
    poro = gps.get_prop_by_name("PORO")
    porv = gps.get_prop_by_name("PORV")
    assert poro.values.mean() == pytest.approx(0.1677402, abs=0.00001)
    assert porv.values.mean() == pytest.approx(13536.2137, abs=0.0001)


def test_import_should_fail(reek_run):
    """Import INIT and UNRST Reek but ask for wrong name or date"""

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


def test_import_should_pass(reek_run):
    """Import INIT and UNRST but ask for wrong name or date , using strict=False"""

    reek_gps = reek_run.get_restart_properties(
        names=["PRESSURE", "DUMMY"],
        dates=[19991201, 19991212],
        strict=(False, False),
    )

    assert "PRESSURE_19991201" in reek_gps
    assert "PRESSURE_19991212" not in reek_gps
    assert "DUMMY_19991201" not in reek_gps


def test_import_restart(reek_run):
    """Import Restart"""

    gps = reek_run.get_restart_properties(
        names=["PRESSURE", "SWAT"], dates=[19991201, 20010101]
    )

    assert gps["PRESSURE_19991201"].values.mean() == pytest.approx(334.52327, abs=0.001)
    assert gps["SWAT_19991201"].values.mean() == pytest.approx(0.87, abs=0.01)
    assert gps["PRESSURE_20010101"].values.mean() == pytest.approx(304.897, abs=0.01)
