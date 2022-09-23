# -*- coding: utf-8 -*-
"""Integration tests towards Roxar API, requires RoxarAPI license.

Creates a tmp RMS project in given version which is used as fixture for all other Roxar
API dependent tests.

Then run tests in Roxar API which focus on IO.

This requires a ROXAPI license, and to be ran in a "roxenvbash" environment if Equinor.
"""
from collections import OrderedDict
from os.path import join

import numpy as np
import pytest
import xtgeo

try:
    import _roxar
    import roxar
except ImportError:
    pass

xtg = xtgeo.common.XTGeoDialog()
logger = xtg.basiclogger(__name__)

if not xtg.testsetup():
    raise SystemExit

TPATH = xtg.testpathobj

PROJNAME = "tmp_project.rmsxxx"

CUBEDATA1 = TPATH / "cubes/reek/syntseis_20000101_seismic_depth_stack.segy"
CUBENAME1 = "synth1"

SURFTOPS1 = [
    TPATH / "surfaces/reek/1/topreek_rota.gri",
    TPATH / "surfaces/reek/1/midreek_rota.gri",
    TPATH / "surfaces/reek/1/lowreek_rota.gri",
]


SURFCAT1 = "DS_whatever"
SURFNAMES1 = ["TopReek", "MidReek", "BaseReek"]

GRIDDATA1 = TPATH / "3dgrids/reek/reek_sim_grid.roff"
PORODATA1 = TPATH / "3dgrids/reek/reek_sim_poro.roff"
ZONEDATA1 = TPATH / "3dgrids/reek/reek_sim_zone.roff"

GRIDNAME1 = "Simgrid"
PORONAME1 = "PORO"
ZONENAME1 = "Zone"

WELLSFOLDER1 = TPATH / "wells/reek/1"
WELLS1 = ["OP1_perf.w", "OP_2.w", "OP_6.w", "XP_with_repeat.w"]

POLYDATA1 = TPATH / "polygons/reek/1/polset2.pol"
POINTSDATA1 = TPATH / "points/reek/1/pointset3.poi"
POINTSDATA2 = TPATH / "points/reek/1/poi_attr.rmsattr"
POLYNAME1 = "Polys"
POINTSNAME1 = "Points"
POINTSNAME2 = "PointsAttrs"


@pytest.fixture(name="tmpdir", scope="module")
def fixture_tmpdir(tmpdir_factory):
    """In order to make tmpdir in a module scope."""
    return tmpdir_factory.mktemp("data")


@pytest.fixture(name="roxinstance", scope="module")
def fixture_roxinstance():
    """Create roxinstance in module scope."""
    project = roxar.Project.create()
    return xtgeo.RoxUtils(project)


@pytest.mark.requires_roxar
@pytest.fixture(name="roxar_project", scope="module")
def fixture_create_project(tmpdir, roxinstance) -> str:
    """Create a tmp RMS project for testing, populate with basic data.

    Returns a path (as str) to project for subsequent jobs.
    """
    tmp_project_path = join(tmpdir, PROJNAME)
    project = roxinstance.project

    logger.info("Roxar version is %s", roxinstance.roxversion)
    logger.info("RMS version is %s", roxinstance.rmsversion(roxinstance.roxversion))
    assert "1." in roxinstance.roxversion

    for wfile in WELLS1:
        wobj = xtgeo.well_from_file(WELLSFOLDER1 / wfile)
        if "XP_with" in wfile:
            wobj.name = "OP2_w_repeat"

        wobj.to_roxar(project, wobj.name, logrun="log", trajectory="My trajectory")

    # populate with cube data
    cube = xtgeo.cube_from_file(CUBEDATA1)
    cube.to_roxar(project, CUBENAME1, domain="depth")

    # populate with surface data
    roxinstance.create_horizons_category(SURFCAT1)
    for num, name in enumerate(SURFNAMES1):
        srf = xtgeo.surface_from_file(SURFTOPS1[num])
        project.horizons.create(name, roxar.HorizonType.interpreted)
        srf.to_roxar(project, name, SURFCAT1)

    # populate with grid and props
    grd = xtgeo.grid_from_file(GRIDDATA1)
    grd.to_roxar(project, GRIDNAME1)
    por = xtgeo.gridproperty_from_file(PORODATA1, name=PORONAME1)
    por.to_roxar(project, GRIDNAME1, PORONAME1)
    zon = xtgeo.gridproperty_from_file(ZONEDATA1, name=ZONENAME1)
    zon.to_roxar(project, GRIDNAME1, ZONENAME1)

    # populate with points and polygons (XYZ data)
    poly = xtgeo.polygons_from_file(POLYDATA1)
    poly.to_roxar(project, POLYNAME1, "", stype="clipboard")

    poi = xtgeo.points_from_file(POINTSDATA1)
    poi.to_roxar(project, POINTSNAME1, "", stype="clipboard")
    logger.info("Initialised RMS project, done!")

    poi = xtgeo.points_from_file(POINTSDATA2, fformat="rms_attr")
    poi.to_roxar(project, POINTSNAME2, "", stype="clipboard", attributes=True)
    logger.info("Initialised RMS project, done!")

    project.save_as(tmp_project_path)
    project.close()
    logger.info("Close initial project and return path")

    return tmp_project_path


@pytest.mark.requires_roxar
def test_rox_getset_cube(roxar_project):
    """Get a cube from a RMS project, do some stuff and store/save."""
    cube = xtgeo.cube_from_roxar(roxar_project, CUBENAME1)
    assert cube.values.mean() == pytest.approx(0.000718, abs=0.001)
    cube.values += 100
    assert cube.values.mean() == pytest.approx(100.000718, abs=0.001)
    cube.to_roxar(roxar_project, CUBENAME1 + "_copy1")
    cube.to_roxar(roxar_project, CUBENAME1 + "_copy2", folder="somefolder")


@pytest.mark.requires_roxar
def test_rox_surfaces(roxar_project):
    """Various get set on surfaces in RMS."""
    srf = xtgeo.surface_from_roxar(roxar_project, "TopReek", SURFCAT1)
    srf2 = xtgeo.surface_from_roxar(roxar_project, "MidReek", SURFCAT1)
    assert srf.ncol == 554
    assert srf.values.mean() == pytest.approx(1698.648, abs=0.01)

    srf.to_roxar(roxar_project, "TopReek_copy", "SomeFolder", stype="clipboard")

    # open project and do save explicit
    rox = xtgeo.RoxUtils(roxar_project)
    prj = rox.project
    iso = srf2 - srf
    rox.create_zones_category("IS_isochore")
    prj.zones.create("UpperReek", prj.horizons["TopReek"], prj.horizons["MidReek"])
    iso.to_roxar(prj, "UpperReek", "IS_isochore", stype="zones")

    iso2 = xtgeo.surface_from_roxar(prj, "UpperReek", "IS_isochore", stype="zones")
    assert iso2.values.mean() == pytest.approx(20.79, abs=0.01)

    prj.save()
    prj.close()


@pytest.mark.requires_roxar
def test_rox_surfaces_alternative_open(roxar_project):
    """Based on previous but instead use a project ref as first argument"""

    rox = xtgeo.RoxUtils(roxar_project)

    assert isinstance(rox.project, _roxar.Project)

    srf = xtgeo.surface_from_roxar(rox.project, "TopReek", SURFCAT1)
    assert srf.ncol == 554
    assert srf.values.mean() == pytest.approx(1698.648, abs=0.01)

    rox.project.close()


@pytest.mark.requires_roxar
def test_rox_surfaces_clipboard_general2d_data(roxar_project, roxinstance):
    """Set and get surfaces on clipboard and general2D data"""

    rox = xtgeo.RoxUtils(roxar_project)
    project = rox.project

    surf = xtgeo.surface_from_roxar(project, "TopReek", SURFCAT1)

    surf.to_roxar(project, "mycase", "myfolder", stype="clipboard")
    surf2 = xtgeo.surface_from_roxar(project, "mycase", "myfolder", stype="clipboard")
    assert surf2.values.mean() == surf.values.mean()

    # general 2D data (from xtgeo version 2.19 and roxar API >= 1.6)
    if not roxinstance.version_required("1.6"):
        with pytest.raises(
            NotImplementedError, match=r"API Support for general2d_data is missing"
        ):
            surf.to_roxar(project, "mycase", "myfolder", stype="general2d_data")
        logger.info("This version of RMS does not support this feature")

    else:
        surf.to_roxar(project, "mycase", "myfolder", stype="general2d_data")
        surf2 = xtgeo.surface_from_roxar(
            project, "mycase", "myfolder", stype="general2d_data"
        )
        assert surf2.values.tolist() == surf.values.tolist()

    rox.safe_close()


@pytest.mark.requires_roxar
def test_rox_get_set_trend_surfaces(roxar_project):
    """Get, modify and set trendsurfaces from a RMS project.

    Since the current RMS API does not support write to trends.surfaces, an automatic
    test cannot be made here. The functions is tested manually in RMS
    """
    surf = xtgeo.surface_from_roxar(roxar_project, "TopReek", SURFCAT1)

    with pytest.raises(ValueError, match=r"Any is not within Trends"):
        surf.to_roxar(roxar_project, "Any", None, stype="trends")

    with pytest.raises(ValueError, match=r"Any is not within Trends"):
        surf = xtgeo.surface_from_roxar(roxar_project, "Any", None, stype="trends")


@pytest.mark.requires_roxar
def test_rox_wells(roxar_project):
    """Various tests on Roxar wells."""
    well = xtgeo.well_from_roxar(
        roxar_project, "OP_2", trajectory="My trajectory", logrun="log"
    )
    assert "Zonelog" in well.lognames

    assert well.dataframe["Poro"].mean() == pytest.approx(0.1637623936)


@pytest.mark.requires_roxar
def test_rox_get_gridproperty(roxar_project):
    """Get a grid property from a RMS project."""
    logger.info("Project is %s", roxar_project)

    poro = xtgeo.gridproperty_from_roxar(roxar_project, GRIDNAME1, PORONAME1)

    assert poro.values.mean() == pytest.approx(0.16774, abs=0.001)
    assert poro.dimensions == (40, 64, 14)

    zone = xtgeo.gridproperty_from_roxar(roxar_project, GRIDNAME1, ZONENAME1)
    assert "int" in str(zone.values.dtype)

    zone._roxar_dtype = np.int32
    with pytest.raises(TypeError):
        zone.to_roxar(roxar_project, GRIDNAME1, ZONENAME1)


@pytest.mark.requires_roxar
def test_rox_get_modify_set_gridproperty(roxar_project):
    """Get and set a grid property from a RMS project."""
    poro = xtgeo.gridproperty_from_roxar(roxar_project, GRIDNAME1, PORONAME1)

    adder = 0.9
    poro.values = poro.values + adder

    poro.to_roxar(roxar_project, GRIDNAME1, PORONAME1 + "_NEW")

    poro.from_roxar(roxar_project, GRIDNAME1, PORONAME1 + "_NEW")
    assert poro.values[1, 0, 0] == pytest.approx(0.14942 + adder, abs=0.0001)


@pytest.mark.requires_roxar
def test_rox_get_modify_set_grid(roxar_project):
    """Get, modify and set a grid from a RMS project."""
    grd = xtgeo.grid_from_roxar(roxar_project, GRIDNAME1)
    grd1 = grd.copy()

    grd.translate_coordinates(translate=(200, 3000, 300))

    grd.to_roxar(roxar_project, GRIDNAME1 + "_edit1")

    grd2 = xtgeo.grid_from_roxar(roxar_project, GRIDNAME1 + "_edit1")

    assert grd2.dimensions == grd1.dimensions


@pytest.mark.requires_roxar
def test_rox_get_modify_set_get_grid_with_subzones(roxar_project, roxinstance):
    """Get, modify and set + get a grid from a RMS project using subzones/subgrids."""

    grd = xtgeo.grid_from_roxar(roxar_project, GRIDNAME1)

    zonation = OrderedDict()
    zonation["intva"] = 4
    zonation["intvb"] = 7
    zonation["intvc"] = 3
    grd.set_subgrids(zonation)

    if not roxinstance.version_required("1.6"):
        with pytest.warns(UserWarning, match=r"Implementation of subgrids is lacking"):
            grd.to_roxar(roxar_project, "NewGrid")
    else:
        grd.to_roxar(roxar_project, "NewGrid")

        # get a new instance for recent storage (subgrids should now be present)
        grd1 = xtgeo.grid_from_roxar(roxar_project, "NewGrid")

        for intv in ["intva", "intvb", "intvc"]:
            assert list(grd.subgrids[intv]) == list(grd1.subgrids[intv])


@pytest.mark.requires_roxar
def test_rox_get_modify_set_polygons(roxar_project, roxinstance):
    """Get, modify and set a polygons from a RMS project."""
    poly = xtgeo.polygons_from_roxar(roxar_project, POLYNAME1, "", stype="clipboard")
    assert poly.dataframe.iloc[-1, 2] == pytest.approx(1595.161377)
    assert poly.dataframe.shape[0] == 25
    assert poly.dataframe.shape[1] == 4

    poly.rescale(300)
    # store in RMS
    poly.to_roxar(roxar_project, "RESCALED", "", stype="clipboard")
    assert poly.dataframe.shape[0] == 127

    # store and retrieve in general2d_data just to see that it works
    if roxinstance.version_required("1.6"):
        poly.to_roxar(roxar_project, "xxx", "folder/sub", stype="general2d_data")
        poly2 = xtgeo.polygons_from_roxar(
            roxar_project, "xxx", "folder/sub", stype="general2d_data"
        )
        assert poly.dataframe[poly.xname].values.tolist() == pytest.approx(
            poly2.dataframe[poly.xname].values.tolist()
        )
    else:
        with pytest.raises(NotImplementedError):
            poly.to_roxar(roxar_project, "xxx", "folder/sub", stype="general2d_data")


@pytest.mark.requires_roxar
def test_rox_get_modify_set_points(roxar_project):
    """Get, modify and set a points from a RMS project."""
    poi = xtgeo.points_from_roxar(roxar_project, POINTSNAME1, "", stype="clipboard")
    assert poi.dataframe.iloc[-1, 1] == pytest.approx(5.932977e06)
    assert poi.dataframe.shape[0] == 20
    assert poi.dataframe.shape[1] == 3

    # snap to a surface as operation and store in RMS
    surf = xtgeo.surface_from_roxar(roxar_project, "TopReek", SURFCAT1)
    poi.snap_surface(surf, activeonly=False)
    poi.to_roxar(roxar_project, "SNAPPED", "", stype="clipboard")
    assert poi.dataframe.iloc[-1, 2] == pytest.approx(1651.805261)


@pytest.mark.requires_roxar
def test_rox_get_modify_set_points_with_attrs(roxar_project):
    """Get, modify and set a points with attributes from a RMS project."""
    poi = xtgeo.points_from_roxar(
        roxar_project, POINTSNAME2, "", stype="clipboard", attributes=True
    )
    assert poi.dataframe["Well"].values[-1] == "WI_3"
    assert poi.dataframe.shape[0] == 8
    assert poi.dataframe.shape[1] == 7

    # snap to a surface as operation and store in RMS
    surf = xtgeo.surface_from_roxar(roxar_project, "TopReek", SURFCAT1)
    poi.snap_surface(surf, activeonly=False)
    poi.to_roxar(roxar_project, "SNAPPED2", "", stype="clipboard")
    assert poi.dataframe.iloc[-1, 2] == pytest.approx(1706.1469, abs=0.01)


@pytest.mark.requires_roxar
def test_rox_get_modify_set_points_with_attrs_pfilter(roxar_project):
    """Get, modify and set a points with attributes from a RMS project incl. pfilter."""
    poi = xtgeo.points_from_roxar(
        roxar_project, POINTSNAME2, "", stype="clipboard", attributes=True
    )
    # store to roxar with attributes using a 'pfilter'
    poi.to_roxar(
        roxar_project,
        "PFILTER_POINTS",
        "",
        stype="clipboard",
        pfilter={"Well": ["OP_1", "OP_2"]},
        attributes=True,
    )

    # reread from roxar; shall now have only 2 rows
    poi2 = xtgeo.points_from_roxar(
        roxar_project, "PFILTER_POINTS", "", stype="clipboard", attributes=True
    )

    assert "OP_4" in poi.dataframe["Well"].values.tolist()
    assert "OP_4" not in poi2.dataframe["Well"].values.tolist()
    assert poi2.dataframe.shape[0] == 2
