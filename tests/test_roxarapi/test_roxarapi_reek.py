# -*- coding: utf-8 -*-
"""Integration tests towards Roxar API, requires RoxarAPI license.

Creates a tmp RMS project in given version which is used as fixture for all other Roxar
API dependent tests.

Then run tests in Roxar API which focus on IO.

This requires a ROXAPI license, and to be ran in a "roxenvbash" environment if Equinor.
"""
from os.path import join, isdir
import shutil
import pytest
import numpy as np
import xtgeo

try:
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

# ======================================================================================
# Initial tmp project


@pytest.fixture(name="tmp_project_path")
def fixture_tmp_project_path(tmpdir):
    return join(tmpdir, PROJNAME)


@pytest.mark.requires_roxar
@pytest.fixture(name="roxar_project")
def fixture_create_project(tmp_project_path):
    """Create a tmp RMS project for testing, populate with basic data."""
    prj1 = tmp_project_path
    prj2 = tmp_project_path + "_initial"

    print("\n******** Setup RMS project!\n")
    if isdir(prj1):
        print("Remove existing project! (1)")
        shutil.rmtree(prj1)

    if isdir(prj2):
        print("Remove existing project! (2)")
        shutil.rmtree(prj2)

    project = roxar.Project.create()

    rox = xtgeo.RoxUtils(project)
    print("Roxar version is", rox.roxversion)
    print("RMS version is", rox.rmsversion(rox.roxversion))
    assert "1." in rox.roxversion

    for wfile in WELLS1:
        wobj = xtgeo.well_from_file(WELLSFOLDER1 / wfile)
        if "XP_with" in wfile:
            wobj.name = "OP2_w_repeat"

        wobj.to_roxar(project, wobj.name, logrun="log", trajectory="My trajectory")

    # populate with cube data
    cube = xtgeo.cube_from_file(CUBEDATA1)
    cube.to_roxar(project, CUBENAME1, domain="depth")

    # populate with surface data
    rox.create_horizons_category(SURFCAT1)
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

    # save project (both an initla version and a work version) and exit
    project.save_as(prj1)
    project.save_as(prj2)
    project.close()

    yield tmp_project_path

    project.close()

    print("\n******* Teardown RMS project!\n")
    if isdir(prj1):
        print("Remove existing project! (1)")
        shutil.rmtree(prj1)

    if isdir(prj2):
        print("Remove existing project! (2)")
        shutil.rmtree(prj2)


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
    print("Project is {}".format(roxar_project))

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
