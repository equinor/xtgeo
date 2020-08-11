# -*- coding: utf-8 -*-
""""
Creates a tmp RMS project in given version which is used as fixture for all other Roxar
API dependent tests.

Then run tests in Roxar API which focus on IO

This requires a ROXAPI license, and to be ran in a "roxenvbash" environment; hence
the decarator "roxapilicense"

"""
from os.path import join, isdir
import shutil
import pytest

import xtgeo

try:
    import roxar
except ImportError:
    pass

import test_common.test_xtg as tsetup

xtg = xtgeo.common.XTGeoDialog()
logger = xtg.basiclogger(__name__)

if not xtg.testsetup():
    raise SystemExit

TMPD = xtg.tmpdir
TESTPATH = xtg.testpath

PROJNAME = "tmp_project.rmsxxx"
PRJ = join(TMPD, PROJNAME)

CUBEDATA1 = "../xtgeo-testdata/cubes/reek/syntseis_20000101_seismic_depth_stack.segy"
CUBENAME1 = "synth1"

SURFTOPS1 = [
    "../xtgeo-testdata/surfaces/reek/1/topreek_rota.gri",
    "../xtgeo-testdata/surfaces/reek/1/midreek_rota.gri",
    "../xtgeo-testdata/surfaces/reek/1/lowreek_rota.gri",
]


SURFCAT1 = "DS_whatever"
SURFNAMES1 = ["TopReek", "MidReek", "BaseReek"]

GRIDDATA1 = "../xtgeo-testdata/3dgrids/reek/reek_sim_grid.roff"
PORODATA1 = "../xtgeo-testdata/3dgrids/reek/reek_sim_poro.roff"
GRIDNAME1 = "Simgrid"
PORONAME1 = "PORO"

WELLSFOLDER1 = "../xtgeo-testdata/wells/reek/1"
WELLS1 = ["OP1_perf.w", "OP_2.w", "OP_6.w", "XP_with_repeat.w"]

# ======================================================================================
# Initial tmp project


@tsetup.roxapilicenseneeded
@pytest.fixture(name="create_project", scope="module", autouse=True)
def test_create_project():
    """Create a tmp RMS project for testing, populate with basic data"""

    prj1 = PRJ
    prj2 = PRJ + "_initial"

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

    # populate with well data (postponed)

    # save project (both an initla version and a work version) and exit
    project.save_as(prj1)
    project.save_as(prj2)
    project.close()


# ======================================================================================
# Cube data


@tsetup.roxapilicenseneeded
def test_rox_getset_cube():
    """Get a cube from a RMS project, do some stuff and store/save."""

    cube = xtgeo.cube_from_roxar(PRJ, CUBENAME1)
    assert cube.values.mean() == pytest.approx(0.000718, abs=0.001)
    cube.values += 100
    assert cube.values.mean() == pytest.approx(100.000718, abs=0.001)
    cube.to_roxar(PRJ, CUBENAME1 + "_copy1")
    cube.to_roxar(PRJ, CUBENAME1 + "_copy2", folder="somefolder")


# ======================================================================================
# Surface data


@tsetup.roxapilicenseneeded
def test_rox_surfaces():
    """Various get set on surfaces in RMS"""
    srf = xtgeo.surface_from_roxar(PRJ, "TopReek", SURFCAT1)
    srf2 = xtgeo.surface_from_roxar(PRJ, "MidReek", SURFCAT1)
    assert srf.ncol == 554
    assert srf.values.mean() == pytest.approx(1698.648, abs=0.01)

    srf.to_roxar(PRJ, "TopReek_copy", "SomeFolder", stype="clipboard")

    # open project and do save explicit
    rox = xtgeo.RoxUtils(PRJ)
    prj = rox.project
    iso = srf2 - srf
    rox.create_zones_category("IS_isochore")
    prj.zones.create("UpperReek", prj.horizons["TopReek"], prj.horizons["MidReek"])
    iso.to_roxar(prj, "UpperReek", "IS_isochore", stype="zones")

    iso2 = xtgeo.surface_from_roxar(prj, "UpperReek", "IS_isochore", stype="zones")
    assert iso2.values.mean() == pytest.approx(20.79, abs=0.01)

    prj.save()
    prj.close()


# ======================================================================================
# Well data


@tsetup.roxapilicenseneeded
def test_rox_wells():
    """Various tests on Roxar wells"""


# ======================================================================================
# 3D grids and props


@tsetup.roxapilicenseneeded
def test_rox_get_gridproperty():
    """Get a grid property from a RMS project."""

    print("Project is {}".format(PRJ))

    poro = xtgeo.gridproperty_from_roxar(PRJ, GRIDNAME1, PORONAME1)

    tsetup.assert_almostequal(poro.values.mean(), 0.16774, 0.001)
    assert poro.dimensions == (40, 64, 14)


@tsetup.roxapilicenseneeded
def test_rox_get_modify_set_gridproperty():
    """Get and set a grid property from a RMS project."""

    poro = xtgeo.gridproperty_from_roxar(PRJ, GRIDNAME1, PORONAME1)

    adder = 0.9
    poro.values = poro.values + adder

    poro.to_roxar(PRJ, GRIDNAME1, PORONAME1 + "_NEW")

    poro.from_roxar(PRJ, GRIDNAME1, PORONAME1 + "_NEW")
    tsetup.assert_almostequal(poro.values[1, 0, 0], 0.14942 + adder, 0.0001)


@tsetup.roxapilicenseneeded
def test_rox_get_modify_set_grid():
    """Get, modify and set a grid from a RMS project."""

    grd = xtgeo.grid_from_roxar(PRJ, GRIDNAME1)

    grd.translate_coordinates(translate=(200, 3000, 300))

    grd.to_roxar(PRJ, GRIDNAME1 + "_edit1")
