# -*- coding: utf-8 -*-
""""
Creates a tmp RMS project in given version which is used as fixture for all other Roxar
API dependent tests.

Then run tests in Roxar API which focus on IO

This requires a ROXAPI license, and to be ran in a "roxenvbash" environment; hence
the decarator "skipunlessroxar"

"""
from os.path import join, isdir
import shutil
import pytest

import xtgeo

import test_common.test_xtg as tsetup

xtg = xtgeo.common.XTGeoDialog()
logger = xtg.basiclogger(__name__)

if not xtg.testsetup():
    raise SystemExit

TMPD = xtg.tmpdir
TESTPATH = xtg.testpath

PROJNAME = "tmp_project.rmsxxx"
PRJ = join(TMPD, PROJNAME)

GRIDDATA1 = "../xtgeo-testdata/3dgrids/reek/reek_sim_grid.roff"
PORODATA1 = "../xtgeo-testdata/3dgrids/reek/reek_sim_poro.roff"


@tsetup.skipunlessroxar
@pytest.fixture(name="create_project", scope="module", autouse=True)
def test_create_project():
    """Create a tmp RMS project for testing, populate with basic data"""
    import roxar

    pname = join(TMPD, PROJNAME)

    if isdir(pname):
        print("Remove existing project!")
        shutil.rmtree(pname)

    project = roxar.Project.create()

    # populate with grid and props
    grd = xtgeo.grid_from_file(GRIDDATA1)
    grd.to_roxar(project, "Simgrid")
    por = xtgeo.gridproperty_from_file(PORODATA1, name="PORO")
    por.to_roxar(project, "Simgrid", "PORO")

    # save project and exit
    project.save_as(join(TMPD, PROJNAME))
    project.close()


@tsetup.skipunlessroxar
def test_rox_get_gridproperty():
    """Get a grid property from a RMS project."""

    print("Project is {}".format(PRJ))

    poro = xtgeo.grid3d.GridProperty()
    poro.from_roxar(PRJ, "Simgrid", "PORO")

    tsetup.assert_almostequal(poro.values.mean(), 0.16774, 0.001)
    assert poro.dimensions == (40, 64, 14)


@tsetup.skipunlessroxar
def test_rox_get_modify_set_gridproperty():
    """Get and set a grid property from a RMS project."""
    from roxar import __version__ as ver

    print("Roxar API version is", ver)
    poro = xtgeo.grid3d.GridProperty()
    poro.from_roxar(PRJ, "Simgrid", "PORO")

    adder = 0.9
    poro.values = poro.values + adder

    poro.to_roxar(PRJ, "Simgrid", "PORO_NEW", saveproject=True)

    poro.from_roxar(PRJ, "Simgrid", "PORO_NEW")
    tsetup.assert_almostequal(poro.values[1, 0, 0], 0.14942 + adder, 0.0001)
