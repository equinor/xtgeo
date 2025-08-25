"""Integration tests towards Roxar API, requires RoxarAPI license.

Creates a temporary RMS project in given version which is used as fixture for
all other Roxar API dependent tests.

Then run tests in Roxar API which focus on IO.

Since these tests require a ROXAPI license, it needs a special host setup and cannot be
ran in e.g. public Github actions.
"""

from __future__ import annotations

import logging
import pathlib
from typing import Any

import numpy as np
import pytest

import xtgeo
from xtgeo.roxutils._roxar_loader import roxar, roxar_jobs, roxar_well_picks

logger = logging.getLogger(__name__)

PROJNAME = "tmp_project.rmsxxx"

CUBEDATA1 = pathlib.Path("cubes/reek/syntseis_20000101_seismic_depth_stack.segy")
CUBENAME1 = "synth1"
CUBENAME2 = "synth2"

SURFTOPS1 = [
    pathlib.Path("surfaces/reek/1/topreek_rota.gri"),
    pathlib.Path("surfaces/reek/1/midreek_rota.gri"),
    pathlib.Path("surfaces/reek/1/lowreek_rota.gri"),
]


SURFCAT1 = "DS_whatever"
SURFNAMES1 = ["TopReek", "MidReek", "BaseReek"]

GRIDDATA1 = pathlib.Path("3dgrids/reek/reek_sim_grid.roff")
PORODATA1 = pathlib.Path("3dgrids/reek/reek_sim_poro.roff")
ZONEDATA1 = pathlib.Path("3dgrids/reek/reek_sim_zone.roff")

GRIDNAME1 = "Simgrid"
PORONAME1 = "PORO"
ZONENAME1 = "Zone"

WELLSFOLDER1 = pathlib.Path("wells/reek/3")
WELLS1 = ["OP1_perf.w", "OP_2.w", "OP_6.w", "XP_with_repeat.w"]

POLYDATA1 = pathlib.Path("polygons/reek/1/polset2.pol")
POINTSDATA1 = pathlib.Path("points/reek/1/pointset3.poi")
POINTSDATA2 = pathlib.Path("points/reek/1/poi_attr.rmsattr")
POINTSCAT1 = "DP_whatever"
POLYNAME1 = "Polys"
POINTSNAME1 = "Points"
POINTSNAME2 = "PointsAttrs"

WELL_PICK_SET = "MyWellPicks"
WELL_PICK_DATA = [
    ["OP_2", 1585.5, "TopReek"],
    ["OP_2", 1602.4, "MidReek"],
    ["OP_2", 1615.8, "BaseReek"],
]


@pytest.fixture(scope="module")
def tmp_data_dir(tmp_path_factory):
    """In order to make tmpdir/data in a module scope."""
    return tmp_path_factory.mktemp("data")


@pytest.mark.requires_roxar
@pytest.fixture(name="roxinstance", scope="module")
def fixture_roxinstance():
    """Create roxinstance in module scope."""
    # Imports indirectly from rmsapi also
    from roxar import LicenseError

    try:
        project = roxar.Project.create()
    except LicenseError:
        pytest.skip(
            "Unable to check out RMS API license. This indicates an error that "
            "must be resolved with the RMS installation or RMS license server.",
            allow_module_level=True,
        )
    return xtgeo.RoxUtils(project)


def _run_blocked_wells_job(
    gmname: str,
    bwname: str,
    wells: list[str],
):
    """Using roxar.jobs to make blocked wells.

    This option with 1.4 (RMS 12).

    The log names (Poro, Perm, etc) are here hardcoded in params dictionary.
    """

    # create the block well job and add it to the grid
    bw_job = roxar_jobs.Job.create(
        ["Grid models", gmname, "Grid"], "Block Wells", "API_BW_Job"
    )

    # wells are formulated as [["Wells", wellname1],["Wells", wellname2]...]
    well_input = [["Wells", wname] for wname in wells]

    # set the output name of the blocked wells data
    params = {
        "BlockedWellsName": bwname,
        "Continuous Blocked Log": [
            {
                "AverageMethod": "ARITHMETIC",
                "AveragePower": 0,
                "CellLayerAveraging": True,
                "Interpolate": False,
                "Name": "Poro",
                "ThicknessWeighting": "MD_WEIGHT",
            },
            {
                "AverageMethod": "ARITHMETIC",
                "AveragePower": 2,
                "CellLayerAveraging": True,
                "Interpolate": True,
                "Name": "Perm",
                "ThicknessWeighting": "MD_WEIGHT",
            },
        ],
        "CreateRawLogs": False,
        "Discrete Blocked Log": [
            {
                "CellLayerAveraging": False,
                "TreatLogAs": "INTERVAL",
                "Name": "Facies",
                "ThicknessWeighting": "MD_WEIGHT",
            }
        ],
        "ExpandTruncatedGridCells": True,
        "KeepOldLogs": True,
        "KeepOldWells": True,
        "UseDeprecatedAlgorithm": True,
        "Wells": well_input,
        "Zone Blocked Log": [
            {
                "CellLayerAveraging": True,
                "TreatLogAs": "POINTS",
                "Name": "Zonelog",
                "ScaleUpType": "SUBGRID_BIAS",
                "ThicknessWeighting": "MD_WEIGHT",
                "ZoneLogArray": [1, 2, 3, 4],
            }
        ],
    }

    bw_job.set_arguments(params)
    bw_job.save()

    # execute the job
    if not bw_job.execute(0):
        raise RuntimeError("Could do execute job for well blocking")


def _add_well_pick_to_project(project: Any, well_pick_data: dict, trajectory: str):
    """Add well picks to the project."""
    wp = roxar_well_picks

    mypicks = [
        wp.WellPick.create(
            intersection_object=project.horizons[horizon],
            trajectory=project.wells[well].wellbore.trajectories[trajectory],
            md=md,
        )
        for well, md, horizon in well_pick_data
    ]
    rox_wps = project.well_picks.sets.create(WELL_PICK_SET)
    rox_wps.append(mypicks)


@pytest.fixture(name="rms_project_path", scope="module")
def fixture_create_project(tmp_data_dir, roxinstance, testdata_path) -> str:
    """Create a temporary RMS project for testing, populate with basic data.

    Returns a path (as str) to project for subsequent jobs.
    """
    tmp_project_path = str(tmp_data_dir / PROJNAME)
    project = roxinstance.project

    logger.info("Roxar version is %s", roxinstance.roxversion)
    logger.info("RMS version is %s", roxinstance.rmsversion(roxinstance.roxversion))
    assert "1." in roxinstance.roxversion

    for wfile in WELLS1:
        wobj = xtgeo.well_from_file(testdata_path / WELLSFOLDER1 / wfile)
        if "XP_with" in wfile:
            wobj.name = "OP2_w_repeat"

        wobj.to_roxar(project, wobj.name, logrun="log", trajectory="My trajectory")

    # populate with cube data
    cube = xtgeo.cube_from_file(testdata_path / CUBEDATA1)
    cube.to_roxar(project, CUBENAME1, domain="depth")

    # make a small synthetic cube with jumps in inline/crossline
    cube2 = xtgeo.Cube(
        ncol=4,
        nrow=5,
        nlay=3,
        xinc=10,
        yinc=12,
        zinc=2,
        values=99,
        ilines=[12, 14, 16, 18],
        xlines=[10030, 10040, 10050, 10060, 10070],
        rotation=20,
    )
    cube2.to_roxar(project, CUBENAME2)

    # populate with surface data
    roxinstance.create_horizons_category(SURFCAT1)
    for num, name in enumerate(SURFNAMES1):
        srf = xtgeo.surface_from_file(testdata_path / SURFTOPS1[num])
        project.horizons.create(name, roxar.HorizonType.interpreted)
        srf.to_roxar(project, name, SURFCAT1)

    # populate with grid and props
    grd = xtgeo.grid_from_file(testdata_path / GRIDDATA1)
    grd.to_roxar(project, GRIDNAME1)
    por = xtgeo.gridproperty_from_file(testdata_path / PORODATA1, name=PORONAME1)
    por.to_roxar(project, GRIDNAME1, PORONAME1)
    zon = xtgeo.gridproperty_from_file(testdata_path / ZONEDATA1, name=ZONENAME1)
    zon.to_roxar(project, GRIDNAME1, ZONENAME1)

    # populate with points and polygons (XYZ data)
    poly = xtgeo.polygons_from_file(testdata_path / POLYDATA1)
    poly.to_roxar(project, POLYNAME1, "", stype="clipboard")

    poi = xtgeo.points_from_file(testdata_path / POINTSDATA1)
    poi.to_roxar(project, POINTSNAME1, "", stype="clipboard")
    logger.info("Initialised RMS project, done!")
    # add some points into the horizon folder as well

    # populate with surface data
    roxinstance.create_horizons_category(POINTSCAT1, htype="points")
    poi.to_roxar(project, SURFNAMES1[0], POINTSCAT1, stype="horizons")

    poi = xtgeo.points_from_file(testdata_path / POINTSDATA2, fformat="rms_attr")
    poi.to_roxar(project, POINTSNAME2, "", stype="clipboard", attributes=True)
    logger.info("Initialised RMS project, done!")

    _run_blocked_wells_job(GRIDNAME1, "BW", ["OP_2", "OP_6"])
    _add_well_pick_to_project(project, WELL_PICK_DATA, trajectory="My trajectory")

    project.save_as(tmp_project_path)
    project.close()
    logger.info("Close initial project and return path")

    return tmp_project_path


@pytest.fixture(scope="module")
def wells_from_rms(rms_project_path) -> list[xtgeo.Well]:
    """Read wells from roxar project and return a list."""

    project = xtgeo.RoxUtils(rms_project_path).project

    wlist = []
    for well in project.wells:
        obj = xtgeo.well_from_roxar(
            project,
            well.name,
            logrun="log",
            trajectory="My trajectory",
        )
        obj.zonelogname = "Zonelog"

        wlist.append(obj)

    project.close()
    return wlist


@pytest.fixture(scope="function")
def rms_project(rms_project_path) -> Any:
    """Get the 'magic' project object from RMS (similar when being inside RMS).

    Technical note: XTGeo functions like ``xtgeo.surface_from_roxar(xx, ...)``
    can handle both that xx is a path to a project or the project object itself. If
    it is a path, then the xtgeo function will open and close the project, 'behind
    the scene'. If it is a project object, then the project is not closed, hence the
    need for this fixture to close the project after the test.
    """

    project = xtgeo.RoxUtils(rms_project_path).project

    yield project

    project.close()


@pytest.mark.requires_roxar
def test_rox_getset_cube(rms_project_path):
    """Get a cube from a RMS project, do some stuff and store/save."""
    cube = xtgeo.cube_from_roxar(rms_project_path, CUBENAME1)
    assert cube.values.mean() == pytest.approx(0.000718, abs=0.001)
    cube.values += 100
    assert cube.values.mean() == pytest.approx(100.000718, abs=0.001)
    cube.to_roxar(rms_project_path, CUBENAME1 + "_copy1")
    cube.to_roxar(rms_project_path, CUBENAME1 + "_copy2", folder="somefolder")


@pytest.mark.requires_roxar
def test_rox_getset_cube_with_ilxl_jumps(rms_project_path, tmp_path):
    """Get a cube from a RMS project which has jumps in inline/xline"""
    cube = xtgeo.cube_from_roxar(rms_project_path, CUBENAME2)
    cube.to_roxar(rms_project_path, CUBENAME2 + "_copy1")
    cube2 = xtgeo.cube_from_roxar(rms_project_path, CUBENAME2 + "_copy1")
    cube2.to_file(tmp_path / "cube2.segy")
    cube3 = xtgeo.cube_from_file(tmp_path / "cube2.segy")
    assert cube3.ilines.tolist() == [12, 14, 16, 18]
    assert cube3.xlines.tolist() == [10030, 10040, 10050, 10060, 10070]


@pytest.mark.requires_roxar
def test_rox_surfaces(rms_project_path):
    """Various get set on surfaces in RMS."""
    srf = xtgeo.surface_from_roxar(rms_project_path, "TopReek", SURFCAT1)
    srf2 = xtgeo.surface_from_roxar(rms_project_path, "MidReek", SURFCAT1)
    assert srf.ncol == 554
    assert srf.values.mean() == pytest.approx(1698.648, abs=0.01)

    srf.to_roxar(rms_project_path, "TopReek_copy", "SomeFolder", stype="clipboard")

    # open project and do save explicit
    rox = xtgeo.RoxUtils(rms_project_path)
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
def test_rox_surfaces_dtype_switching(rms_project_path):
    """Test dtype switching for from_roxar"""
    srf = xtgeo.surface_from_roxar(
        rms_project_path, "TopReek", SURFCAT1, dtype="float32"
    )
    assert srf.ncol == 554
    assert srf.values.mean() == pytest.approx(1698.648, abs=0.01)
    assert srf.dtype == np.float32
    srf.to_roxar(rms_project_path, "TopReek_copy", "SomeFolder", stype="clipboard")

    srf2 = srf.copy()
    assert srf2.dtype == np.float32

    srf2.dtype = np.float64
    np.testing.assert_allclose(srf.values, srf2.values)


@pytest.mark.requires_roxar
def test_rox_surfaces_alternative_open(rms_project_path):
    """Based on previous but instead use a project ref as first argument"""

    rox = xtgeo.RoxUtils(rms_project_path)

    assert isinstance(rox.project, roxar.Project)

    srf = xtgeo.surface_from_roxar(rox.project, "TopReek", SURFCAT1)
    assert srf.ncol == 554
    assert srf.values.mean() == pytest.approx(1698.648, abs=0.01)

    rox.project.close()


@pytest.mark.requires_roxar
def test_rox_surfaces_clipboard_general2d_data(rms_project, roxinstance):
    """Set and get surfaces on clipboard and general2D data"""

    surf = xtgeo.surface_from_roxar(rms_project, "TopReek", SURFCAT1)

    surf.to_roxar(rms_project, "mycase", "myfolder", stype="clipboard")
    surf2 = xtgeo.surface_from_roxar(
        rms_project, "mycase", "myfolder", stype="clipboard"
    )
    assert surf2.values.mean() == surf.values.mean()

    # general 2D data (from xtgeo version 2.19 and roxar API >= 1.6)
    if not roxinstance.version_required("1.6"):
        with pytest.raises(
            NotImplementedError, match=r"API Support for general2d_data is missing"
        ):
            surf.to_roxar(rms_project, "mycase", "myfolder", stype="general2d_data")
        logger.info("This version of RMS does not support this feature")

    else:
        surf.to_roxar(rms_project, "mycase", "myfolder", stype="general2d_data")
        surf2 = xtgeo.surface_from_roxar(
            rms_project, "mycase", "myfolder", stype="general2d_data"
        )
        assert surf2.values.tolist() == surf.values.tolist()


@pytest.mark.requires_roxar
def test_rox_get_set_trend_surfaces(rms_project_path):
    """Get, modify and set trendsurfaces from a RMS project.

    Since the current RMS API does not support write to trends.surfaces, an automatic
    test cannot be made here. The functions were tested manually in RMS.
    """
    surf = xtgeo.surface_from_roxar(rms_project_path, "TopReek", SURFCAT1)

    with pytest.raises(ValueError, match=r"Any is not within Trends"):
        surf.to_roxar(rms_project_path, "Any", None, stype="trends")

    with pytest.raises(ValueError, match=r"Any is not within Trends"):
        surf = xtgeo.surface_from_roxar(rms_project_path, "Any", None, stype="trends")


@pytest.mark.requires_roxar
def test_rox_wells(rms_project_path):
    """Various tests on Roxar wells."""
    well = xtgeo.well_from_roxar(
        rms_project_path, "OP_2", trajectory="My trajectory", logrun="log"
    )
    assert "Zonelog" in well.lognames

    assert well.get_dataframe()["Poro"].mean() == pytest.approx(0.1637623936)


@pytest.mark.requires_roxar
def test_rox_get_gridproperty(rms_project_path):
    """Get a grid property from a RMS project."""
    logger.info("Project is %s", rms_project_path)

    poro = xtgeo.gridproperty_from_roxar(rms_project_path, GRIDNAME1, PORONAME1)

    assert poro.values.mean() == pytest.approx(0.16774, abs=0.001)
    assert poro.dimensions == (40, 64, 14)

    zone = xtgeo.gridproperty_from_roxar(rms_project_path, GRIDNAME1, ZONENAME1)
    assert "int" in str(zone.values.dtype)

    zone._roxar_dtype = np.int32
    with pytest.raises(TypeError):
        zone.to_roxar(rms_project_path, GRIDNAME1, ZONENAME1)


@pytest.mark.requires_roxar
def test_rox_gridproperty_dtypes(rms_project_path):
    """Various work with a grid property using dtype."""
    logger.info("Project is %s", rms_project_path)
    prj = rms_project_path

    grid = xtgeo.grid_from_roxar(rms_project_path, GRIDNAME1)

    prop = xtgeo.GridProperty(grid, discrete=False, values=999)
    assert prop.roxar_dtype == np.float32
    prop.to_roxar(prj, GRIDNAME1, "myprop1")

    # change to discrete
    prop.isdiscrete = True
    assert prop.roxar_dtype == np.uint16
    # try to overwite the continous icon after changing data
    prop.values = 251
    with pytest.warns(UserWarning) as warning_info:
        prop.to_roxar(prj, GRIDNAME1, "myprop1")
        assert "Existing RMS icon has data type" in str(warning_info[0].message)
    # read icon again, it will still be a float
    newprop = xtgeo.gridproperty_from_roxar(prj, GRIDNAME1, "myprop1")
    assert newprop.roxar_dtype == np.float32
    assert newprop.values.dtype == np.float64  # internal in xtgeo

    # now try to save this as discrete
    newprop.isdiscrete = True
    assert newprop.roxar_dtype == np.uint16
    assert newprop.values.dtype == np.int32  # internal in xtgeo

    # store it again, should issue some warnings
    with pytest.warns(UserWarning):
        newprop.to_roxar(prj, GRIDNAME1, "myprop1")

    newprop.to_roxar(prj, GRIDNAME1, "myprop2")  # should not give warning

    newprop.isdiscrete = False
    assert newprop.roxar_dtype == np.float32
    assert newprop.values.dtype == np.float64


@pytest.mark.requires_roxar
def test_rox_get_modify_set_gridproperty(rms_project_path):
    """Get and set a grid property from a RMS project."""
    poro = xtgeo.gridproperty_from_roxar(rms_project_path, GRIDNAME1, PORONAME1)
    cell_value = poro.values[1, 0, 0]

    adder = 0.9
    poro.values = poro.values + adder

    poro.to_roxar(rms_project_path, GRIDNAME1, PORONAME1 + "_NEW")

    poronew = xtgeo.gridproperty_from_roxar(
        rms_project_path, GRIDNAME1, PORONAME1 + "_NEW"
    )
    assert poronew.values[1, 0, 0] == pytest.approx(cell_value + adder, abs=0.0001)


@pytest.mark.requires_roxar
def test_rox_get_modify_set_grid_basic(rms_project_path):
    """Get, modify and set a grid from a RMS project."""
    grd = xtgeo.grid_from_roxar(rms_project_path, GRIDNAME1)
    grd1 = grd.copy()

    grd1.translate_coordinates(translate=(200, 3000, 300))

    grd1.to_roxar(rms_project_path, GRIDNAME1 + "_edit1")

    grd2 = xtgeo.grid_from_roxar(rms_project_path, GRIDNAME1 + "_edit1")

    assert grd2.dimensions == grd1.dimensions


@pytest.mark.requires_roxar
def test_rox_get_modify_set_grid_method_roff(rms_project_path):
    """Get, modify and set a grid from a RMS project."""
    grd = xtgeo.grid_from_roxar(rms_project_path, GRIDNAME1)
    grd1 = grd.copy()

    grd1.translate_coordinates(translate=(200, 3000, 300))

    grd1.to_roxar(rms_project_path, GRIDNAME1 + "_roff", method="roff")
    grd1.to_roxar(rms_project_path, GRIDNAME1 + "_cpg", method="cpg")

    grd2_roff = xtgeo.grid_from_roxar(rms_project_path, GRIDNAME1 + "_roff")
    grd2_cpg = xtgeo.grid_from_roxar(rms_project_path, GRIDNAME1 + "_cpg")

    assert grd2_roff.dimensions == grd2_cpg.dimensions


@pytest.mark.benchmark(group="grid_to_roxar_method")
@pytest.mark.requires_roxar
def test_rox_set_grid_method_benchmark_cpg(rms_project_path, benchmark):
    """Get, modify and set a grid from a RMS project."""
    grd = xtgeo.grid_from_roxar(rms_project_path, GRIDNAME1)
    grd1 = grd.copy()

    grd1.translate_coordinates(translate=(200, 3000, 300))

    def store_method_cpg():
        grd1.to_roxar(rms_project_path, GRIDNAME1 + "_cpg", method="cpg")

    benchmark(store_method_cpg)


@pytest.mark.benchmark(group="grid_to_roxar_method")
@pytest.mark.requires_roxar
def test_rox_set_grid_method_benchmark_roff(rms_project_path, benchmark):
    """Get, modify and set a grid from a RMS project."""
    grd = xtgeo.grid_from_roxar(rms_project_path, GRIDNAME1)
    grd1 = grd.copy()

    grd1.translate_coordinates(translate=(200, 3000, 300))

    def store_method_roff():
        grd1.to_roxar(rms_project_path, GRIDNAME1 + "_roff", method="roff")

    benchmark(store_method_roff)


@pytest.mark.requires_roxar
def test_rox_get_modify_set_get_grid_with_subzones(rms_project_path, roxinstance):
    """Get, modify and set + get a grid from a RMS project using subzones/subgrids."""

    grd = xtgeo.grid_from_roxar(rms_project_path, GRIDNAME1)

    zonation = {}
    zonation["intva"] = 4
    zonation["intvb"] = 7
    zonation["intvc"] = 3
    grd.set_subgrids(zonation)

    if not roxinstance.version_required("1.6"):
        with pytest.warns(UserWarning, match=r"Implementation of subgrids is lacking"):
            grd.to_roxar(rms_project_path, "NewGrid")
    else:
        grd.to_roxar(rms_project_path, "NewGrid")

        # get a new instance for recent storage (subgrids should now be present)
        grd1 = xtgeo.grid_from_roxar(rms_project_path, "NewGrid")

        for intv in ["intva", "intvb", "intvc"]:
            assert list(grd.subgrids[intv]) == list(grd1.subgrids[intv])


@pytest.mark.requires_roxar
def test_rox_get_modify_set_polygons(rms_project_path, roxinstance):
    """Get, modify and set a polygons from a RMS project."""
    poly = xtgeo.polygons_from_roxar(rms_project_path, POLYNAME1, "", stype="clipboard")
    assert poly.get_dataframe().iloc[-1, 2] == pytest.approx(1595.161377)
    assert poly.get_dataframe().shape[0] == 25
    assert poly.get_dataframe().shape[1] == 4

    poly.rescale(300)
    # store in RMS
    poly.to_roxar(rms_project_path, "RESCALED", "", stype="clipboard")
    assert poly.get_dataframe().shape[0] == 127

    # store and retrieve in general2d_data just to see that it works
    if roxinstance.version_required("1.6"):
        poly.to_roxar(rms_project_path, "xxx", "folder/sub", stype="general2d_data")
        poly2 = xtgeo.polygons_from_roxar(
            rms_project_path, "xxx", "folder/sub", stype="general2d_data"
        )
        assert poly.get_dataframe()[poly.xname].values.tolist() == pytest.approx(
            poly2.get_dataframe()[poly.xname].values.tolist()
        )
    else:
        with pytest.raises(NotImplementedError):
            poly.to_roxar(rms_project_path, "xxx", "folder/sub", stype="general2d_data")


@pytest.mark.requires_roxar
def test_rox_get_modify_set_points(rms_project_path):
    """Get, modify and set a points from a RMS project."""
    poi = xtgeo.points_from_roxar(rms_project_path, POINTSNAME1, "", stype="clipboard")
    assert poi.get_dataframe().iloc[-1, 1] == pytest.approx(5.932977e06)
    assert poi.get_dataframe().shape[0] == 20
    assert poi.get_dataframe().shape[1] == 3

    # snap to a surface as operation and store in RMS
    surf = xtgeo.surface_from_roxar(rms_project_path, "TopReek", SURFCAT1)
    poi.snap_surface(surf, activeonly=False)
    poi.to_roxar(rms_project_path, "SNAPPED", "", stype="clipboard")
    assert poi.get_dataframe().iloc[-1, 2] == pytest.approx(1651.805261)


@pytest.mark.requires_roxar
def test_rox_get_modify_set_points_from_horizons(rms_project_path):
    """Get, modify and set a points from a RMS project."""
    poi = xtgeo.points_from_roxar(
        rms_project_path, SURFNAMES1[0], POINTSCAT1, stype="horizons"
    )
    assert poi.get_dataframe().iloc[-1, 1] == pytest.approx(5.932977e06)
    assert poi.get_dataframe().shape[0] == 20
    assert poi.get_dataframe().shape[1] == 3
    poi.to_roxar(rms_project_path, SURFNAMES1[0], POINTSCAT1, stype="horizons")


@pytest.mark.requires_roxar
def test_rox_set_points_with_inconsistent_xyz_names(rms_project_path):
    """
    Export points to a RMS project where the dataframe has another zname
    than the zname attribute. This should fail.
    """
    poi = xtgeo.points_from_roxar(
        rms_project_path, SURFNAMES1[0], POINTSCAT1, stype="horizons"
    )

    df = poi.get_dataframe()
    df.rename(columns={poi.zname: "Z"}, inplace=True)
    poi.set_dataframe(df)

    # inconsistency between z column name and zname attribute should fail
    with pytest.raises(ValueError, match="One or all"):
        poi.to_roxar(rms_project_path, SURFNAMES1[0], POINTSCAT1, stype="horizons")


@pytest.mark.requires_roxar
def test_rox_set_points_with_nonstandard_xyz_names(rms_project_path):
    """Export points with nonstandard xyz names to RMS."""
    poi = xtgeo.points_from_roxar(
        rms_project_path, SURFNAMES1[0], POINTSCAT1, stype="horizons"
    )

    # first check that "Z" is not part of the dataframe
    assert "X" not in poi.get_dataframe(copy=False)

    # then update the zname attribute this should also set the column name
    poi.zname = "X"
    assert "X" in poi.get_dataframe(copy=False)

    # check that storing to roxar works fine
    poi.to_roxar(rms_project_path, SURFNAMES1[0], POINTSCAT1, stype="horizons")

    # another indirect check using points from surface.
    # here Z name is set on initialisation
    srf = xtgeo.surface_from_roxar(rms_project_path, "TopReek", SURFCAT1)
    poi = xtgeo.points_from_surface(srf, zname="MyZ")
    assert "MyZ" in poi.get_dataframe(copy=False)
    poi.to_roxar(rms_project_path, SURFNAMES1[0], POINTSCAT1, stype="horizons")


@pytest.mark.requires_roxar
def test_check_presence_in_project_errors(rms_project_path):
    # test category not existing in project

    rox = xtgeo.RoxUtils(rms_project_path)
    with pytest.raises(ValueError) as exc_info:
        name = "I_dont_exist"
        xtgeo.points_from_roxar(rox.project, name, POINTSCAT1, stype="horizons")
    assert str(exc_info.value) == f"Cannot access {name=} in horizons"

    # test category not given
    with pytest.raises(ValueError) as exc_info:
        xtgeo.points_from_roxar(
            rox.project, SURFNAMES1[0], category=None, stype="horizons"
        )
    assert (
        str(exc_info.value) == "Need to specify category for horizons, zones and faults"
    )

    # test category not existing in project
    with pytest.raises(ValueError) as exc_info:
        category = "I_dont_exist"
        xtgeo.points_from_roxar(rox.project, SURFNAMES1[0], category, stype="horizons")
        rox.project.close()
    assert str(exc_info.value) == f"Cannot access {category=} in horizons"

    # test empty data in project
    with pytest.raises(RuntimeError) as exc_info:
        name = SURFNAMES1[1]
        category = POINTSCAT1
        xtgeo.points_from_roxar(rox.project, name, category, stype="horizons")
        rox.project.close()
    assert str(exc_info.value) == f"'{name}' is empty for horizons {category=}"

    rox.project.close()


@pytest.mark.requires_roxar
def test_rox_get_modify_set_points_with_attrs(rms_project_path):
    """Get, modify and set a points with attributes from a RMS project."""
    poi = xtgeo.points_from_roxar(
        rms_project_path, POINTSNAME2, "", stype="clipboard", attributes=True
    )
    assert poi.get_dataframe()["Well"].values[-1] == "WI_3"
    assert poi.get_dataframe().shape[0] == 8
    assert poi.get_dataframe().shape[1] == 7

    # snap to a surface as operation and store in RMS
    surf = xtgeo.surface_from_roxar(rms_project_path, "TopReek", SURFCAT1)
    poi.snap_surface(surf, activeonly=False)
    poi.to_roxar(rms_project_path, "SNAPPED2", "", stype="clipboard")
    assert poi.get_dataframe().iloc[-1, 2] == pytest.approx(1706.1469, abs=0.01)


@pytest.mark.requires_roxar
def test_rox_get_modify_set_points_with_attrs_pfilter(rms_project_path):
    """Get, modify and set a points with attributes from a RMS project incl. pfilter."""
    poi = xtgeo.points_from_roxar(
        rms_project_path, POINTSNAME2, "", stype="clipboard", attributes=True
    )
    # store to roxar with attributes using a 'pfilter'
    poi.to_roxar(
        rms_project_path,
        "PFILTER_POINTS",
        "",
        stype="clipboard",
        pfilter={"Well": ["OP_1", "OP_2"]},
        attributes=True,
    )

    # reread from roxar; shall now have only 2 rows
    poi2 = xtgeo.points_from_roxar(
        rms_project_path, "PFILTER_POINTS", "", stype="clipboard", attributes=True
    )

    assert "OP_4" in poi.get_dataframe()["Well"].values.tolist()
    assert "OP_4" not in poi2.get_dataframe()["Well"].values.tolist()
    assert poi2.get_dataframe().shape[0] == 2


@pytest.mark.requires_roxar
def test_get_well_picks_as_points(rms_project_path):
    """Get, well picks as points with attributes"""

    rox = xtgeo.RoxUtils(rms_project_path)
    project = rox.project

    assert set(project.well_picks.sets.keys()) == {"Default", "MyWellPicks"}
    if not rox.version_required("1.6"):
        return

    # collect well pick set as points
    poi = xtgeo.points_from_roxar(
        project, WELL_PICK_SET, "horizon", stype="well_picks", attributes=True
    )

    poi_df = poi.get_dataframe()
    assert poi_df.shape[0] == 3
    assert "My attribute" not in poi.dataframe
    assert (poi.dataframe["Depth Uncertainty"] == xtgeo.UNDEF).all()

    # update regular well pick attribute
    poi_df["Depth Uncertainty"] = 10
    # create user defined well pick attribute
    poi_df.loc[poi_df["HORIZON"] == "TopReek", "My attribute"] = "OK"
    poi.set_dataframe(poi_df)

    # store to new well pick set
    poi.to_roxar(
        project,
        WELL_PICK_SET + "_new",
        "horizon",
        stype="well_picks",
        attributes=True,
    )

    # test reading the new well pick set
    poi2 = xtgeo.points_from_roxar(
        project, WELL_PICK_SET + "_new", "horizon", stype="well_picks", attributes=True
    )
    poi2_df = poi2.get_dataframe()

    # check that the attributes are present and updated
    assert "My attribute" in poi2_df
    assert (poi2_df["Depth Uncertainty"] == 10).all()

    # test store to clipboard using a pfilter
    poi2.to_roxar(
        project,
        WELL_PICK_SET,
        "",
        stype="clipboard",
        pfilter={"HORIZON": ["TopReek"]},
        attributes=True,
    )

    # reread from clipboard; should now have only 1 rows
    poi3 = xtgeo.points_from_roxar(
        project, WELL_PICK_SET, "", stype="clipboard", attributes=True
    )
    poi3_df = poi3.get_dataframe()

    assert set(poi2_df["HORIZON"].unique()) == {"TopReek", "MidReek", "BaseReek"}
    assert set(poi3_df["HORIZON"].unique()) == {"TopReek"}
    assert poi3_df.shape[0] == 1
    assert (poi3_df["Depth Uncertainty"] == 10).all()

    # store once more as a well pick set without attributes and reread
    poi3.to_roxar(
        project,
        WELL_PICK_SET + "_newest",
        "horizon",
        stype="well_picks",
        attributes=False,
    )
    poi4 = xtgeo.points_from_roxar(
        project,
        WELL_PICK_SET + "_newest",
        "horizon",
        stype="well_picks",
        attributes=True,
    )
    poi4_df = poi4.get_dataframe()

    # should now have undefined Depth Uncertainty and only one horizon
    assert (poi4_df["Depth Uncertainty"] == xtgeo.UNDEF).all()
    assert set(poi4_df["HORIZON"].unique()) == {"TopReek"}
    assert poi4_df.shape[0] == 1

    # check that new wll pick sets have been created in the project
    assert set(project.well_picks.sets.keys()) == {
        "Default",
        "MyWellPicks",
        "MyWellPicks_new",
        "MyWellPicks_newest",
    }


@pytest.mark.requires_roxar
def test_points_from_well_tops(rms_project, wells_from_rms):
    """Extracting tops from well zonelog."""

    zonelist = list(wells_from_rms[0].wlogrecords["Zonelog"].keys())

    wtops = xtgeo.points_from_wells(
        wells_from_rms,
        zonelist=zonelist,
    )
    assert wtops._attrs["TopName"] == "str"
    assert wtops._attrs["X_UTME"] == "float"

    for topname in ["TopUppReek", "TopMidReek", "TopLowReek"]:
        wtops.to_roxar(
            rms_project,
            topname,
            "MyWellPoints",
            pfilter={"TopName": [topname]},
            attributes=True,
            stype="clipboard",
        )

    uppreek_points = xtgeo.points_from_roxar(
        rms_project,
        "TopUppReek",
        "MyWellPoints",
        stype="clipboard",
        attributes=True,
    )
    assert uppreek_points.get_dataframe()["Zone"][0] == 1

    nwells = uppreek_points.get_nwells()
    assert nwells == 3


@pytest.mark.requires_roxar
def test_points_from_well_thickness(rms_project, wells_from_rms):
    """Extracting points for thickness / isocores from wells."""

    w_isos = xtgeo.points_from_wells(
        wells_from_rms,
        zonelist=list(wells_from_rms[0].wlogrecords["Zonelog"].keys()),
        tops=False,
    )
    assert w_isos._attrs["ZoneName"] == "str"
    assert w_isos._attrs["X_UTME"] == "float"

    for iname in ["UppReek", "MidReek", "LowReek"]:
        w_isos.to_roxar(
            rms_project,
            iname,
            "MyWellIsos",
            pfilter={"ZoneName": [iname]},
            attributes=True,
            stype="clipboard",
        )

    uppreek_iso_points = xtgeo.points_from_roxar(
        rms_project,
        "UppReek",
        "MyWellIsos",
        stype="clipboard",
        attributes=True,
    )
    assert "UppReek" in uppreek_iso_points.get_dataframe()["ZoneName"].to_numpy()
    assert "MidReek" not in uppreek_iso_points.get_dataframe()["ZoneName"].to_numpy()


@pytest.mark.requires_roxar
def test_lines_from_well(rms_project, wells_from_rms):
    """Extracting line data (pieces) from well data, per zone."""

    zonelist = list(wells_from_rms[0].wlogrecords["Zonelog"].keys())
    zonenames = list(wells_from_rms[0].wlogrecords["Zonelog"].values())

    for code in zonelist[1:]:
        w_line = xtgeo.polygons_from_wells(
            wells_from_rms,
            code,
            resample=1,
        )
        w_line.to_roxar(rms_project, zonenames[code], "MyWellLines", stype="clipboard")

        w_line_read = xtgeo.polygons_from_roxar(
            rms_project,
            zonenames[code],
            "MyWellLines",
            stype="clipboard",
        )
        if code == 1:
            assert (w_line_read.get_dataframe()["POLY_ID"] == 2).sum() == 26


@pytest.mark.requires_roxar
def test_well_picks_version_requirement(rms_project_path):
    """Chech rox version requirement for well picks"""

    rox = xtgeo.RoxUtils(rms_project_path)
    project = rox.project

    if not rox.version_required("1.6"):
        with pytest.raises(NotImplementedError) as exc_info:
            xtgeo.points_from_roxar(
                project, WELL_PICK_SET, "horizon", stype="well_picks"
            )
        assert str(exc_info.value).startswith("API Support for well_picks is missing")

    rox.safe_close()


@pytest.mark.requires_roxar
def test_get_well_picks_attributes(rms_project_path):
    """Get, well picks as points with attributes"""

    rox = xtgeo.RoxUtils(rms_project_path)
    project = rox.project

    if not rox.version_required("1.6"):
        return

    poi = xtgeo.points_from_roxar(
        project, WELL_PICK_SET, "horizon", stype="well_picks", attributes=True
    )

    poi_df = poi.get_dataframe()
    assert "Depth Uncertainty" in poi_df
    assert "Azimuth" in poi_df

    poi = xtgeo.points_from_roxar(
        project, WELL_PICK_SET, "horizon", stype="well_picks", attributes=False
    )
    poi_df = poi.get_dataframe()
    assert "Depth Uncertainty" not in poi_df
    assert "Azimuth" not in poi_df

    poi = xtgeo.points_from_roxar(
        project,
        WELL_PICK_SET,
        "horizon",
        stype="well_picks",
        attributes=["Azimuth"],
    )
    poi_df = poi.get_dataframe()
    assert "Depth Uncertainty" not in poi_df
    assert "Azimuth" in poi_df


@pytest.mark.requires_roxar
def test_rox_well_with_added_logs(rms_project_path):
    """Operations on discrete well logs"""
    well = xtgeo.well_from_roxar(
        rms_project_path,
        WELLS1[1].replace(".w", ""),
        logrun="log",
        trajectory="My trajectory",
    )
    assert well.get_dataframe()["Facies"].mean() == pytest.approx(0.357798165)
    well.to_roxar(rms_project_path, "dummy1", logrun="log", trajectory="My trajectory")
    dataframe = well.get_dataframe()
    dataframe["Facies"] = np.nan
    well.set_dataframe(dataframe)
    assert np.isnan(well.get_dataframe()["Facies"].values).all()
    well.to_roxar(rms_project_path, "dummy2", logrun="log", trajectory="My trajectory")
    # check that export with set codes
    well.set_logrecord("Facies", {1: "name"})
    well.to_roxar(rms_project_path, "dummy3", logrun="log", trajectory="My trajectory")


@pytest.mark.requires_roxar
@pytest.mark.parametrize(
    "update_option, expected_logs, expected_poroavg",
    [
        (None, ["Poro", "NewPoro"], 0.26376),
        ("overwrite", ["Zonelog", "Perm", "Poro", "Facies", "NewPoro"], 0.26376),
        ("append", ["Zonelog", "Perm", "Poro", "Facies", "NewPoro"], 0.16376),
    ],
)
def test_rox_well_update(
    rms_project_path, update_option, expected_logs, expected_poroavg
):
    """Operations on discrete well logs"""
    initial_wellname = WELLS1[1].replace(".w", "")
    wellname = "TESTWELL"

    initial_well = xtgeo.well_from_roxar(
        rms_project_path,
        initial_wellname,
        logrun="log",
        lognames="all",
        trajectory="My trajectory",
    )
    initial_well.to_roxar(rms_project_path, wellname)

    well = xtgeo.well_from_roxar(
        rms_project_path,
        wellname,
        lognames=["Poro"],
    )
    well.create_log("NewPoro")
    dataframe = well.get_dataframe()
    dataframe["Poro"] += 0.1
    well.set_dataframe(dataframe)

    well.to_roxar(
        rms_project_path,
        wellname,
        lognames=well.lognames,
        update_option=update_option,
    )
    print("Lognames are", well.lognames)

    rox = xtgeo.RoxUtils(rms_project_path)

    rox_lcurves = (
        rox.project.wells[wellname]
        .wellbore.trajectories["Drilled trajectory"]
        .log_runs["log"]
        .log_curves
    )
    rox_lognames = [lname.name for lname in rox_lcurves]
    assert rox_lognames == expected_logs

    assert rox_lcurves["Poro"].get_values().mean() == pytest.approx(
        expected_poroavg, abs=0.001
    )


@pytest.mark.requires_roxar
def test_blocked_well_from_to_roxar(rms_project_path):
    """Test getting blocked wells from RMS API."""

    rox = xtgeo.RoxUtils(rms_project_path)

    bw = xtgeo.blockedwell_from_roxar(
        rox.project, GRIDNAME1, "BW", "OP_2", lognames="all"
    )
    assert list(bw.get_dataframe().columns) == [
        "X_UTME",
        "Y_UTMN",
        "Z_TVDSS",
        "I_INDEX",
        "J_INDEX",
        "K_INDEX",
        "Zonelog",
        "Poro",
        "Perm",
        "Facies",
    ]

    bw.delete_log("Zonelog")
    bw.create_log("Some_new")

    bw.to_roxar(rox.project, GRIDNAME1, "BW", "OP_2")

    # read again from RMS
    bw_2 = xtgeo.blockedwell_from_roxar(
        rox.project, GRIDNAME1, "BW", "OP_2", lognames="all"
    )

    # zonelog will still be in Roxar since it was there from before
    assert "Zonelog" in list(bw_2.get_dataframe().columns)
    assert "Some_new" in list(bw_2.get_dataframe().columns)

    rox.project.close()


@pytest.mark.requires_roxar
def test_blocked_well_roxar_to_from_file(rms_project_path, tmp_path):
    """Test getting a single blocked well from RMS, store to file and import again."""

    rox = xtgeo.RoxUtils(rms_project_path)

    bw = xtgeo.blockedwell_from_roxar(
        rox.project, GRIDNAME1, "BW", "OP_2", lognames="all"
    )
    filename = tmp_path / "op2.bw"
    bw.to_file(filename)
    with open(filename, "r") as fhandle:
        ff = str(fhandle.readlines())
        assert "Unknown" in ff
        assert "460297.7747" in ff

    bw_op2 = xtgeo.blockedwell_from_file(filename)
    assert bw_op2.name == "OP_2"
    rox.project.close()


@pytest.mark.requires_roxar
def test_blocked_wells_roxar_to_from_file(rms_project_path, tmp_path):
    """Test getting blocked wells (plural) from RMS, store to files and import again."""

    rox = xtgeo.RoxUtils(rms_project_path)

    bwells = xtgeo.blockedwells_from_roxar(rox.project, GRIDNAME1, "BW", lognames="all")
    assert bwells.names == ["OP_2", "OP_6"]

    filenames = []
    for bw in bwells:
        filename = tmp_path / f"{bw.name}.bw"
        bw.to_file(filename)
        filenames.append(filename)
        with open(filename, "r") as fhandle:
            ff = str(fhandle.readlines())
            assert bw.name in ff

    bwells2 = xtgeo.blockedwells_from_files(filenames)
    assert bwells2.names == bwells.names
    rox.project.close()
