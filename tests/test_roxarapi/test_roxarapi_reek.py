# -*- coding: utf-8 -*-
"""Integration tests towards Roxar API, requires RoxarAPI license.

Creates a tmp RMS project in given version which is used as fixture for all other Roxar
API dependent tests.

Then run tests in Roxar API which focus on IO.

This requires a ROXAPI license, and to be ran in a "roxenvbash" environment if Equinor.
"""
from __future__ import annotations

from os.path import join
from typing import Any

import numpy as np
import pytest

import xtgeo
from xtgeo.common import XTGeoDialog, null_logger

try:
    import roxar
except ImportError:
    pass

xtg = XTGeoDialog()

if not xtg.testsetup():
    raise SystemExit

logger = null_logger(__name__)

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


def _run_blocked_wells_job(
    project: Any,
    gmname: str,
    bwname: str,
    wells: list[str],
):
    """Using roxar.jobs to make blocked wells.

    This option with 1.4 (RMS 12).

    The log names (Poro, Perm, etc) are here hardcoded in params dictionary.
    """
    import roxar.jobs

    # create the block well job and add it to the grid
    bw_job = roxar.jobs.Job.create(
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


@pytest.mark.requires_roxar
@pytest.fixture(name="roxar_project", scope="module")
def fixture_create_project(tmpdir, roxinstance) -> str:
    """Create a tmp RMS project for testing, populate with basic data.

    Returns a path (as str) to project for subsequent jobs.
    """
    print("MANANA2")
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

    _run_blocked_wells_job(project, GRIDNAME1, "BW", ["OP_2", "OP_6"])

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
def test_rox_surfaces_dtype_switching(roxar_project):
    """Test dtype switching for from_roxar"""
    srf = xtgeo.surface_from_roxar(roxar_project, "TopReek", SURFCAT1, dtype="float32")
    assert srf.ncol == 554
    assert srf.values.mean() == pytest.approx(1698.648, abs=0.01)
    assert srf.dtype == np.float32
    srf.to_roxar(roxar_project, "TopReek_copy", "SomeFolder", stype="clipboard")

    srf2 = srf.copy()
    assert srf2.dtype == np.float32

    srf2.dtype = np.float64
    np.testing.assert_allclose(srf.values, srf2.values)


@pytest.mark.requires_roxar
def test_rox_surfaces_alternative_open(roxar_project):
    """Based on previous but instead use a project ref as first argument"""

    rox = xtgeo.RoxUtils(roxar_project)

    assert isinstance(rox.project, roxar.Project)

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

    zonation = dict()
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


@pytest.mark.requires_roxar
def test_rox_well_with_added_logs(roxar_project):
    """Operations on discrete well logs"""
    well = xtgeo.well_from_roxar(
        roxar_project,
        WELLS1[1].replace(".w", ""),
        logrun="log",
        trajectory="My trajectory",
    )
    assert well.dataframe["Facies"].mean() == pytest.approx(0.357798165)
    well.to_roxar(roxar_project, "dummy1", logrun="log", trajectory="My trajectory")
    well.dataframe["Facies"] = np.nan
    assert np.isnan(well.dataframe["Facies"].values).all()
    well.to_roxar(roxar_project, "dummy2", logrun="log", trajectory="My trajectory")
    # check that export with set codes
    well.set_logrecord("Facies", {1: "name"})
    well.to_roxar(roxar_project, "dummy3", logrun="log", trajectory="My trajectory")


@pytest.mark.requires_roxar
@pytest.mark.parametrize(
    "update_option, expected_logs, expected_poroavg",
    [
        (None, ["Poro", "NewPoro"], 0.26376),
        ("overwrite", ["Zonelog", "Perm", "Poro", "Facies", "NewPoro"], 0.26376),
        ("append", ["Zonelog", "Perm", "Poro", "Facies", "NewPoro"], 0.16376),
    ],
)
def test_rox_well_update(roxar_project, update_option, expected_logs, expected_poroavg):
    """Operations on discrete well logs"""
    initial_wellname = WELLS1[1].replace(".w", "")
    wellname = "TESTWELL"

    initial_well = xtgeo.well_from_roxar(
        roxar_project,
        initial_wellname,
        logrun="log",
        lognames="all",
        trajectory="My trajectory",
    )
    initial_well.to_roxar(roxar_project, wellname)

    well = xtgeo.well_from_roxar(
        roxar_project,
        wellname,
        lognames=["Poro"],
    )
    well.create_log("NewPoro")
    well.dataframe["Poro"] += 0.1

    well.to_roxar(
        roxar_project,
        wellname,
        lognames=well.lognames,
        update_option=update_option,
    )
    print("Lognames are", well.lognames)

    rox = xtgeo.RoxUtils(roxar_project)

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
def test_blocked_well_from_to_roxar(roxar_project):
    """Test getting blocked wells from RMS API."""

    rox = xtgeo.RoxUtils(roxar_project)

    bw = xtgeo.blockedwell_from_roxar(
        rox.project, GRIDNAME1, "BW", "OP_2", lognames="all"
    )
    assert list(bw.dataframe.columns) == [
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
    print(bw.dataframe)

    bw.to_roxar(rox.project, GRIDNAME1, "BW", "OP_2")

    # read again from RMS
    bw_2 = xtgeo.blockedwell_from_roxar(
        rox.project, GRIDNAME1, "BW", "OP_2", lognames="all"
    )
    assert "Zonelog" not in list(bw_2.dataframe.columns)
    assert "Some_new" in list(bw_2.dataframe.columns)

    rox.project.close()


@pytest.mark.requires_roxar
def test_blocked_well_roxar_to_from_file(roxar_project, tmp_path):
    """Test getting a single blocked well from RMS, store to file and import again."""

    rox = xtgeo.RoxUtils(roxar_project)

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
def test_blocked_wells_roxar_to_from_file(roxar_project, tmp_path):
    """Test getting blocked wells (plural) from RMS, store to files and import again."""

    rox = xtgeo.RoxUtils(roxar_project)

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
