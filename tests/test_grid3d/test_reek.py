import pathlib
from collections import OrderedDict
from os.path import join

import numpy as np
import pytest

import xtgeo

from .ecl_run_fixtures import *  # noqa: F401, F403


# pylint: disable=redefined-outer-name
@pytest.fixture()
def reek_path(testpath):
    return pathlib.Path(testpath) / "3dgrids/reek"


@pytest.fixture()
def reek_grid(reek_run):
    return reek_run.grid


def test_roffbin_import_v2_wsubgrids(reek_path):
    assert xtgeo.grid_from_file(
        reek_path / "reek_geo2_grid_3props.roff"
    ).subgrids == OrderedDict(
        [
            ("subgrid_0", range(1, 21)),
            ("subgrid_1", range(21, 41)),
            ("subgrid_2", range(41, 57)),
        ]
    )


def test_import_grdecl_and_bgrdecl(testpath):
    """Eclipse import of GRDECL and binary GRDECL."""
    grd1 = xtgeo.grid_from_file(
        testpath + "/3dgrids/reek3/reek_sim.grdecl", fformat="grdecl"
    )
    grd2 = xtgeo.grid_from_file(
        testpath + "/3dgrids/reek3/reek_sim.bgrdecl", fformat="bgrdecl"
    )

    assert grd1.dimensions == (40, 64, 14)
    assert grd1.nactive == 35812

    assert grd2.dimensions == (40, 64, 14)
    assert grd2.nactive == 35812

    np.testing.assert_allclose(grd1.get_dz().values, grd2.get_dz().values, atol=0.001)


def test_eclgrid_import2(tmp_path, reek_grid):
    """Eclipse EGRID import, also change ACTNUM."""

    assert reek_grid.ncol == 40, "EGrid NX from Eclipse"
    assert reek_grid.nrow == 64, "EGrid NY from Eclipse"
    assert reek_grid.nactive == 35838, "EGrid NTOTAL from Eclipse"
    assert reek_grid.ntotal == 35840, "EGrid NACTIVE from Eclipse"

    actnum = reek_grid.get_actnum()
    assert actnum.values[12, 22, 5] == 0, "ACTNUM 0"

    actnum.values[:, :, :] = 1
    actnum.values[:, :, 4:6] = 0
    reek_grid.set_actnum(actnum)
    newactive = reek_grid.ncol * reek_grid.nrow * reek_grid.nlay - 2 * (
        reek_grid.ncol * reek_grid.nrow
    )
    assert reek_grid.nactive == newactive, "Changed ACTNUM"
    reek_grid.to_file(tmp_path / "reek_new_actnum.roff")


def test_geometrics_reek(reek_grid):
    """Import Reek and test geometrics."""
    geom = reek_grid.get_geometrics(return_dict=True, cellcenter=False)

    # compared with RMS info:
    assert geom["xmin"] == pytest.approx(456510.6, abs=0.1), "Xmin"
    assert geom["ymax"] == pytest.approx(5938935.5, abs=0.1), "Ymax"

    # cellcenter True:
    geom = reek_grid.get_geometrics(return_dict=True, cellcenter=True)
    assert geom["xmin"] == pytest.approx(456620, abs=1), "Xmin cell center"


@pytest.mark.parametrize("fformat", ["grdecl", "roff", "egrid"])
def test_grid_roundtrip_reek(tmp_path, reek_grid, fformat):
    reek_grid.to_file(tmp_path / f"grid.{fformat}", fformat=fformat)
    reek_grid2 = xtgeo.grid_from_file(tmp_path / f"grid.{fformat}", fformat=fformat)

    assert reek_grid2.dimensions == (40, 64, 14)

    np.testing.assert_allclose(reek_grid._coordsv, reek_grid2._coordsv)
    assert np.array_equal(reek_grid._actnumsv, reek_grid2._actnumsv)
    np.testing.assert_allclose(reek_grid._zcornsv, reek_grid2._zcornsv)


def test_grid_layer_slice(reek_grid):
    """Test grid slice coordinates."""

    sarr1, _ibarr = reek_grid.get_layer_slice(1)
    sarrn, _ibarr = reek_grid.get_layer_slice(reek_grid.nlay, top=False)

    cell1 = reek_grid.get_xyz_cell_corners(ijk=(1, 1, 1))
    celln = reek_grid.get_xyz_cell_corners(ijk=(1, 1, reek_grid.nlay))
    celll = reek_grid.get_xyz_cell_corners(ijk=reek_grid.dimensions)

    assert sarr1[0, 0, 0] == cell1[0]
    assert sarr1[0, 0, 1] == cell1[1]

    assert sarrn[0, 0, 0] == celln[12]
    assert sarrn[0, 0, 1] == celln[13]

    assert sarrn[-1, 0, 0] == celll[12]
    assert sarrn[-1, 0, 1] == celll[13]


def test_get_ijk_from_points(reek_grid):
    """Testing getting IJK coordinates from points"""
    pointset = [
        (456620.790918, 5.935660e06, 1727.649124),  # 1, 1, 1
        (456620.806270, 5.935660e06, 1744.557755),  # 1, 1, 5
        (467096.108653, 5.930145e06, 1812.760864),  # 40, 64, 14
        (333333, 5555555, 1333),  # outside
        (459168.0442550212, 5931614.347020548, 1715.4637298583984),  # 2, 31, 14
        (464266.1687414392, 5933844.674959661, 1742.2762298583984),  # 36, 35, 11
    ]

    po = xtgeo.Points(pointset)

    ijk = reek_grid.get_ijk_from_points(po)

    assert ijk["IX"][0] == 1
    assert ijk["IX"][1] == 1
    assert ijk["IX"][2] == 40

    assert ijk["JY"][0] == 1

    assert ijk["KZ"][0] == 1
    assert ijk["KZ"][1] == 5
    assert ijk["KZ"][2] == 14

    assert ijk["KZ"][3] == -1
    assert ijk["KZ"][4] == 14
    assert ijk["KZ"][5] == 11

    if reek_grid.ijk_handedness == "right":
        reek_grid.ijk_handedness = "left"
        reek_grid._tmp = {}

    ijk = reek_grid.get_ijk_from_points(po)

    assert ijk["IX"][0] == 1
    assert ijk["IX"][1] == 1
    assert ijk["IX"][2] == 40

    assert ijk["JY"][0] == 64


def test_get_ijk_from_points_full(reek_grid):
    """Testing getting IJK coordinates from points, for all cells"""

    df1 = reek_grid.get_dataframe(ijk=True, xyz=False)
    df2 = reek_grid.get_dataframe(ijk=False, xyz=True)

    po = xtgeo.Points()
    po.dataframe = df2

    ijk = reek_grid.get_ijk_from_points(po, includepoints=False)

    ijk_i = ijk["IX"].values.tolist()
    ijk_j = ijk["JY"].values.tolist()
    ijk_k = ijk["KZ"].values.tolist()

    df1_i = df1["IX"].values.tolist()
    df1_j = df1["JY"].values.tolist()
    df1_k = df1["KZ"].values.tolist()

    notok = 0
    allc = 0

    for inum, _val in enumerate(ijk_i):
        allc += 1

        ijkt = tuple((ijk_i[inum], ijk_j[inum], ijk_k[inum]))
        df1t = tuple((df1_i[inum], df1_j[inum], df1_k[inum]))

        if ijkt != df1t:
            notok += 1

    fails = notok / allc * 100
    assert fails < 0.5  # < 0.5% deviation; x_chk_in_cell ~4 % error!


def test_get_dataframe(reek_grid, reek_path):
    """Get a Pandas dataframe from the gridproperties"""

    x = xtgeo.GridProperties()

    names = ["SOIL", "SWAT", "PRESSURE"]
    dates = [19991201]
    x.from_file(
        reek_path / "REEK.UNRST",
        fformat="unrst",
        names=names,
        dates=dates,
        grid=reek_grid,
    )
    df = x.dataframe(activeonly=True, ijk=True, xyz=False)

    assert df["SWAT_19991201"].mean() == pytest.approx(0.87802, abs=0.001)
    assert df["PRESSURE_19991201"].mean() == pytest.approx(334.523, abs=0.005)


def test_values_in_polygon(reek_grid, reek_path, testpath):
    """Test replace values in polygons"""

    xprop = xtgeo.gridproperty_from_file(
        join(testpath, "3dgrids/reek/reek_sim_poro.roff"),
        fformat="roff",
        name="PORO",
        grid=reek_grid,
    )
    poly = xtgeo.Polygons(join(testpath, "polygons/reek/1/polset2.pol"))
    xprop.geometry = reek_grid
    xorig = xprop.copy()

    xprop.operation_polygons(poly, 99, inside=True)
    assert xprop.values.mean() == pytest.approx(25.1788, abs=0.01)

    xp2 = xorig.copy()
    xp2.values *= 100
    xp2.continuous_to_discrete()
    xp2.set_inside(poly, 44)

    xp2.dtype = np.uint8
    xp2.set_inside(poly, 44)
    print(xp2.values)

    xp2.dtype = np.uint16
    xp2.set_inside(poly, 44)
    print(xp2.values)

    xp3 = xorig.copy()
    xp3.values *= 100
    print(xp3.values.mean())
    xp3.dtype = np.float32
    xp3.set_inside(poly, 44)
    print(xp3.values.mean())

    assert xp3.values.mean() == pytest.approx(23.40642788381048, abs=0.001)


def test_create_from_grid(reek_grid):
    """Create a simple property from grid"""

    poro = xtgeo.GridProperty(reek_grid, name="poro", values=0.33)
    assert poro.ncol == reek_grid.ncol

    assert poro.isdiscrete is False
    assert poro.values.mean() == 0.33

    assert poro.values.dtype.kind == "f"

    faci = xtgeo.GridProperty(reek_grid, name="FAC", values=1, discrete=True)
    assert faci.nlay == reek_grid.nlay
    assert faci.values.mean() == 1

    assert faci.values.dtype.kind == "i"

    some = xtgeo.GridProperty(reek_grid, name="SOME")
    assert some.isdiscrete is False
    some.values = np.where(some.values == 0, 0, 1)
    assert some.isdiscrete is False


def test_create_from_gridproperty(reek_grid):
    """Create a simple property from grid"""

    poro = xtgeo.GridProperty(reek_grid, name="poro", values=0.33)
    assert poro.ncol == reek_grid.ncol

    # create from gridproperty
    faci = xtgeo.GridProperty(poro, name="FAC", values=1, discrete=True)
    assert faci.nlay == reek_grid.nlay
    assert faci.values.mean() == 1

    assert faci.values.dtype.kind == "i"

    some = xtgeo.GridProperty(faci, name="SOME", values=22)
    assert some.values.mean() == 22.0
    assert some.isdiscrete is False
    some.values = np.where(some.values == 0, 0, 1)
    assert some.isdiscrete is False
