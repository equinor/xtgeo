import pathlib

import pandas as pd
import pytest

import xtgeo

REEKGRID = pathlib.Path("3dgrids/reek/REEK.EGRID")
SMALL1 = pathlib.Path("3dgrids/etc/TEST_SP.EGRID")
SMALL2 = pathlib.Path("3dgrids/etc/TEST_DP.EGRID")
SMALL3 = pathlib.Path("3dgrids/etc/small.roff")
DROGON = pathlib.Path("3dgrids/drogon/1/geogrid.roff")
BANALCASE1 = pathlib.Path("3dgrids/etc/banal1.grdecl")
BANALCASE2 = pathlib.Path("3dgrids/etc/banal2.grdecl")
BANALCASE3 = pathlib.Path("3dgrids/etc/banal3.grdecl")
QCGRID = pathlib.Path("3dgrids/etc/gridqc1.roff")

QCFIL1 = pathlib.Path("3dgrids/etc/gridqc1_rms_cellcenter.csv")
QCFIL2 = pathlib.Path("3dgrids/etc/gridqc1_rms_anypoint.csv")


@pytest.fixture
def reekgrid(testdata_path):
    """Fixture for loading a grid (pytest setup)"""
    return xtgeo.grid_from_file(testdata_path / REEKGRID)


@pytest.fixture
def smallgrid1(testdata_path):
    """Fixture for loading a grid (pytest setup)"""
    return xtgeo.grid_from_file(testdata_path / SMALL1)


@pytest.fixture
def smallgrid2(testdata_path):
    """Fixture for loading a grid (pytest setup)"""
    return xtgeo.grid_from_file(testdata_path / SMALL2)


@pytest.fixture
def smallgrid3(testdata_path):
    """Fixture for loading a grid (pytest setup)"""
    return xtgeo.grid_from_file(testdata_path / SMALL3)


@pytest.fixture
def drogongrid(testdata_path):
    """Fixture for loading a grid (pytest setup)"""
    return xtgeo.grid_from_file(testdata_path / DROGON)


@pytest.fixture
def banalcase1(testdata_path):
    """Fixture for loading a grid (pytest setup)"""
    return xtgeo.grid_from_file(testdata_path / BANALCASE1)


@pytest.fixture
def banalcase2(testdata_path):
    """Fixture for loading a grid (pytest setup)"""
    return xtgeo.grid_from_file(testdata_path / BANALCASE2)


@pytest.fixture
def banalcase3(testdata_path):
    """Fixture for loading a grid (pytest setup)"""
    return xtgeo.grid_from_file(testdata_path / BANALCASE3)


@pytest.fixture
def qcgrid(testdata_path):
    """Fixture for loading a grid (pytest setup)"""
    return xtgeo.grid_from_file(testdata_path / QCGRID)


def test_get_ijk_from_points_banalcase2(banalcase2):
    """Testing getting IJK coordinates from points on a perfect case case"""
    g1 = banalcase2

    pointset = [
        (50, 50, -0.01),  # -1, -1, -1
        (50, 50, 0.000),  # 1, 1, 1
        (50, 50, 0.990),  # 1, 1, 1
        (50, 50, 1.200),  # 1, 1, 1  # could be 1,1,2
        (50, 50, 1.400),  # 1, 1, 1  # could be 1,1,2
        (50, 50, 1.600),  # 1, 1, 1  # could be 1,1,2
        (50, 50, 1.900),  # 1, 1, 1  # could be 1,1,2
        (50, 50, 2.100),  # 1, 1, 3
        (50, 50, 2.600),  # 1, 1, 3
    ]

    po = xtgeo.Points(pointset)
    ijk = g1.get_ijk_from_points(po)

    assert ijk["KZ"][2] == 1

    pointset = [
        (25, 25, -0.01),  # -1, -1, -1
        (25, 25, 0.000),  # 1, 1, 1
        (25, 25, 0.990),  # 1, 1, 1
        (25, 25, 1.200),  # 1, 1, 2
        (25, 25, 1.400),  # 1, 1, 2
        (25, 25, 1.600),  # 1, 1, 2
        (25, 25, 1.900),  # 1, 1, 2
        (25, 25, 2.100),  # 1, 1, 3
        (25, 25, 2.600),  # 1, 1, 3
    ]

    po = xtgeo.Points(pointset)

    ijk = g1.get_ijk_from_points(po)
    assert ijk["KZ"][5] == 2

    pointset = [
        (0, 0, -0.01),  # -1, -1, -1
        (0, 0, 0.000),  # 1, 1, 1
        (0, 0, 0.990),  # 1, 1, 1
        (0, 0, 1.200),  # 1, 1, 2
        (0, 0, 1.400),  # 1, 1, 2
        (0, 0, 1.600),  # 1, 1, 2
        (0, 0, 1.900),  # 1, 1, 2
        (0, 0, 2.100),  # 1, 1, 3
        (0, 0, 2.600),  # 1, 1, 3
    ]

    po = xtgeo.Points(pointset)

    ijk = g1.get_ijk_from_points(po)
    assert ijk["KZ"][7] == 3


def test_get_ijk_from_points_banalcase3(banalcase3):
    """Testing getting IJK coordinates from points on a perfect case case"""
    g1 = banalcase3

    pointset = [
        (50, 50, -0.01),  # outside
        (50, 50, 0.000),  # 1, 1, 1
        (50, 50, 0.990),  # 1, 1, 1
        (50, 50, 1.200),  # 1, 1, 1
        (50, 50, 1.400),  # 1, 1, 1
        (50, 50, 1.600),  # 1, 1, 2
        (50, 50, 1.900),  # 1, 1, 2
        (50, 50, 2.100),  # 1, 1, 3
        (50, 50, 2.600),  # 1, 1, 3
    ]

    po = xtgeo.Points(pointset)

    ijk = g1.get_ijk_from_points(po)

    assert ijk["KZ"][7] == 3


@pytest.mark.bigtest
def test_get_ijk_from_points_tricky(drogongrid):
    """Testing getting IJK coordinates from points on a tricky case"""
    g1 = drogongrid

    pointset = [
        (465100.100000, 5931340.000000, 1681.28772),  # 1, 2, 1
    ]

    po = xtgeo.Points(pointset)

    ijk = g1.get_ijk_from_points(po)
    assert ijk["IX"][0] == 110  # 110 171/172
    assert ijk["JY"][0] == 172  # 110 171/172


def test_get_ijk_from_points(reekgrid):
    """Testing getting IJK coordinates from points"""

    pointset = [
        (456620.790918, 5.935660e06, 1727.649124),  # 1, 1, 1
        (456620.806270, 5.935660e06, 1744.557755),  # 1, 1, 5
        (467096.108653, 5.930145e06, 1812.760864),  # 40, 64, 14
        (333333, 5555555, 1333),  # outside
        (459168.0442550212, 5931614.347020548, 1715.4637298583984),  # 2, 31, 14
        (
            464266.1687414392,
            5933844.674959661,
            1742.2762298583984,
        ),  # 36, 35, 11/12 boundary!
    ]

    po = xtgeo.Points(pointset)

    ijk = reekgrid.get_ijk_from_points(po)

    assert ijk["IX"][0] == 1
    assert ijk["IX"][1] == 1
    assert ijk["IX"][2] == 40

    assert ijk["JY"][0] == 1

    assert ijk["KZ"][0] == 1
    assert ijk["KZ"][1] == 5
    assert ijk["KZ"][2] == 14

    assert ijk["KZ"][3] == -1
    assert ijk["KZ"][4] == 14
    assert ijk["KZ"][5] == 11  #  11/12 boundary!

    if reekgrid.ijk_handedness == "right":
        reekgrid.ijk_handedness = "left"

    ijk = reekgrid.get_ijk_from_points(po)

    assert ijk["IX"][0] == 1
    assert ijk["IX"][1] == 1
    assert ijk["IX"][2] == 40

    assert ijk["JY"][0] == 64


def test_get_ijk_from_points_small(smallgrid1, smallgrid2):
    """Test IJK getting in small grid, test for active or not cells"""

    pointset = [
        (1.5, 1.5, 1000.5),  # 2, 2, 1  is active
        (3.5, 2.5, 1000.5),  # 4, 3, 1  is inactive, but dualporo is active
    ]

    po = xtgeo.Points(pointset)

    ijk = smallgrid1.get_ijk_from_points(po)

    assert ijk["JY"][0] == 2
    assert ijk["JY"][1] == -1

    # activeonly False
    ijk = smallgrid1.get_ijk_from_points(po, activeonly=False)
    assert ijk["JY"][1] == 3

    # dualporo grid
    ijk = smallgrid2.get_ijk_from_points(po, activeonly=False)
    assert ijk["JY"][1] == 3


def test_get_ijk_from_points_smallcase(smallgrid3):
    """Testing getting IJK coordinates from points, for all cells in small case"""

    g1 = smallgrid3

    df1 = g1.get_dataframe(ijk=True, xyz=False)
    df2 = g1.get_dataframe(ijk=False, xyz=True)

    po = xtgeo.Points()
    po.set_dataframe(df2)

    ijk = g1.get_ijk_from_points(po, includepoints=False)

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
        ijkt = (ijk_i[inum], ijk_j[inum], ijk_k[inum])
        df1t = (df1_i[inum], df1_j[inum], df1_k[inum])

        if ijkt != df1t:
            notok += 1

    fails = notok / allc * 100

    assert fails < 13  # < 0.5% deviation; x_chk_in_cell ~4 % error!


@pytest.mark.bigtest
def test_get_ijk_from_points_full(reekgrid):
    """Testing getting IJK coordinates from points, for all cells"""

    g1 = reekgrid
    df1 = g1.get_dataframe(ijk=True, xyz=False)
    df2 = g1.get_dataframe(ijk=False, xyz=True)

    po = xtgeo.Points()
    po.set_dataframe(df2)

    ijk = g1.get_ijk_from_points(po, includepoints=False)

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

        ijkt = (ijk_i[inum], ijk_j[inum], ijk_k[inum])
        df1t = (df1_i[inum], df1_j[inum], df1_k[inum])

        if ijkt != df1t:
            notok += 1

    fails = notok / allc * 100
    assert fails < 0.5  # < 0.5% deviation; x_chk_in_cell ~4 % error!


def test_point_in_cell_compare_rms(qcgrid, testdata_path):
    """Test IJK in cells, compare with a list made in RMS IPL"""

    # from RMS
    pointset = pd.read_csv(testdata_path / QCFIL1, skiprows=3)

    attrs = {"I": "int", "J": "int", "K": "int"}
    p1 = xtgeo.Points(
        values=pointset, xname="X", yname="Y", zname="Z", attributes=attrs
    )
    grd = qcgrid
    dfr = grd.get_ijk_from_points(p1)

    for cname, cxname in {"I": "IX", "J": "JY", "K": "KZ"}.items():
        list1 = p1.get_dataframe(copy=False)[cname].tolist()
        list2 = dfr[cxname].tolist()

        nall = len(list1)
        suc = 0
        for ino, item in enumerate(list1):
            if item == list2[ino]:
                suc += 1

        succesrate = suc / nall
        assert succesrate > 0.99, (
            f"Success rate: {succesrate:.2f} for {cname} ({suc}/{nall})"
        )
