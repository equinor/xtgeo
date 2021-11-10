# -*- coding: utf-8 -*-

import pandas as pd

import xtgeo

xtg = xtgeo.common.XTGeoDialog()
logger = xtg.basiclogger(__name__)

TPATH = xtg.testpathobj

SMALL1 = TPATH / "3dgrids/etc/TEST_SP.EGRID"
SMALL2 = TPATH / "3dgrids/etc/TEST_DP.EGRID"
SMALL3 = TPATH / "3dgrids/etc/small.roff"
DROGON = TPATH / "3dgrids/drogon/1/geogrid.roff"
BANALCASE1 = TPATH / "3dgrids/etc/banal1.grdecl"
BANALCASE2 = TPATH / "3dgrids/etc/banal2.grdecl"
BANALCASE3 = TPATH / "3dgrids/etc/banal3.grdecl"
QCGRID = TPATH / "3dgrids/etc/gridqc1.roff"

QCFIL1 = TPATH / "3dgrids/etc/gridqc1_rms_cellcenter.csv"
QCFIL2 = TPATH / "3dgrids/etc/gridqc1_rms_anypoint.csv"


def test_get_ijk_from_points_banalcase2():
    """Testing getting IJK coordinates from points on a perfect case case"""
    g1 = xtgeo.grid_from_file(BANALCASE2)

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


def test_get_ijk_from_points_banalcase3():
    """Testing getting IJK coordinates from points on a perfect case case"""
    g1 = xtgeo.grid_from_file(BANALCASE3)

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


def test_get_ijk_from_points_tricky():
    """Testing getting IJK coordinates from points on a tricky case"""
    g1 = xtgeo.grid_from_file(DROGON)

    pointset = [
        (465100.100000, 5931340.000000, 1681.28772),  # 1, 2, 1
    ]

    po = xtgeo.Points(pointset)

    ijk = g1.get_ijk_from_points(po)
    assert ijk["IX"][0] == 110  # 110 171/172
    assert ijk["JY"][0] == 171  # 110 171/172


def test_get_ijk_from_points_smallcase():
    """Testing getting IJK coordinates from points, for all cells in small case"""

    g1 = xtgeo.grid_from_file(SMALL3)

    # g1.crop((1, 1), (1, 1), (1, 2))
    df1 = g1.get_dataframe(ijk=True, xyz=False)
    df2 = g1.get_dataframe(ijk=False, xyz=True)

    po = xtgeo.Points()
    po.dataframe = df2

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
        x = df2["X_UTME"].values[inum]
        y = df2["Y_UTMN"].values[inum]
        z = df2["Z_TVDSS"].values[inum]

        ijkt = tuple((ijk_i[inum], ijk_j[inum], ijk_k[inum]))
        df1t = tuple((df1_i[inum], df1_j[inum], df1_k[inum]))

        if ijkt != df1t:
            notok += 1
            logger.info("%s %s %s: input %s vs output %s", x, y, z, ijkt, df1t)

    fails = notok / allc * 100
    assert fails < 13  # < 0.5% deviation; x_chk_in_cell ~4 % error!


def test_get_ijk_from_points_small():
    """Test IJK getting in small grid, test for active or not cells"""

    g1 = xtgeo.grid_from_file(SMALL1)

    pointset = [
        (1.5, 1.5, 1000.5),  # 2, 2, 1  is active
        (3.5, 2.5, 1000.5),  # 4, 3, 1  is inactive, but dualporo is active
    ]

    po = xtgeo.Points(pointset)

    ijk = g1.get_ijk_from_points(po)

    assert ijk["JY"][0] == 2
    assert ijk["JY"][1] == -1

    # activeonly False
    ijk = g1.get_ijk_from_points(po, activeonly=False)
    assert ijk["JY"][1] == 3

    # dualporo grid
    g1 = xtgeo.grid_from_file(SMALL2)
    ijk = g1.get_ijk_from_points(po, activeonly=False)
    assert ijk["JY"][1] == 3


def test_point_in_cell_compare_rms():
    """Test IJK in cells, compare with a list made in RMS IPL"""

    # from RMS
    pointset = pd.read_csv(QCFIL1, skiprows=3)

    p1 = xtgeo.Points()
    attrs = {"IX": "I", "JY": "J", "KZ": "K"}
    p1.from_dataframe(pointset, east="X", north="Y", tvdmsl="Z", attributes=attrs)

    grd = xtgeo.grid_from_file(QCGRID)
    dfr = grd.get_ijk_from_points(p1)

    for cname in ("IX", "JY", "KZ"):
        list1 = p1.dataframe[cname].tolist()
        list2 = dfr[cname].tolist()

        nall = len(list1)
        suc = 0
        for ino, item in enumerate(list1):
            if item == list2[ino]:
                suc += 1

        succesrate = suc / nall
        print(cname, succesrate, suc, nall)
