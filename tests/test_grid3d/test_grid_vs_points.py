# -*- coding: utf-8 -*-
from __future__ import division, absolute_import
from __future__ import print_function

import xtgeo

xtg = xtgeo.common.XTGeoDialog()
logger = xtg.basiclogger(__name__)

TDMP = xtg.tmpdir
TESTPATH = xtg.testpath

# =============================================================================
# Do tests
# =============================================================================

REEKGRID = "../xtgeo-testdata/3dgrids/reek/REEK.EGRID"
SMALL1 = "../xtgeo-testdata/3dgrids/etc/TEST_SP.EGRID"
SMALL2 = "../xtgeo-testdata/3dgrids/etc/TEST_DP.EGRID"


def test_get_ijk_from_points():
    """Testing getting IJK coordinates from points"""
    g1 = xtgeo.grid3d.Grid(REEKGRID)

    pointset = [
        (456620.790918, 5.935660e06, 1727.649124),  # 1, 1, 1
        (456620.806270, 5.935660e06, 1744.557755),  # 1, 1, 5
        (467096.108653, 5.930145e06, 1812.760864),  # 40, 64, 14
        (333333, 5555555, 1333),  # outside
        (459168.0442550212, 5931614.347020548, 1715.4637298583984),  # 2, 31, 14
        (464266.1687414392, 5933844.674959661, 1742.2762298583984),  # 36, 35, 11
    ]

    po = xtgeo.Points(pointset)

    ijk = g1.get_ijk_from_points(po, tolerance=0.6)
    print(ijk)

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

    if g1.ijk_handedness == "right":
        g1.ijk_handedness = "left"
        g1._tmp = {}

    ijk = g1.get_ijk_from_points(po)
    print(ijk)

    assert ijk["IX"][0] == 1
    assert ijk["IX"][1] == 1
    assert ijk["IX"][2] == 40

    assert ijk["JY"][0] == 64


def test_get_ijk_from_points_full():
    """Testing getting IJK coordinates from points, for all cells"""

    g1 = xtgeo.grid3d.Grid(REEKGRID)
    df1 = g1.get_dataframe(ijk=True, xyz=False)
    df2 = g1.get_dataframe(ijk=False, xyz=True)

    po = xtgeo.Points()
    po.dataframe = df2
    print(po.dataframe)
    print(df1)

    ijk = g1.get_ijk_from_points(po, includepoints=False)
    print(ijk)

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
    print("Percent failing: ", fails)
    assert fails < 0.5  # < 0.5% deviation; x_chk_in_cell ~4 % error!


def test_get_ijk_from_points_small():
    """Test IJK getting in small grid, test for active or not cells"""

    g1 = xtgeo.grid3d.Grid(SMALL1)

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
    g1 = xtgeo.grid3d.Grid(SMALL2)
    ijk = g1.get_ijk_from_points(po, activeonly=False)
    assert ijk["JY"][1] == 3
