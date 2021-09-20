import pathlib

import xtgeo
import tests.test_common.test_xtg as tsetup

SFILE1A = pathlib.Path("surfaces/reek/1/topupperreek.gri")
SFILE2A = pathlib.Path("surfaces/reek/2/01_topreek_rota.gri")
PFILE3 = pathlib.Path("points/reek/1/pointset3.poi")


def test_snap_to_surface(testpath):
    """Import XYZ points from file."""

    mypoints = xtgeo.Points(testpath / PFILE3)
    assert mypoints.nrow == 20

    surf1 = xtgeo.surface_from_file(testpath / SFILE1A)

    mypoints.snap_surface(surf1)
    assert mypoints.nrow == 11

    tsetup.assert_almostequal(mypoints.dataframe["Z_TVDSS"].mean(), 1661.45, 0.01)

    # repeat,using surface whithg rotaion and partial masks

    mypoints = xtgeo.Points(testpath / PFILE3)
    surf2 = xtgeo.surface_from_file(testpath / SFILE2A)

    mypoints.snap_surface(surf2)
    assert mypoints.nrow == 12
    tsetup.assert_almostequal(mypoints.dataframe["Z_TVDSS"].mean(), 1687.45, 0.01)

    # alternative; keep values as is using activeobnly=False
    mypoints = xtgeo.Points(testpath / PFILE3)
    mypoints.snap_surface(surf2, activeonly=False)
    assert mypoints.nrow == 20
    tsetup.assert_almostequal(mypoints.dataframe["Z_TVDSS"].mean(), 1012.47, 0.01)
