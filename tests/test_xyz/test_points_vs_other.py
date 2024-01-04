import pathlib

import pytest
import xtgeo

SFILE1A = pathlib.Path("surfaces/reek/1/topupperreek.gri")
SFILE2A = pathlib.Path("surfaces/reek/2/01_topreek_rota.gri")
PFILE3 = pathlib.Path("points/reek/1/pointset3.poi")


def test_snap_to_surface(testpath):
    """Import XYZ points from file."""

    mypoints = xtgeo.points_from_file(testpath / PFILE3)
    assert mypoints.nrow == 20

    surf1 = xtgeo.surface_from_file(testpath / SFILE1A)

    mypoints.snap_surface(surf1)
    assert mypoints.nrow == 11

    assert mypoints.get_dataframe()["Z_TVDSS"].mean() == pytest.approx(
        1661.45, abs=0.01
    )

    # repeat,using surface whithg rotaion and partial masks

    mypoints = xtgeo.points_from_file(testpath / PFILE3)
    surf2 = xtgeo.surface_from_file(testpath / SFILE2A)

    mypoints.snap_surface(surf2)
    assert mypoints.nrow == 12
    assert mypoints.get_dataframe()["Z_TVDSS"].mean() == pytest.approx(
        1687.45, abs=0.01
    )

    # alternative; keep values as is using activeobnly=False
    mypoints = xtgeo.points_from_file(testpath / PFILE3)
    mypoints.snap_surface(surf2, activeonly=False)
    assert mypoints.nrow == 20
    assert mypoints.get_dataframe()["Z_TVDSS"].mean() == pytest.approx(
        1012.47, abs=0.01
    )
