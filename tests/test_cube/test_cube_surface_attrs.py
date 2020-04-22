# -*- coding: utf-8 -*-
from os.path import join
import pytest
import xtgeo

xtg = xtgeo.common.XTGeoDialog()
logger = xtg.basiclogger(__name__)

if not xtg.testsetup():
    raise SystemExit("Cannot find test setup")

TMD = xtg.tmpdir
TPATH = xtg.testpath

SFILE1 = join(TPATH, "cubes/etc/ib_synth_iainb.segy")


# ======================================================================================
# This is a a set of tests towards a synthetic small cube made by I Bush in order to
# test all attributes in detail
# ======================================================================================


@pytest.fixture(name="loadsfile1")
def fixture_loadsfile1():
    """Fixture for loading a SFILE1"""
    logger.info("Load seismic file 1")
    return xtgeo.Cube(SFILE1)


def test_single_slice_yflip_positive_snapxy(loadsfile1):
    cube1 = loadsfile1
    cube1.swapaxes()
    samplings = ["nearest", "trilinear"]

    for sampling in samplings:

        surf1 = xtgeo.surface_from_cube(cube1, 1000.0)

        surf1.slice_cube(cube1, sampling=sampling, snapxy=True)

        assert surf1.values.mean() == pytest.approx(cube1.values[0, 0, 0], abs=0.0001)
        print(surf1.values.mean())
        print(cube1.values[0, 0, 0])


# def test_avg_surface(loadsfile1):
#     cube1 = loadsfile1
#     surf1 = xtgeo.surface_from_cube(cube1, 1100.0)
#     surf2 = xtgeo.surface_from_cube(cube1, 2900.0)

#     print(cube1)
#     attrs = surf1.slice_cube_window(
#         cube1,
#         other=surf2,
#         other_position="below",
#         attribute=["max", "min", "mean"],
#         sampling="nearest",
#         snapxy=True,
#         ndiv=4,
#     )

#     for name, val in attrs.items():
#         print(name, val.values.mean())
