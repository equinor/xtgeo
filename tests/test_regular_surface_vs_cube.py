import os
import os.path
import sys
import logging
import warnings
import pytest
import numpy.ma as ma

from xtgeo.surface import RegularSurface
from xtgeo.cube import Cube
from xtgeo.common import XTGeoDialog

path = 'TMP'
try:
    os.makedirs(path)
except OSError:
    if not os.path.isdir(path):
        raise

xtg = XTGeoDialog()

try:
    bigtest = int(os.environ['BIGTEST'])
except Exception:
    bigtest = 0


def custom_formatwarning(msg, *a):
    # ignore everything except the message
    return str(msg) + '\n'


warnings.formatwarning = custom_formatwarning

# =============================================================================
# Do tests
# =============================================================================


def getlogger(name):

    format = xtg.loggingformat

    logging.basicConfig(format=format, stream=sys.stdout)
    logging.getLogger().setLevel(xtg.logginglevel)  # root logger!

    return logging.getLogger(name)


def test_dummy():

    dummy = 1
    assert dummy == 1


def test_slice_nearest():
    """Slice a cube with a surface, nearest node."""

    logger = getlogger('test_slice_nearest')

    logger.info("Loading surface")
    x = RegularSurface()
    x.from_file("../xtgeo-testdata/surfaces/gfb/1/gullfaks_top.irapbin")

    x.to_file("TMP/surf_slice_cube_initial.gri")

    logger.info("Loading cube")
    cc = Cube()
    cc.from_file("../xtgeo-testdata/cubes/gfb/gf_depth_1985_10_01.segy",
                 engine=1)
    # Now slice
    logger.info("Slicing cube which has YFLIP {}".format(cc.yflip))
    x.slice_cube(cc)
    logger.info("Slicing...done")

    x.to_file("TMP/surf_slice_cube.fgr", fformat='irap_ascii')

    x.quickplot(filename="TMP/surf_slice_cube.png", colortable='seismic',
                minmax=(-1, 1))

    mean = x.values.mean()
    logger.info(x.values.min())
    logger.info(x.values.max())
    mean = x.values.mean()
    assert mean == pytest.approx(-0.0755, abs=0.003)
    logger.info("Avg X is {}".format(mean))

    # try same ting with swapaxes active ==================================
    y = RegularSurface()
    y.from_file("../xtgeo-testdata/surfaces/gfb/1/gullfaks_top.irapbin")

    cc.swapaxes()
    # Now slice
    logger.info("Slicing... (now with swapaxes)")
    y.slice_cube(cc)
    logger.info("Slicing...done")
    mean = y.values.mean()
    logger.info("Avg Y is {}".format(mean))

    y.to_file("TMP/surf_slice_cube_swap.gri")
    assert mean == pytest.approx(-0.0755, abs=0.003)


def test_slice_interpol():
    """Slice a cube with a surface, using trilinear interpol."""

    logger = getlogger('test_slice_interpol')

    logger.info("Loading surface")
    x = RegularSurface()
    x.from_file("../xtgeo-testdata/surfaces/gfb/1/gullfaks_top.irapbin")

    logger.info("Loading cube")
    cc = Cube()
    cc.from_file("../xtgeo-testdata/cubes/gfb/gf_depth_1985_10_01.segy",
                 engine=1)
    logger.info("Loading cube, done")

    # Now slice
    logger.info("Slicing...")
    x.slice_cube(cc, sampling=1)
    logger.info("Slicing...done")

    logger.info(x.values.min())
    logger.info(x.values.max())

    x.to_file("TMP/surf_slice_cube_interpol.fgr", fformat='irap_ascii')

    x.quickplot(filename="TMP/surf_slice_cube_interpol.png",
                colortable='seismic', minmax=(-1, 1))

    logger.info('Avg value is {}'.format(x.values.mean()))
    assert x.values.mean() == pytest.approx(-0.0755, abs=0.003)

    def test_slice2(self):
        """
        Slice a larger cube with a surface
        """

        getlogger('test_slice2')

        if bigtest == 0:
            warnings.warn("TEST SKIPPED, enable with env BIGTEST = 1  .... ")
            return

        cfile = "/project/gullfaks/resmod/gfmain_brent/2015a/users/eza/"\
                + "r004/r004_20170303/sim2seis/input/"\
                + "4D_Res_PSTM_LM_ST8511D11_Full_rmsexport.segy"


        if os.path.isfile(cfile):

            logger.info("Loading surface")
            x = RegularSurface()
            x.from_file("../../testdata/Surface/G/gullfaks_top.irapbin")

            zd = RegularSurface()
            zd.from_file("../../testdata/Surface/G/gullfaks_top.irapbin")

            logger.info("Loading cube")
            cc = Cube()
            cc.from_file(cfile)
            logger.info("Loading cube, done")

            # Now slice
            logger.info("Slicing a large cube...")
            x.slice_cube(cc, zsurf=zd)
            logger.info("Slicing a large cube ... done")
            x.to_file("TMP/surf_slice2_cube.gri")
        else:
            logger.warning("No big file; skip test")


def test_slice_nearest_manytimes():
    """Slice a cube with a surface, nearest node many times for speed check."""

    logger = getlogger('test_slice_nearest_manytimes')

    ntimes = 10

    logger.info("Loading surface")
    x = RegularSurface()
    x.from_file("../xtgeo-testdata/surfaces/gfb/1/gullfaks_top.irapbin")

    logger.info("Loading cube")
    cc = Cube()
    cc.from_file("../xtgeo-testdata/cubes/gfb/gf_depth_1985_10_01.segy",
                 engine=1)
    # Now slice
    logger.info("Slicing...")
    surf = []
    npcollect = []

    for i in range(ntimes):
        surf.append(x.copy())
        if i < 5:
            surf[i].values -= i * 2
        else:
            surf[i].values += (i - 4) * 2

        logger.info("Avg depth... {}".format(surf[i].values.mean()))
        logger.info("Slicing... {}".format(i))
        surf[i].slice_cube(cc)
        logger.info("MEAN is {}".format(surf[i].values.mean()))
        npcollect.append(surf[i].values)

    logger.info("Slicing...done")

    # now the numpies are stacked
    stacked = ma.dstack(npcollect)

    maxnp = ma.max(stacked, axis=2)

    x.values = maxnp
    x.quickplot(filename="TMP/surf_slice_cube_max.png", colortable='seismic',
                minmax=(-1, 1))


    # assert mean == pytest.approx(-0.0755, abs=0.003)
