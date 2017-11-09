import sys
import pytest

from xtgeo.surface import RegularSurface
from xtgeo.cube import Cube
from xtgeo.common import XTGeoDialog

xtg = XTGeoDialog()
logger = xtg.basiclogger(__name__)

if not xtg._testsetup():
    sys.exit(-9)

skiplargetest = pytest.mark.skipif(xtg.bigtest is False,
                                   reason="Big tests skip")

td = xtg.tmpdir

# =============================================================================
# Do tests
# =============================================================================
gtop1 = '../xtgeo-testdata/surfaces/gfb/1/gullfaks_top.irapbin'
gsgy1 = '../xtgeo-testdata/cubes/gfb/gf_depth_1985_10_01.segy'

ftop1 = '../xtgeo-testdata/surfaces/fos/2/topaare1.gri'
fsgy1 = '../xtgeo-testdata/cubes/fos/4d_11.segy'

ftop2 = '../xtgeo-testdata/surfaces/fos/2/top_ile_depth.irap'
fsgy2 = '../xtgeo-testdata/cubes/fos/syntseis_2011_seismic_depth.segy'


def test_slice_nearest():
    """Slice a cube with a surface, nearest node."""

    t1 = xtg.timer()
    logger.info('Loading surface')
    xs = RegularSurface(gtop1)

    xs.to_file(td + '/surf_slice_cube_initial.gri')

    logger.info('Loading cube')
    cc = Cube(gsgy1)

    # now slice
    logger.info('Slicing cube which has YFLIP status {}'.format(cc.yflip))

    t1 = xtg.timer()
    print(t1)
    xs.slice_cube(cc)
    t2 = xtg.timer(t1)
    logger.info('Slicing...done in {} seconds'.format(t2))

    xs.to_file(td + '/surf_slice_cube.fgr', fformat='irap_ascii')

    xs.quickplot(filename=td + '/surf_slice_cube.png', colortable='seismic',
                 minmax=(-1, 1), title='Gullfaks', infotext='Method: nearest')

    mean = xs.values.mean()

    logger.info(xs.values.min())
    logger.info(xs.values.max())

    mean = xs.values.mean()
    assert mean == pytest.approx(-0.0755, abs=0.003)
    logger.info('Avg X is {}'.format(mean))

    # try same ting with swapaxes active ==================================
    ys = RegularSurface()
    ys.from_file('../xtgeo-testdata/surfaces/gfb/1/gullfaks_top.irapbin')

    cc.swapaxes()
    # Now slice
    logger.info('Slicing... (now with swapaxes)')
    ys.slice_cube(cc)
    logger.info('Slicing...done')
    mean = ys.values.mean()
    logger.info('Avg for surface is now {}'.format(mean))

    ys.to_file(td + '/surf_slice_cube_swap.gri')
    assert mean == pytest.approx(-0.0755, abs=0.003)


def test_slice_various_fos():
    """Slice a cube with a surface, both nearest node and interpol,
    Fossekall
    """

    logger.info('Loading surface')
    xs = RegularSurface(ftop1)

    logger.info('Loading cube')
    cc = Cube(fsgy1)

    t1 = xtg.timer()
    xs.slice_cube(cc)
    t2 = xtg.timer(t1)
    logger.info('Slicing... nearest, done in {} seconds'.format(t2))

    xs.to_file(td + '/surf_slice_cube_fos_interp.gri')

    xs.quickplot(filename=td + '/surf_slice_cube_fos_interp.png',
                 colortable='seismic',
                 minmax=(-1, 1), title='Fossekall', infotext='Method: nearest')

    # trilinear interpolation:

    logger.info('Loading surface')
    xs = RegularSurface(ftop1)

    t1 = xtg.timer()
    xs.slice_cube(cc, sampling='trilinear')
    t2 = xtg.timer(t1)
    logger.info('Slicing... trilinear, done in {} seconds'.format(t2))

    xs.to_file(td + '/surf_slice_cube_fos_trilinear.gri')

    xs.quickplot(filename=td + '/surf_slice_cube_fos_trilinear.png',
                 colortable='seismic',
                 minmax=(-1, 1), title='Fossekall',
                 infotext='Method: trilinear')


def test_slice_various_fos2():
    """Slice another cube with a surface, nearest node,
    Fossekall alt 2
    """

    logger.info('Loading surface')
    xs = RegularSurface(ftop2)

    logger.info('Loading cube')
    cc = Cube(fsgy2)

    t1 = xtg.timer()
    xs.slice_cube(cc)
    t2 = xtg.timer(t1)
    logger.info('Slicing... nearest, done in {} seconds'.format(t2))

    xs.to_file(td + '/surf_slice_cube_fos_interp_v2.gri')

    xs.quickplot(filename=td + '/surf_slice_cube_fos_interp_v2.png',
                 colortable='seismic',
                 minmax=(-1, 1), title='Fossekall', infotext='Method: nearest')


def test_slice_interpol():
    """Slice a cube with a surface, using trilinear interpol."""

    logger.info('Loading surface')
    x = RegularSurface('../xtgeo-testdata/surfaces/gfb/1/gullfaks_top.irapbin')

    logger.info('Loading cube')
    cc = Cube(gsgy1)
    logger.info('Loading cube, done')

    # Now slice
    logger.info('Slicing...')
    t1 = xtg.timer()
    x.slice_cube(cc, sampling='trilinear')
    t2 = xtg.timer(t1)
    logger.info('Slicing...done trilinear using {} secs'. format(t2))

    logger.info(x.values.min())
    logger.info(x.values.max())

    x.to_file(td + '/surf_slice_cube_interpol.fgr', fformat='irap_ascii')

    x.quickplot(filename=td + '/surf_slice_cube_interpol.png',
                colortable='seismic', minmax=(-1, 1), title='Gullfaks',
                infotext='Method: trilinear')

    logger.info('Avg value is {}'.format(x.values.mean()))
    assert x.values.mean() == pytest.approx(-0.0755, abs=0.003)


def test_slice_attr_window_max():
    """Slice a cube within a surface window, egt max, using trilinear
    interpol.
    """

    logger.info('Loading surface')
    xs1 = RegularSurface(gtop1)
    xs2 = xs1.copy()
    xs3 = xs1.copy()

    logger.info('Loading cube')
    cc = Cube(gsgy1)

    t1 = xtg.timer()
    xs1.slice_cube_window(cc, attribute='min', sampling='trilinear')
    t2 = xtg.timer(t1)
    logger.info('Window slicing... {} secs'. format(t2))

    xs1.quickplot(filename=td + '/surf_slice_cube_window_min.png',
                  colortable='seismic', minmax=(-1, 1),
                  title='Gullfaks Minimum',
                  infotext='Method: trilinear, window')

    xs2.slice_cube_window(cc, attribute='max', sampling='trilinear')

    xs2.quickplot(filename=td + '/surf_slice_cube_window_max.png',
                  colortable='seismic', minmax=(-1, 1),
                  title='Gullfaks Maximum',
                  infotext='Method: trilinear, window')

    xs3.slice_cube_window(cc, attribute='rms', sampling='trilinear')

    xs3.quickplot(filename=td + '/surf_slice_cube_window_rms.png',
                  colortable='jet', minmax=(0, 1),
                  title='Gullfaks rms',
                  infotext='Method: trilinear, window')

# @skiplargetest
# def test_slice2(self):
#     """Slice a larger cube with a surface"""

#     cfile = "/project/gullfaks/resmod/gfmain_brent/2015a/users/eza/"\
#             + "r004/r004_20170303/sim2seis/input/"\
#             + "4D_Res_PSTM_LM_ST8511D11_Full_rmsexport.segy"

#     if os.path.isfile(cfile):

#         logger.info("Loading surface")
#         x = RegularSurface()
#         x.from_file("../../testdata/Surface/G/gullfaks_top.irapbin")

#         zd = RegularSurface()
#         zd.from_file("../../testdata/Surface/G/gullfaks_top.irapbin")

#         logger.info("Loading cube")
#         cc = Cube()
#         cc.from_file(cfile)
#         logger.info("Loading cube, done")

#         # Now slice
#         logger.info("Slicing a large cube...")
#         x.slice_cube(cc, zsurf=zd)
#         logger.info("Slicing a large cube ... done")
#         x.to_file("TMP/surf_slice2_cube.gri")
#     else:
#         logger.warning("No big file; skip test")


# def test_slice_nearest_manytimes():
#     """Slice a cube with a surface, nearest node many times for speed check."""

#     ntimes = 10

#     logger.info("Loading surface")
#     x = RegularSurface()
#     x.from_file("../xtgeo-testdata/surfaces/gfb/1/gullfaks_top.irapbin")

#     logger.info("Loading cube")
#     cc = Cube()
#     cc.from_file("../xtgeo-testdata/cubes/gfb/gf_depth_1985_10_01.segy",
#                  engine=1)
#     # Now slice
#     logger.info("Slicing...")
#     surf = []
#     npcollect = []

#     for i in range(ntimes):
#         surf.append(x.copy())
#         if i < 5:
#             surf[i].values -= i * 2
#         else:
#             surf[i].values += (i - 4) * 2

#         logger.info("Avg depth... {}".format(surf[i].values.mean()))
#         logger.info("Slicing... {}".format(i))
#         surf[i].slice_cube(cc)
#         logger.info("MEAN is {}".format(surf[i].values.mean()))
#         npcollect.append(surf[i].values)

#     logger.info("Slicing...done")

#     # now the numpies are stacked
#     stacked = ma.dstack(npcollect)

#     maxnp = ma.max(stacked, axis=2)

#     x.values = maxnp
#     x.quickplot(filename="TMP/surf_slice_cube_max.png", colortable='seismic',
#                 minmax=(-1, 1))


    # assert mean == pytest.approx(-0.0755, abs=0.003)
