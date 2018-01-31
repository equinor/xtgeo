import sys
import pytest
import os.path

from xtgeo.surface import RegularSurface
from xtgeo.cube import Cube
from xtgeo.common import XTGeoDialog
import test_common.test_xtg as tsetup

xtg = XTGeoDialog()
logger = xtg.basiclogger(__name__)

if not xtg.testsetup():
    sys.exit(-9)

td = xtg.tmpdir

# =============================================================================
# Do tests
# =============================================================================

rpath1 = '../xtgeo-testdata/surfaces/reek'
rpath2 = '../xtgeo-testdata/cubes/reek'
rtop1 = os.path.join(rpath1, '1/topreek_rota.gri')
rbas1 = os.path.join(rpath1, '1/basereek_rota.gri')
rsgy1 = os.path.join(rpath2, 'syntseis_20000101_seismic_depth_stack.segy')


@tsetup.skipsegyio
@tsetup.skipifroxar
def test_slice_nearest():
    """Slice a cube with a surface, nearest node."""

    t1 = xtg.timer()
    logger.info('Loading surface')
    xs = RegularSurface(rtop1)

    xs.to_file(td + '/surf_slice_cube_initial.gri')

    logger.info('Loading cube')
    cc = Cube(rsgy1)

    # now slice
    logger.info('Slicing cube which has YFLIP status {}'.format(cc.yflip))

    t1 = xtg.timer()
    print(t1)
    xs.slice_cube(cc)
    t2 = xtg.timer(t1)
    logger.info('Slicing...done in {} seconds'.format(t2))

    xs.to_file(td + '/surf_slice_cube.fgr', fformat='irap_ascii')
    xs.to_file(td + '/surf_slice_cube.gri', fformat='irap_binary')

    xs.quickplot(filename=td + '/surf_slice_cube.png', colortable='seismic',
                 minmax=(-1, 1), title='Reek', infotext='Method: nearest')

    mean = xs.values.mean()

    logger.info(xs.values.min())
    logger.info(xs.values.max())

    mean = xs.values.mean()
    assert mean == pytest.approx(0.0198142, abs=0.001)
    logger.info('Avg X is {}'.format(mean))

    # try same ting with swapaxes active ==================================
    ys = RegularSurface()
    ys.from_file(rtop1)

    cc.swapaxes()
    # Now slice
    logger.info('Slicing... (now with swapaxes)')
    ys.slice_cube(cc)
    logger.info('Slicing...done')
    mean = ys.values.mean()
    logger.info('Avg for surface is now {}'.format(mean))

    ys.to_file(td + '/surf_slice_cube_swap.gri')
    assert mean == pytest.approx(0.0198142, abs=0.003)


@tsetup.skipsegyio
@tsetup.skipifroxar
def test_slice_various_reek():
    """Slice a cube with a surface, both nearest node and interpol, Reek."""

    logger.info('Loading surface')
    xs = RegularSurface(rtop1)

    logger.info('Loading cube')
    cc = Cube(rsgy1)

    t1 = xtg.timer()
    xs.slice_cube(cc)
    t2 = xtg.timer(t1)
    logger.info('Slicing... nearest, done in {} seconds'.format(t2))

    xs.to_file(td + '/surf_slice_cube_reek_interp.gri')

    xs.quickplot(filename=td + '/surf_slice_cube_reek_interp.png',
                 colortable='seismic',
                 minmax=(-1, 1), title='Reek', infotext='Method: nearest')

    # trilinear interpolation:

    logger.info('Loading surface')
    xs = RegularSurface(rtop1)

    t1 = xtg.timer()
    xs.slice_cube(cc, sampling='trilinear')
    t2 = xtg.timer(t1)
    logger.info('Slicing... trilinear, done in {} seconds'.format(t2))

    xs.to_file(td + '/surf_slice_cube_reek_trilinear.gri')

    xs.quickplot(filename=td + '/surf_slice_cube_reek_trilinear.png',
                 colortable='seismic',
                 minmax=(-1, 1), title='Reek',
                 infotext='Method: trilinear')



@tsetup.skipsegyio
@tsetup.skipifroxar
def test_slice_attr_window_max():
    """Slice a cube within a window, get max, using trilinear interpol."""

    logger.info('Loading surface')
    xs1 = RegularSurface(rtop1)
    xs2 = xs1.copy()
    xs3 = xs1.copy()

    logger.info('Loading cube')
    cc = Cube(rsgy1)

    t1 = xtg.timer()
    xs1.slice_cube_window(cc, attribute='min', sampling='trilinear')
    t2 = xtg.timer(t1)
    logger.info('Window slicing... {} secs'. format(t2))

    xs1.quickplot(filename=td + '/surf_slice_cube_window_min.png',
                  colortable='seismic', minmax=(-1, 1),
                  title='Reek Minimum',
                  infotext='Method: trilinear, window')

    xs2.slice_cube_window(cc, attribute='max', sampling='trilinear')

    xs2.quickplot(filename=td + '/surf_slice_cube_window_max.png',
                  colortable='seismic', minmax=(-1, 1),
                  title='Reek Maximum',
                  infotext='Method: trilinear, window')

    xs3.slice_cube_window(cc, attribute='rms', sampling='trilinear')

    xs3.quickplot(filename=td + '/surf_slice_cube_window_rms.png',
                  colortable='jet', minmax=(0, 1),
                  title='Reek rms (root mean square)',
                  infotext='Method: trilinear, window')


@tsetup.skipifroxar
def test_cube_attr_mean_two_surfaces():
    """Get cube attribute (mean) between two surfaces."""

    logger.info('Loading surfaces {} {}'.format(rtop1, rbas1))
    xs1 = RegularSurface(rtop1)
    xs2 = RegularSurface(rbas1)

    logger.info('Loading cube {}'.format(rsgy1))
    cc = Cube(rsgy1)

    xss = xs1.copy()
    xss.slice_cube_window(cc, other=xs2, other_position='below',
                          attribute='mean', sampling='trilinear')

    xss.to_file(td + '/surf_slice_cube_2surf_meantri.gri')

    xss.quickplot(filename=td + '/surf_slice_cube_2surf_mean.png',
                  colortable='jet',
                  title='Reek two surfs mean', minmax=(-0.1, 0.1),
                  infotext='Method: trilinear, 2 surfs')

    logger.info('Mean is {}'.format(xss.values.mean()))
