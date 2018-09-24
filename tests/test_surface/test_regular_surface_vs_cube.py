import sys
import pytest
from os.path import join as ojn

import numpy.ma as ma

import xtgeo
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
rpath3 = '../xtgeo-testdata/surfaces/etc'
rpath2 = '../xtgeo-testdata/cubes/reek'
rpath4 = '../xtgeo-testdata/cubes/etc'
rtop1 = ojn(rpath1, '1/topreek_rota.gri')
rbas1 = ojn(rpath1, '1/basereek_rota.gri')
rbas2 = ojn(rpath1, '1/basereek_rota_v2.gri')
rsgy1 = ojn(rpath2, 'syntseis_20000101_seismic_depth_stack.segy')

xtop1 = ojn(rpath3, 'ib_test_horizon2.gri')
xcub1 = ojn(rpath4, 'ib_test_cube2.segy')
xcub2 = ojn(rpath4, 'cube_w_deadtraces.segy')


@tsetup.skipsegyio
@tsetup.skipifroxar
def test_get_surface_from_cube():
    """Construct a constant surface from cube."""

    surf = RegularSurface()
    cube = Cube(rsgy1)

    surf.from_cube(cube, 1999.0)

    assert surf.xinc == cube.xinc
    assert surf.nrow == cube.nrow

    surf.describe()
    cube.describe()


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
    assert mean == pytest.approx(0.0198142, abs=0.001)  # 0.019219 in RMS
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

    xs2.slice_cube_window(cc, attribute='max', sampling='trilinear',
                          showprogress=True)

    xs2.quickplot(filename=td + '/surf_slice_cube_window_max.png',
                  colortable='seismic', minmax=(-1, 1),
                  title='Reek Maximum',
                  infotext='Method: trilinear, window')

    xs3.slice_cube_window(cc, attribute='rms', sampling='trilinear')

    xs3.quickplot(filename=td + '/surf_slice_cube_window_rms.png',
                  colortable='jet', minmax=(0, 1),
                  title='Reek rms (root mean square)',
                  infotext='Method: trilinear, window')


@tsetup.skipsegyio
@tsetup.skipifroxar
def test_slice_attr_window_max_algorithm2():
    """Slice a cube within a window, get max, using trilinear interpol,
    alg 2.
    """

    logger.info('Loading surface')
    xs1 = RegularSurface(rtop1)
    xs2 = xs1.copy()
    xs3 = xs1.copy()

    logger.info('Loading cube')
    cc = Cube(rsgy1)

    t1 = xtg.timer()
    xs1.slice_cube_window(cc, attribute='min', sampling='trilinear',
                          algorithm=2)
    t2 = xtg.timer(t1)
    # logger.info('Window slicing... {} secs'. format(t2))

    # xs1.quickplot(filename=td + '/surf_slice_cube_window_min.png',
    #               colortable='seismic', minmax=(-1, 1),
    #               title='Reek Minimum',
    #               infotext='Method: trilinear, window')

    # xs2.slice_cube_window(cc, attribute='max', sampling='trilinear',
    #                       showprogress=True)

    # xs2.quickplot(filename=td + '/surf_slice_cube_window_max.png',
    #               colortable='seismic', minmax=(-1, 1),
    #               title='Reek Maximum',
    #               infotext='Method: trilinear, window')

    # xs3.slice_cube_window(cc, attribute='rms', sampling='trilinear')

    # xs3.quickplot(filename=td + '/surf_slice_cube_window_rms.png',
    #               colortable='jet', minmax=(0, 1),
    #               title='Reek rms (root mean square)',
    #               infotext='Method: trilinear, window')


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


@tsetup.skipifroxar
def test_cube_attr_rms_two_surfaces_compare_window():
    """Get cube attribute (rms) between two surfaces, and compare with
    window."""

    logger.info('Loading surfaces {} {}'.format(rtop1, rbas1))
    xs1 = RegularSurface(rtop1)
    xs2 = xs1.copy()
    xs2.values += 30

    logger.info('Loading cube {}'.format(rsgy1))
    cc = Cube(rsgy1)

    xss1 = xs1.copy()
    xss1.slice_cube_window(cc, other=xs2, other_position='below',
                           attribute='rms', sampling='trilinear')

    xss1.quickplot(filename=td + '/surf_slice_cube_2surf_rms1.png',
                   colortable='jet', minmax=[0, 0.5],
                   # TODO: itle='Reek two surfs mean', minmax=(-0.1, 0.1),
                   infotext='Method: trilinear, 2 surfs, 30ms apart')

    print('\n\n{}\n'.format('=' * 100))

    xss2 = xs1.copy()
    xss2.values += 15
    xss2.slice_cube_window(cc, zrange=15,
                           attribute='rms', sampling='trilinear')

    xss2.quickplot(filename=td + '/surf_slice_cube_2surf_rms2.png',
                   colortable='jet', minmax=[0, 0.5],
                   # TODO: itle='Reek two surfs mean', minmax=(-0.1, 0.1),
                   infotext='Method: trilinear, 2 surfs, +- 15ms window')

    assert xss1.values.mean() == xss2.values.mean()


@tsetup.skipifroxar
def test_cube_attr_mean_two_surfaces_with_zeroiso():
    """Get cube attribute between two surfaces with partly zero isochore."""

    logger.info('Loading surfaces {} {}'.format(rtop1, rbas1))
    xs1 = RegularSurface(rtop1)
    xs2 = RegularSurface(rbas2)

    logger.info('Loading cube {}'.format(rsgy1))
    cc = Cube(rsgy1)

    xss = xs1.copy()
    xss.slice_cube_window(cc, other=xs2, other_position='below',
                          attribute='mean', sampling='trilinear')

    xss.to_file(td + '/surf_slice_cube_2surf_meantri.gri')

    xss.quickplot(filename=td + '/surf_slice_cube_2surf_mean_v2.png',
                  colortable='jet',
                  title='Reek two surfs mean', minmax=(-0.1, 0.1),
                  infotext='Method: trilinear, 2 surfs, partly zero isochore')

    logger.info('Mean is {}'.format(xss.values.mean()))


@tsetup.skipifroxar
def test_cube_slice_auto4d_data():
    """Get cube slice aka Auto4D input, with synthetic/scrambled data"""

    xs1 = RegularSurface(xtop1, fformat='gri')
    xs1.describe()

    xs1out = ojn(td, 'xtop1.ijxyz')
    xs1.to_file(xs1out, fformat='ijxyz')

    xs2 = RegularSurface(xs1out, fformat='ijxyz')

    assert xs1.values.mean() == pytest.approx(xs2.values.mean(), abs=0.0001)

    cc1 = Cube(xcub1)
    cc1.describe()

    assert xs2.nactive == 10830

    xs2.slice_cube_window(cc1, sampling='trilinear', mask=True,
                          attribute='max')

    xs2out1 = ojn(td, 'xtop2_sampled_from_cube.ijxyz')
    xs2out2 = ojn(td, 'xtop2_sampled_from_cube.gri')
    xs2out3 = ojn(td, 'xtop2_sampled_from_cube.png')

    xs2.to_file(xs2out1, fformat='ijxyz')
    xs2.to_file(xs2out2)

    assert xs2.nactive == 3320  # shall be fewer cells

    xs2.quickplot(filename=xs2out3,
                  colortable='seismic',
                  title='Auto4D Test', minmax=(0, 12000),
                  infotext='Method: max')


@tsetup.skipifroxar
def test_cube_slice_w_ignore_dead_traces_nearest():
    """Get cube slice nearest aka Auto4D input, with scrambled data with
    dead traces, various YFLIP cases, ignore dead traces."""

    cube1 = Cube(xcub2)

    surf1 = RegularSurface()
    surf1.from_cube(cube1, 1000.1)

    cells = ((18, 12), (20, 2), (0, 4))

    surf1.slice_cube(cube1, deadtraces=False)
    plotfile = ojn(td, 'slice_nea1.png')
    title = 'Cube with dead traces; nearest; use just values as is'
    surf1.quickplot(filename=plotfile, minmax=(-10000, 10000), title=title)

    for cell in cells:
        icell, jcell = cell
        assert surf1.values[icell, jcell] == \
            pytest.approx(cube1.values[icell, jcell, 0], abs=0.01)
    assert ma.count_masked(surf1.values) == 0  # shall be no masked cells

    # swap surface
    surf2 = surf1.copy()
    surf2.values = 1000.1
    surf2.swapaxes()

    surf2.slice_cube(cube1, deadtraces=False)
    assert surf2.values.mean() == surf1.values.mean()

    # swap surface and cube
    surf2 = surf1.copy()
    surf2.values = 1000.1
    surf2.swapaxes()

    cube2 = cube1.copy()
    cube2.swapaxes()
    surf2.slice_cube(cube2, deadtraces=False)
    assert surf2.values.mean() == surf1.values.mean()

    # swap cube only
    surf2 = surf1.copy()
    surf2.values = 1000.1

    cube2 = cube1.copy()
    cube2.swapaxes()
    surf2.slice_cube(cube2, deadtraces=False)
    assert surf2.values.mean() == surf1.values.mean()


@tsetup.skipifroxar
def test_cube_slice_w_dead_traces_nearest():
    """Get cube slice nearest aka Auto4D input, with scrambled data with
    dead traces, various YFLIP cases, undef at dead traces."""

    cube1 = Cube(xcub2)

    surf1 = RegularSurface()
    surf1.from_cube(cube1, 1000.1)

    cells = ((18, 12),)

    surf1.slice_cube(cube1, deadtraces=True)
    plotfile = ojn(td, 'slice_nea1_dead.png')
    title = 'Cube with dead traces; nearest; UNDEF at dead traces'
    surf1.quickplot(filename=plotfile, minmax=(-10000, 10000), title=title)

    for cell in cells:
        icell, jcell = cell
        assert surf1.values[icell, jcell] == cube1.values[icell, jcell, 0]

    ndead = (cube1.traceidcodes == 2).sum()
    print(ndead)

    assert ma.count_masked(surf1.values) == ndead

    # swap cube only
    surf2 = surf1.copy()
    surf2.values = 1000.1

    cube2 = cube1.copy()
    cube2.swapaxes()
    surf2.slice_cube(cube2, deadtraces=True)
    plotfile = ojn(td, 'slice_nea1_dead_cubeswap.png')
    surf2.quickplot(filename=plotfile, minmax=(-10000, 10000))
    assert ma.count_masked(surf2.values) == ndead
    assert surf2.values.mean() == surf1.values.mean()


@tsetup.skipifroxar
def test_cube_slice_w_ignore_dead_traces_trilinear():
    """Get cube slice trilinear aka Auto4D input, with scrambled data with
    dead traces to be ignored, various YFLIP cases."""

    cube1 = Cube(xcub2)

    surf1 = RegularSurface()
    surf1.from_cube(cube1, 1000.0)

    cells = [(18, 12), (20, 2), (0, 4)]

    surf1.slice_cube(cube1, sampling='trilinear', snapxy=True,
                     deadtraces=False)
    plotfile = ojn(td, 'slice_tri1.png')
    title = 'Cube with dead traces; trilinear; keep as is at dead traces'
    surf1.quickplot(filename=plotfile, minmax=(-10000, 10000), title=title)

    for cell in cells:
        icell, jcell = cell
        assert surf1.values[icell, jcell] == \
            pytest.approx(cube1.values[icell, jcell, 0], abs=0.1)
    assert ma.count_masked(surf1.values) == 0  # shall be no masked cells


@tsetup.skipifroxar
def test_cube_slice_w_dead_traces_trilinear():
    """Get cube slice trilinear aka Auto4D input, with scrambled data with
    dead traces to be ignored, various YFLIP cases."""

    cube1 = Cube(xcub2)

    surf1 = xtgeo.surface_from_cube(cube1, 1000.0)

    cells = [(18, 12)]

    surf1.slice_cube(cube1, sampling='trilinear', snapxy=True,
                     deadtraces=True)
    plotfile = ojn(td, 'slice_tri1_dead.png')
    title = 'Cube with dead traces; trilinear; UNDEF at dead traces'
    surf1.quickplot(filename=plotfile, minmax=(-10000, 10000), title=title)

    ndead = (cube1.traceidcodes == 2).sum()

    for cell in cells:
        icell, jcell = cell
        assert surf1.values[icell, jcell] == \
            pytest.approx(cube1.values[icell, jcell, 0], 0.1)
    assert ma.count_masked(surf1.values) == ndead

    # swap cube only
    surf2 = surf1.copy()
    surf2.values = 1000.0
    cube2 = cube1.copy()
    cube2.swapaxes()
    surf2.slice_cube(cube2, sampling='trilinear', deadtraces=True)
    plotfile = ojn(td, 'slice_tri1__dead_cubeswap.png')
    surf2.quickplot(filename=plotfile, minmax=(-10000, 10000))
    assert ma.count_masked(surf2.values) == ndead
    assert surf2.values.mean() == surf1.values.mean()


@tsetup.skipifroxar
def test_cube_attr_mean_two_surfaces_multiattr():
    """Get cube attribute (mean) between two surfaces, many attr at the same
    time.
    """

    logger.info('Loading surfaces {} {}'.format(rtop1, rbas1))
    xs1 = RegularSurface(rtop1)
    xs2 = RegularSurface(rbas1)

    logger.info('Loading cube {}'.format(rsgy1))
    cc = Cube(rsgy1)

    xss = xs1.copy()
    xss.slice_cube_window(cc, other=xs2, other_position='below',
                          attribute='rms',
                          sampling='trilinear', showprogress=True)

    logger.debug(xss.values.mean())

    xsx = xs1.copy()
    attrs = xsx.slice_cube_window(cc, other=xs2, other_position='below',
                                  attribute=['max', 'mean', 'min', 'rms'],
                                  sampling='trilinear', showprogress=True)
    logger.debug(attrs['rms'].values.mean())

    assert xss.values.mean() == attrs['rms'].values.mean()
    assert xss.values.std() == attrs['rms'].values.std()

    for attr in attrs.keys():
        logger.info('Working with %s', attr)

        xxx = attrs[attr]
        xxx.to_file(ojn(td, 'surf_slice_cube_2surf_' + attr +
                            'multi.gri'))

        minmax = (None, None)
        if attr == 'mean':
            minmax = (-0.1, 0.1)

        xxx.quickplot(filename=ojn(td, 'surf_slice_cube_2surf_' +
                                   attr + 'multi.png'),
                      colortable='jet', minmax=minmax,
                      title='Reek two surfs mean multiattr: ' + attr,
                      infotext='Method: trilinear, 2 surfs multiattr ' + attr)
