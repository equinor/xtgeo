# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import

import os
from os.path import join
import glob

import matplotlib.pyplot as plt

from xtgeo.plot import XSection
from xtgeo.well import Well
from xtgeo.xyz import Polygons
from xtgeo.surface import RegularSurface
from xtgeo.cube import Cube
from xtgeo.common import XTGeoDialog
import test_common.test_xtg as tsetup

xtg = XTGeoDialog()
logger = xtg.basiclogger(__name__)

if not xtg.testsetup():
    raise SystemExit

td = xtg.tmpdir
testpath = xtg.testpath

xtgshow = False
if 'XTG_SHOW' in os.environ:
    xtgshow = True

logger.info('Use env variable XTG_SHOW to show interactive plots to screen')

# =========================================================================
# Do tests
# =========================================================================

usefile1 = '../xtgeo-testdata/wells/reek/1/OP_1.w'
usefile2 = '../xtgeo-testdata/surfaces/reek/1/topreek_rota.gri'
usefile3 = '../xtgeo-testdata/polygons/reek/1/mypoly.pol'
usefile4 = '../xtgeo-testdata/wells/reek/1/OP_5.w'
usefile5 = '../xtgeo-testdata/surfaces/reek/2/*.gri'
usefile6 = '../xtgeo-testdata/cubes/reek/' +\
           'syntseis_20000101_seismic_depth_stack.segy'

usefile7 = '../xtgeo-testdata/wells/reek/1/OP_2.w'


@tsetup.skipifroxar
def test_very_basic():
    """Just test that matplotlib works."""
    assert 'matplotlib' in str(plt)

    plt.title('Hello world')
    plt.savefig(join(td, 'helloworld1.png'))
    plt.savefig(join(td, 'helloworld1.svg'))
    if xtgshow:
        plt.show()
    logger.info('Very basic plotting')
    plt.close()


@tsetup.skipifroxar
def test_xsection_init():
    """Trigger XSection class, basically."""
    xsect = XSection()
    assert xsect.pagesize == 'A4'


@tsetup.skipifroxar
def test_simple_plot():
    """Test as simple XSECT plot."""

    mywell = Well(usefile4)

    mysurfaces = []
    mysurf = RegularSurface()
    mysurf.from_file(usefile2)

    for i in range(10):
        xsurf = mysurf.copy()
        xsurf.values = xsurf.values + i * 20
        xsurf.name = 'Surface_{}'.format(i)
        mysurfaces.append(xsurf)

    myplot = XSection(zmin=1500, zmax=1800, well=mywell,
                      surfaces=mysurfaces)

    # set the color table, from file
    clist = [0, 1, 222, 3, 5, 7, 3, 12, 11, 10, 9, 8]
    cfil1 = 'xtgeo'
    cfil2 = '../xtgeo-testdata/etc/colortables/colfacies.txt'

    assert 222 in clist
    assert 'xtgeo' in cfil1
    assert 'colfacies' in cfil2

    myplot.set_colortable(cfil1, colorlist=None)

    myplot.canvas(title="Manamana", subtitle="My Dear Well")

    myplot.plot_surfaces(fill=False)

    myplot.plot_well(zonelogname='Zonelog')

    # myplot.plot_map()

    myplot.savefig(join(td, 'xsect_gbf1.png'))

    print('XTGSHOW', xtgshow)
    if xtgshow:
        print('Show plot')
        myplot.show()


@tsetup.skipifroxar
def test_simple_plot_with_seismics():
    """Test as simple XSECT plot with seismic backdrop."""

    mywell = Well(usefile7)
    mycube = Cube(usefile6)

    # mycube.values += 10
    # mycube.values *= 100
    # noise = np.random.normal(0, 10, mycube.values.shape)
    # mycube.values += noise

    mysurfaces = []
    mysurf = RegularSurface()
    mysurf.from_file(usefile2)

    for i in range(10):
        xsurf = mysurf.copy()
        xsurf.values = xsurf.values + i * 20
        xsurf.name = 'Surface_{}'.format(i)
        mysurfaces.append(xsurf)

    myplot = XSection(zmin=1000, zmax=1900, well=mywell,
                      surfaces=mysurfaces, cube=mycube)

    # set the color table, from file
    clist = [0, 1, 222, 3, 5, 7, 3, 12, 11, 10, 9, 8]
    cfil1 = 'xtgeo'
    cfil2 = '../xtgeo-testdata/etc/colortables/colfacies.txt'

    assert 222 in clist
    assert 'xtgeo' in cfil1
    assert 'colfacies' in cfil2

    myplot.set_colortable(cfil1, colorlist=None)

    myplot.canvas(title="Manamana", subtitle="My Dear Well")

    myplot.plot_cube()
    myplot.plot_surfaces(fill=False)

    myplot.plot_well()

    myplot.plot_map()

    myplot.savefig(join(td, 'xsect_wcube.png'), closeplot=False)

    if xtgshow:
        myplot.show()


@tsetup.skipifroxar
def test_reek1():
    """Test XSect for a Reek well."""

    myfield = Polygons()
    myfield.from_file(usefile3, fformat='xyz')

    mywells = []
    wnames = glob.glob(usefile4)
    wnames.sort()
    for wname in wnames:
        mywell = Well(wname)
        mywells.append(mywell)

    logger.info("Wells are read...")

    mysurfaces = []
    surfnames = glob.glob(usefile5)
    surfnames.sort()
    for fname in surfnames:
        logger.info("Read {}".format(fname))
        mysurf = RegularSurface()
        mysurf.from_file(fname)
        mysurfaces.append(mysurf)

    # Troll lobes
    mylobes = []
    surfnames = glob.glob(usefile5)
    surfnames.sort()
    for fname in surfnames:
        logger.info("Read {}".format(fname))
        mysurf = RegularSurface()
        mysurf.from_file(fname)
        mylobes.append(mysurf)

    for wo in mywells:
        myplot = XSection(zmin=1500, zmax=1700, well=wo,
                          surfaces=mysurfaces, zonelogshift=-1,
                          outline=myfield)

        myplot.canvas(title=wo.truewellname,
                      subtitle="Before my corrections",
                      infotext="Heisan sveisan", figscaling=1.2)

        # myplot.colortable('xtgeo')

        myplot.plot_surfaces(fill=True)

        myplot.plot_surfaces(surfaces=mylobes, fill=False, linewidth=4,
                             legendtitle="Lobes", fancyline=True)

        myplot.plot_well(zonelogname='Zonelog')

        myplot.plot_wellmap()

        myplot.plot_map()

        myplot.savefig(join(td, 'xsect2a.svg'), fformat='svg', closeplot=False)
        myplot.savefig(join(td, 'xsect2a.png'), fformat='png')
