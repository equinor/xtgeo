from os.path import join

import numpy as np
import pytest

from xtgeo.common import XTGeoDialog
from xtgeo.grid3d import Grid, GridProperty
from xtgeo.surface import RegularSurface
from xtgeo.xyz import Polygons

# set default level
xtg = XTGeoDialog()

logger = xtg.basiclogger(__name__)
TPATH = xtg.testpathobj

# ======================================================================================
# This tests a combination of methods, in order to produce maps of HC thickness
# ======================================================================================
# gfile1 = '../xtgeo-testdata/3dgrids/bri/B.GRID'
# ifile1 = '../xtgeo-testdata/3dgrids/bri/B.INIT'

GFILE2 = TPATH / "3dgrids/reek/REEK.EGRID"
IFILE2 = TPATH / "3dgrids/reek/REEK.INIT"
RFILE2 = TPATH / "3dgrids/reek/REEK.UNRST"

FFILE1 = TPATH / "polygons/reek/1/top_upper_reek_faultpoly.zmap"


@pytest.mark.skipifroxar
def test_avg02(tmpdir, generate_plot):
    """Make average map from Reek Eclipse."""
    grd = Grid()
    grd.from_file(GFILE2, fformat="egrid")

    # get the poro
    po = GridProperty()
    po.from_file(IFILE2, fformat="init", name="PORO", grid=grd)

    # get the dz and the coordinates
    dz = grd.get_dz(asmasked=False)
    xc, yc, _zc = grd.get_xyz(asmasked=False)

    # get actnum
    actnum = grd.get_actnum()

    # convert from masked numpy to ordinary
    xcuse = np.copy(xc.values3d)
    ycuse = np.copy(yc.values3d)
    dzuse = np.copy(dz.values3d)
    pouse = np.copy(po.values3d)

    # dz must be zero for undef cells
    dzuse[actnum.values3d < 0.5] = 0.0
    pouse[actnum.values3d < 0.5] = 0.0

    # make a map... estimate from xc and yc
    zuse = np.ones((xcuse.shape))

    avgmap = RegularSurface(
        ncol=200,
        nrow=250,
        xinc=50,
        yinc=50,
        xori=457000,
        yori=5927000,
        values=np.zeros((200, 250)),
    )

    avgmap.avg_from_3dprop(
        xprop=xcuse,
        yprop=ycuse,
        zoneprop=zuse,
        zone_minmax=(1, 1),
        mprop=pouse,
        dzprop=dzuse,
        truncate_le=None,
    )

    # add the faults in plot
    fau = Polygons(FFILE1, fformat="zmap")
    fspec = {"faults": fau}

    if generate_plot:
        avgmap.quickplot(
            filename=join(tmpdir, "tmp_poro2.png"), xlabelrotation=30, faults=fspec
        )
        avgmap.to_file(join(tmpdir, "tmp.poro.gri"), fformat="irap_ascii")

    logger.info(avgmap.values.mean())
    assert avgmap.values.mean() == pytest.approx(0.1653, abs=0.01)


@pytest.mark.skipifroxar
def test_avg03(tmpdir):
    """Make average map from Reek Eclipse, speed up by zone_avgrd."""
    grd = Grid()
    grd.from_file(GFILE2, fformat="egrid")

    # get the poro
    po = GridProperty()
    po.from_file(IFILE2, fformat="init", name="PORO", grid=grd)

    # get the dz and the coordinates
    dz = grd.get_dz(asmasked=False)
    xc, yc, _zc = grd.get_xyz(asmasked=False)

    # get actnum
    actnum = grd.get_actnum()
    actnum = actnum.get_npvalues3d()

    # convert from masked numpy to ordinary
    xcuse = xc.get_npvalues3d()
    ycuse = yc.get_npvalues3d()
    dzuse = dz.get_npvalues3d(fill_value=0.0)
    pouse = po.get_npvalues3d(fill_value=0.0)

    # dz must be zero for undef cells
    dzuse[actnum < 0.5] = 0.0
    pouse[actnum < 0.5] = 0.0

    # make a map... estimate from xc and yc
    zuse = np.ones((xcuse.shape))

    avgmap = RegularSurface(
        ncol=200,
        nrow=250,
        xinc=50,
        yinc=50,
        xori=457000,
        yori=5927000,
        values=np.zeros((200, 250)),
    )

    avgmap.avg_from_3dprop(
        xprop=xcuse,
        yprop=ycuse,
        zoneprop=zuse,
        zone_minmax=(1, 1),
        mprop=pouse,
        dzprop=dzuse,
        truncate_le=None,
        zone_avg=True,
    )

    # add the faults in plot
    fau = Polygons(FFILE1, fformat="zmap")
    fspec = {"faults": fau}

    avgmap.quickplot(
        filename=join(tmpdir, "tmp_poro3.png"), xlabelrotation=30, faults=fspec
    )
    avgmap.to_file(join(tmpdir, "tmp.poro3.gri"), fformat="irap_ascii")

    logger.info(avgmap.values.mean())
    assert avgmap.values.mean() == pytest.approx(0.1653, abs=0.01)
