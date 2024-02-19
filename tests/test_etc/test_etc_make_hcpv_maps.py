import pathlib

import numpy as np
import numpy.ma as ma
import pytest
import xtgeo
from xtgeo.common import XTGeoDialog
from xtgeo.surface import RegularSurface

# set default level
xtg = XTGeoDialog()

logger = xtg.basiclogger(__name__)

ROFF1_GRID = pathlib.Path("3dgrids/eme/1/emerald_hetero_grid.roff")
ROFF1_PROPS = pathlib.Path("3dgrids/eme/1/emerald_hetero.roff")


@pytest.mark.bigtest
def test_hcpvfz1(tmp_path, generate_plot, testdata_path):
    """HCPV thickness map."""

    # It is important that input are pure numpies, not masked

    logger.info("Name is %s", __name__)
    logger.info("Import roff...")
    grd = xtgeo.grid_from_file(testdata_path / ROFF1_GRID, fformat="roff")

    st = xtgeo.gridproperty_from_file(testdata_path / ROFF1_PROPS, name="Oil_HCPV")
    to = xtgeo.gridproperty_from_file(testdata_path / ROFF1_PROPS, name="Oil_bulk")

    # get the dz and the coordinates, with no mask (ie get value for outside)
    dz = grd.get_dz(asmasked=False)
    xc, yc, _zc = grd.get_xyz(asmasked=False)

    xcv = ma.filled(xc.values3d)
    ycv = ma.filled(yc.values3d)
    dzv = ma.filled(dz.values3d)

    hcpfz = ma.filled(st.values3d, fill_value=0.0)
    tov = ma.filled(to.values3d, fill_value=10)
    tov[tov < 1.0e-32] = 1.0e-32
    hcpfz = hcpfz * dzv / tov

    # make a map... estimate from xc and yc
    xmin = xcv.min()
    xmax = xcv.max()
    ymin = ycv.min()
    ymax = ycv.max()
    xinc = (xmax - xmin) / 50
    yinc = (ymax - ymin) / 50

    logger.debug(
        "xmin xmax ymin ymax, xinc, yinc: %s %s %s %s %s %s",
        xmin,
        xmax,
        ymin,
        ymax,
        xinc,
        yinc,
    )

    hcmap = RegularSurface(
        ncol=50,
        nrow=50,
        xinc=xinc,
        yinc=yinc,
        xori=xmin,
        yori=ymin,
        values=np.zeros((50, 50)),
    )

    hcmap2 = RegularSurface(
        ncol=50,
        nrow=50,
        xinc=xinc,
        yinc=yinc,
        xori=xmin,
        yori=ymin,
        values=np.zeros((50, 50)),
    )

    zp = np.ones((grd.ncol, grd.nrow, grd.nlay))
    # now make hcpf map

    t1 = xtg.timer()
    hcmap.hc_thickness_from_3dprops(
        xprop=xcv,
        yprop=ycv,
        dzprop=dzv,
        hcpfzprop=hcpfz,
        zoneprop=zp,
        zone_minmax=(1, 1),
    )

    assert hcmap.values.mean() == pytest.approx(1.447, abs=0.1)

    t2 = xtg.timer(t1)

    logger.info("Speed basic is %s", t2)

    t1 = xtg.timer()
    hcmap2.hc_thickness_from_3dprops(
        xprop=xcv,
        yprop=ycv,
        dzprop=dzv,
        hcpfzprop=hcpfz,
        zoneprop=zp,
        coarsen=2,
        zone_avg=True,
        zone_minmax=(1, 1),
        mask_outside=True,
    )
    t2 = xtg.timer(t1)

    logger.info("Speed zoneavg coarsen 2 is %s", t2)

    if generate_plot:
        hcmap.quickplot(filename=tmp_path / "quickplot_hcpv.png")
        hcmap2.quickplot(filename=tmp_path / "quickplot_hcpv_zavg_coarsen.png")
