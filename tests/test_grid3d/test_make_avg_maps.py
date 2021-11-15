from os.path import join

import numpy as np
import pytest

import xtgeo

from .ecl_run_fixtures import *  # noqa: F401, F403


def test_avg02(tmpdir, generate_plot, reek_run, testpath):
    """Make average map from Reek Eclipse."""

    # get the poro
    po = reek_run.get_property_from_init(name="PORO")

    # get the dz and the coordinates
    dz = reek_run.grid.get_dz(asmasked=False)
    xc, yc, _zc = reek_run.grid.get_xyz(asmasked=False)

    # get actnum
    actnum = reek_run.grid.get_actnum()

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

    avgmap = xtgeo.RegularSurface(
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
    fau = xtgeo.Polygons(
        join(testpath, "polygons/reek/1/top_upper_reek_faultpoly.zmap"),
        fformat="zmap",
    )
    fspec = {"faults": fau}

    if generate_plot:
        avgmap.quickplot(
            filename=join(tmpdir, "tmp_poro2.png"), xlabelrotation=30, faults=fspec
        )
        avgmap.to_file(join(tmpdir, "tmp.poro.gri"), fformat="irap_ascii")

    assert avgmap.values.mean() == pytest.approx(0.1653, abs=0.01)


def test_avg03(tmpdir, generate_plot, reek_run, testpath):
    """Make average map from Reek Eclipse, speed up by zone_avgrd."""
    # get the poro
    po = reek_run.get_property_from_init(name="PORO")

    # get the dz and the coordinates
    dz = reek_run.grid.get_dz(asmasked=False)
    xc, yc, _zc = reek_run.grid.get_xyz(asmasked=False)

    # get actnum
    actnum = reek_run.grid.get_actnum()
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

    avgmap = xtgeo.RegularSurface(
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
    fau = xtgeo.Polygons(
        join(testpath, "polygons/reek/1/top_upper_reek_faultpoly.zmap"),
        fformat="zmap",
    )
    fspec = {"faults": fau}

    if generate_plot:
        avgmap.quickplot(
            filename=join(tmpdir, "tmp_poro3.png"), xlabelrotation=30, faults=fspec
        )
    avgmap.to_file(join(tmpdir, "tmp.poro3.gri"), fformat="irap_ascii")

    assert avgmap.values.mean() == pytest.approx(0.1653, abs=0.01)
