import logging
import pathlib

import numpy as np
import pytest

import xtgeo

logger = logging.getLogger(__name__)

GFILE2 = pathlib.Path("3dgrids/reek/REEK.EGRID")
IFILE2 = pathlib.Path("3dgrids/reek/REEK.INIT")
RFILE2 = pathlib.Path("3dgrids/reek/REEK.UNRST")

FFILE1 = pathlib.Path("polygons/reek/1/top_upper_reek_faultpoly.zmap")


@pytest.mark.benchmark(group="statistical_maps_from_3dprops")
def test_avg02(tmp_path, generate_plot, testdata_path):
    """Make average map from Reek Eclipse."""
    grd = xtgeo.grid_from_file(testdata_path / GFILE2, fformat="egrid")

    # get the poro
    po = xtgeo.gridproperty_from_file(
        testdata_path / IFILE2, fformat="init", name="PORO", grid=grd
    )

    # get the dz and the coordinates
    dz = grd.get_dz(asmasked=False)
    xc, yc, _zc = grd.get_xyz(asmasked=False)

    # get actnum
    actnum = grd.get_actnum()

    # convert from masked numpy to ordinary
    xcuse = np.copy(xc.values)
    ycuse = np.copy(yc.values)
    dzuse = np.copy(dz.values)
    pouse = np.copy(po.values)

    # dz must be zero for undef cells
    dzuse[actnum.values < 0.5] = 0.0
    pouse[actnum.values < 0.5] = 0.0

    # make a map... estimate from xc and yc
    zuse = np.ones((xcuse.shape))

    avgmap = xtgeo.surface_from_grid3d(grd)

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
    fau = xtgeo.polygons_from_file(testdata_path / FFILE1, fformat="zmap")
    fspec = {"faults": fau}

    if generate_plot:
        avgmap.quickplot(
            filename=tmp_path / "tmp_poro2.png", xlabelrotation=30, faults=fspec
        )
        avgmap.to_file(tmp_path / "tmp.poro.gri", fformat="irap_ascii")

    assert avgmap.values.mean() == pytest.approx(0.165, abs=0.001)
    assert avgmap.values.std() == pytest.approx(0.030, abs=0.001)
    assert avgmap.values.min() == pytest.approx(0.072, abs=0.001)
    assert avgmap.values.max() == pytest.approx(0.260, abs=0.001)
    assert avgmap.values.mask.sum() == 2962


@pytest.mark.benchmark(group="statistical_maps_from_3dprops")
def test_avg03(tmp_path, generate_plot, testdata_path):
    """Make average map from Reek Eclipse, speed up by zone_avgrd."""
    grd = xtgeo.grid_from_file(testdata_path / GFILE2, fformat="egrid")

    # get the poro
    po = xtgeo.gridproperty_from_file(
        testdata_path / IFILE2, fformat="init", name="PORO", grid=grd
    )

    # get the dz and the coordinates
    dz = grd.get_dz(asmasked=False)
    xc, yc, _zc = grd.get_xyz(asmasked=False)

    # get actnum
    actnum = grd.actnum_array

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

    avgmap = xtgeo.surface_from_grid3d(grd)

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
    fau = xtgeo.polygons_from_file(testdata_path / FFILE1, fformat="zmap")
    fspec = {"faults": fau}

    if generate_plot:
        avgmap.quickplot(
            filename=tmp_path / "tmp_poro3.png", xlabelrotation=30, faults=fspec
        )
    avgmap.to_file(tmp_path / "tmp.poro3.gri", fformat="irap_ascii")

    assert avgmap.values.mean() == pytest.approx(0.165, abs=0.001)
    assert avgmap.values.std() == pytest.approx(0.030, abs=0.001)
    assert avgmap.values.min() == pytest.approx(0.073, abs=0.001)
    assert avgmap.values.max() == pytest.approx(0.265, abs=0.001)
    assert avgmap.values.mask.sum() == 2969


@pytest.mark.benchmark(group="statistical_maps_from_3dprops")
def test_avg_irregular_zone_layout(testdata_path):
    """Average map from Reek with irregular (non-layered) zone layout."""
    grd = xtgeo.grid_from_file(testdata_path / GFILE2, fformat="egrid")
    po = xtgeo.gridproperty_from_file(
        testdata_path / IFILE2, fformat="init", name="PORO", grid=grd
    )

    dz = grd.get_dz(asmasked=False)
    xc, yc, _zc = grd.get_xyz(asmasked=False)
    actnum = grd.actnum_array

    xcuse = xc.get_npvalues3d()
    ycuse = yc.get_npvalues3d()
    dzuse = dz.get_npvalues3d()
    pouse = po.get_npvalues3d()

    dzuse[actnum < 0.5] = 0.0
    pouse[actnum < 0.5] = 0.0

    ii, jj, kk = np.indices(grd.dimensions)
    zuse = (ii + 2 * jj + 3 * kk) % 7 == 0
    zuse[actnum < 0.5] = 0

    avgmap = xtgeo.surface_from_grid3d(grd)
    avgmap.avg_from_3dprop(
        xprop=xcuse,
        yprop=ycuse,
        zoneprop=zuse,
        zone_minmax=(1, 1),
        mprop=pouse,
        dzprop=dzuse,
        truncate_le=None,
        zone_avg=False,
    )

    # the thickness is weighted by the zone, so the result is in the same
    # range as the regular average maps
    assert avgmap.values.mean() == pytest.approx(0.166, abs=0.001)
    assert avgmap.values.std() == pytest.approx(0.039, abs=0.001)
    assert avgmap.values.min() == pytest.approx(0.047, abs=0.001)
    assert avgmap.values.max() == pytest.approx(0.288, abs=0.001)
    assert avgmap.values.mask.sum() == 2964
