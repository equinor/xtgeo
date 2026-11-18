import logging
import pathlib

import numpy as np
import numpy.ma as ma
import pytest

import xtgeo

logger = logging.getLogger(__name__)

ROFF1_GRID = pathlib.Path("3dgrids/eme/1/emerald_hetero_grid.roff")
ROFF1_PROPS = pathlib.Path("3dgrids/eme/1/emerald_hetero.roff")


@pytest.mark.benchmark(group="statistical_maps_from_3dprops")
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

    xcv = ma.filled(xc.values)
    ycv = ma.filled(yc.values)
    dzv = ma.filled(dz.values)

    hcpfz = ma.filled(st.values, fill_value=0.0)
    tov = ma.filled(to.values, fill_value=10)
    tov[tov < 1.0e-32] = 1.0e-32
    hcpfz = hcpfz * dzv / tov

    hcmap = xtgeo.surface_from_grid3d(grd)
    hcmap2 = xtgeo.surface_from_grid3d(grd)

    zp = np.ones((grd.ncol, grd.nrow, grd.nlay))
    # now make hcpf map

    hcmap.hc_thickness_from_3dprops(
        xprop=xcv,
        yprop=ycv,
        dzprop=dzv,
        hcpfzprop=hcpfz,
        zoneprop=zp,
        zone_minmax=(1, 1),
    )

    vals1 = ma.array(hcmap.values)
    assert vals1.mean() == pytest.approx(1.354, abs=0.01)
    assert vals1.std() == pytest.approx(3.38, abs=0.01)
    assert vals1.min() == 0
    assert vals1.max() == pytest.approx(23.4, abs=0.01)

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

    vals2 = ma.array(hcmap2.values)
    assert vals2.mean() == pytest.approx(3.87, abs=0.01)
    assert vals2.std() == pytest.approx(4.75, abs=0.01)
    assert vals2.min() == 0.0
    assert vals2.max() == pytest.approx(23.4, abs=0.01)
    assert vals2.mask.sum() == 9309

    if generate_plot:
        hcmap.quickplot(filename=tmp_path / "quickplot_hcpv.png")
        hcmap2.quickplot(filename=tmp_path / "quickplot_hcpv_zavg_coarsen.png")


@pytest.mark.benchmark(group="statistical_maps_from_3dprops")
def test_hcpvfz_irregular_zone_layout(testdata_path):
    """HCPV thickness map with irregular (non-layered) zone layout."""
    grd = xtgeo.grid_from_file(testdata_path / ROFF1_GRID, fformat="roff")

    st = xtgeo.gridproperty_from_file(testdata_path / ROFF1_PROPS, name="Oil_HCPV")
    to = xtgeo.gridproperty_from_file(testdata_path / ROFF1_PROPS, name="Oil_bulk")

    dz = grd.get_dz(asmasked=False)
    xc, yc, _zc = grd.get_xyz(asmasked=False)

    xcv = ma.filled(xc.values)
    ycv = ma.filled(yc.values)
    dzv = ma.filled(dz.values)

    hcpfz = ma.filled(st.values, fill_value=0.0)
    tov = ma.filled(to.values, fill_value=10)
    tov[tov < 1.0e-32] = 1.0e-32
    hcpfz = hcpfz * dzv / tov

    # Make an irregular 3D zone mask (not following full K-layers)
    ii, jj, kk = np.indices((grd.ncol, grd.nrow, grd.nlay))
    zp_irregular = (ii + 2 * jj + 3 * kk) % 5 == 0
    zp_irregular[grd.actnum_array == 0] = 0

    hcmap = xtgeo.surface_from_grid3d(grd)
    hcmap2 = xtgeo.surface_from_grid3d(grd)

    hcmap.hc_thickness_from_3dprops(
        xprop=xcv,
        yprop=ycv,
        dzprop=dzv,
        hcpfzprop=hcpfz,
        zoneprop=zp_irregular,
        zone_minmax=(1, 1),
    )

    vals1 = ma.array(hcmap.values)
    assert vals1.mean() == pytest.approx(0.27, abs=0.01)
    assert vals1.std() == pytest.approx(0.68, abs=0.01)
    assert vals1.min() == 0
    assert vals1.max() == pytest.approx(4.94, abs=0.01)

    hcmap2.hc_thickness_from_3dprops(
        xprop=xcv,
        yprop=ycv,
        dzprop=dzv,
        hcpfzprop=hcpfz,
        zoneprop=zp_irregular,
        coarsen=2,
        zone_avg=True,
        zone_minmax=(1, 1),
        mask_outside=True,
    )

    vals2 = ma.array(hcmap2.values)
    assert vals2.mean() == pytest.approx(0.85, abs=0.01)
    assert vals2.std() == pytest.approx(0.98, abs=0.01)
    assert vals2.min() == 0.0
    assert vals2.max() == pytest.approx(4.72, abs=0.01)
    assert vals2.mask.sum() == 9815
