"""Testing regular surface vs resampling."""
from os.path import join

import numpy as np
import pytest

import xtgeo
from xtgeo.common import logger
from xtgeo.common.xtgeo_dialog import testdatafolder
from xtgeo.surface import RegularSurface
from xtgeo.xyz import Points

TPATH = testdatafolder

# =============================================================================
# Do tests
# =============================================================================
FTOP1 = TPATH / "surfaces/reek/1/topreek_rota.gri"


@pytest.fixture(name="reek_map")
def fixture_reek_map():
    """Fixture for map input."""
    logger.info("Loading surface")
    return xtgeo.surface_from_file(FTOP1)


def test_resample_small():
    """Do resampling with minimal dataset to test for various yflip etc."""
    xs1 = RegularSurface(
        xori=0,
        yori=0,
        ncol=3,
        nrow=3,
        xinc=100,
        yinc=100,
        values=-888.0,
        yflip=1,
    )
    xs2 = RegularSurface(
        xori=0,
        yori=0,
        ncol=3,
        nrow=3,
        xinc=100,
        yinc=100,
        values=888.0,
        yflip=1,
    )
    xs3 = RegularSurface(
        xori=0,
        yori=200,
        ncol=3,
        nrow=3,
        xinc=100,
        yinc=100,
        values=2888.0,
        yflip=-1,
    )

    xsx = xs1.copy()
    xsx.resample(xs2)
    assert list(xsx.values.data.flatten()) == [
        888.0,
        888.0,
        888.0,
        888.0,
        888.0,
        888.0,
        888.0,
        888.0,
        888.0,
    ]

    xsx = xs3.copy()
    xsx.resample(xs2)
    assert list(xsx.values.data.flatten()) == [
        888.0,
        888.0,
        888.0,
        888.0,
        888.0,
        888.0,
        888.0,
        888.0,
        888.0,
    ]

    xsx = xs1.copy()
    xsx.resample(xs3)
    assert list(xsx.values.data.flatten()) == [
        2888.0,
        2888.0,
        2888.0,
        2888.0,
        2888.0,
        2888.0,
        2888.0,
        2888.0,
        2888.0,
    ]


def test_resample_small_nearest_sampling():
    """Do resampling with minimal dataset to test resmapling with nearest option."""
    xs1 = RegularSurface(
        xori=0,
        yori=0,
        ncol=3,
        nrow=3,
        xinc=100,
        yinc=100,
        values=np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]),
        yflip=1,
    )
    xs2 = RegularSurface(
        xori=0,
        yori=0,
        ncol=3,
        nrow=3,
        xinc=100,
        yinc=100,
        values=np.array([0, 0, 0, 444, 0, 0, 0, 0, 0]),
        yflip=1,
    )
    xs3 = RegularSurface(
        xori=100,
        yori=100,
        ncol=3,
        nrow=3,
        xinc=50,
        yinc=50,
        values=888.0,
        yflip=1,
    )
    xsx = xs1.copy()
    xsx.resample(xs2, sampling="nearest")
    assert list(xsx.values.data.flatten()) == [
        0.0,
        0.0,
        0.0,
        444.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    xsx = xs1.copy()
    xsx.resample(xs3, sampling="nearest")
    assert list(xsx.values.data.flatten()) == [
        1e33,
        1e33,
        1e33,
        1e33,
        888.0,
        888.0,
        1e33,
        888.0,
        888.0,
    ]


def test_resample(tmpdir, reek_map):
    """Do resampling from one surface to another."""
    xs = reek_map
    assert xs.ncol == 554

    xs_copy = xs.copy()

    # create a new map instance, unrotated, based on this map
    ncol = int((xs.xmax - xs.xmin) / 10)
    nrow = int((xs.ymax - xs.ymin) / 10)
    values = np.zeros((nrow, ncol))
    snew = RegularSurface(
        xori=xs.xmin,
        xinc=10,
        yori=xs.ymin,
        yinc=10,
        nrow=nrow,
        ncol=ncol,
        values=values,
    )

    snew.resample(xs)

    fout = join(tmpdir, "reek_resampled.gri")
    snew.to_file(fout, fformat="irap_binary")

    assert snew.values.mean() == pytest.approx(1698.458, abs=2)
    assert snew.values.mean() == pytest.approx(xs.values.mean(), abs=2)

    # check that the "other" in snew.resample(other) is unchanged:
    assert xs.xinc == xs_copy.xinc
    np.testing.assert_allclose(xs.values, xs_copy.values, atol=1e-4)


def test_resample_nearest(reek_map):
    """Do resampling from one surface to another, using nearest node."""
    xs = reek_map
    assert xs.ncol == 554

    xs_copy = xs.copy()

    # turn into a discrete map
    xs_copy.values[xs.values > 1700] = 888.0
    xs_copy.values[xs.values <= 1700] = 0.0

    # create a new map instance, unrotated, based on this map with corase sampling
    ncol = int((xs.xmax - xs.xmin) / 200)
    nrow = int((xs.ymax - xs.ymin) / 200)
    values = np.zeros((nrow, ncol))
    snew = RegularSurface(
        xori=xs.xmin,
        xinc=200,
        yori=xs.ymin,
        yinc=200,
        nrow=nrow,
        ncol=ncol,
        values=values,
    )

    snew.resample(xs_copy, sampling="nearest")

    assert list(np.unique(snew.values))[0:-1] == [0.0, 888.0]  # 2 values only
    assert snew.values.mean() == pytest.approx(xs_copy.values.mean(), abs=2)


def test_resample_partial_sample(tmp_path, reek_map, generate_plot):
    """Do resampling from one surface to another with partial sampling."""
    sml = reek_map

    # note: values is missing by purpose:
    snew = RegularSurface(
        ncol=round(sml.ncol * 0.6),
        nrow=round(sml.nrow * 0.6),
        xori=sml.xori - 2000,
        yori=sml.yori + 3000,
        xinc=sml.xinc * 0.6,
        yinc=sml.xinc * 0.6,
        rotation=sml.rotation,
        yflip=sml.yflip,
    )

    print(sml.yflip)

    logger.info(snew.values)

    snew.resample(sml, mask=True)

    assert snew.values.mean() == pytest.approx(1726.65)

    if generate_plot:
        sml.quickplot(tmp_path / "resampled_input.png")
        snew.quickplot(tmp_path / "resampled_output.png")
        sml.to_file(tmp_path / "resampled_input.gri")
        snew.to_file(tmp_path / "resampled_output.gri")

    snew2 = snew.copy()
    snew2._yflip = -1
    snew2._xori -= 4000
    snew2._yori -= 2000
    snew2.resample(sml, mask=True)

    if generate_plot:
        snew2.to_file(tmp_path / "resampled_output2.gri")
    assert snew2.values.mean() == pytest.approx(1747.20, abs=0.2)


def test_refine(tmpdir, reek_map, generate_plot):
    """Do refining of a surface."""
    xs = reek_map
    assert xs.ncol == 554

    xs_orig = xs.copy()
    xs.refine(4)

    fout = join(tmpdir, "reek_refined.gri")
    xs.to_file(fout, fformat="irap_binary")

    assert xs_orig.values.mean() == pytest.approx(xs.values.mean(), abs=0.8)

    if generate_plot:
        logger.info("Output plots to file (may be time consuming)")
        xs_orig.quickplot(filename=join(tmpdir, "reek_orig.png"))
        xs.quickplot(filename=join(tmpdir, "reek_refined4.png"))


def test_coarsen(tmpdir, reek_map, generate_plot):
    """Do a coarsening of a surface."""
    xs = reek_map
    assert xs.ncol == 554

    xs_orig = xs.copy()
    xs.coarsen(3)

    fout = join(tmpdir, "reek_coarsened.gri")
    xs.to_file(fout, fformat="irap_binary")

    assert xs_orig.values.mean() == pytest.approx(xs.values.mean(), abs=0.8)

    if generate_plot:
        logger.info("Output plots to file (may be time consuming)")
        xs_orig.quickplot(filename=join(tmpdir, "reek_orig.png"))
        xs.quickplot(filename=join(tmpdir, "reek_coarsen3.png"))


@pytest.mark.bigtest
def test_points_gridding(tmpdir, reek_map, generate_plot):
    """Make points of surface; then grid back to surface."""
    xs = reek_map
    assert xs.ncol == 554

    xyz = Points(xs)

    xyz.dataframe["Z_TVDSS"] = xyz.dataframe["Z_TVDSS"] + 300

    logger.info("Avg of points: %s", xyz.dataframe["Z_TVDSS"].mean())

    xscopy = xs.copy()

    logger.info(xs.values.flags)
    logger.info(xscopy.values.flags)

    # now regrid
    xscopy.gridding(xyz, coarsen=1)  # coarsen will speed up test a lot

    if generate_plot:
        logger.info("Output plots to file (may be time consuming)")
        xs.quickplot(filename=join(tmpdir, "s1.png"))
        xscopy.quickplot(filename=join(tmpdir, "s2.png"))

    np.testing.assert_allclose(xscopy.values, xs.values + 300, atol=2)

    xscopy.to_file(join(tmpdir, "reek_points_to_map.gri"), fformat="irap_binary")
