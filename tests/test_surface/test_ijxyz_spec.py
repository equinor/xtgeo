from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

import numpy as np
import pytest

import xtgeo
from xtgeo.surface._regsurf_ijxyz_parser import SurfaceIJXYZ

if TYPE_CHECKING:
    from pathlib import Path

import logging

logger = logging.getLogger(__name__)


IJXYZFILE1 = pathlib.Path("surfaces/etc/ijxyz1.dat")  # OW IJXYZ format with comments
OTHERFILE1 = pathlib.Path("surfaces/reek/1/topreek_rota.gri")  # for template test


SIMPLE_VALUES1 = [[2, 3, 4, 5], [6, 5, 4, 99], [44.0, 33.0, 22.0, 11.0]]
SIMPLE_MASK1 = [
    [True, False, False, False],
    [True, False, True, False],
    [False, False, True, False],
]

SIMPLE_VALUES2 = range(300)
SIMPLE_MASK2 = [0, 1, 4, 5, 19, 95, 96, 97, 121]

# real file(s)


@pytest.fixture(name="simple1")
def fixture_simple1(tmp_path: Path):
    """Test reading small simple case."""
    surf = xtgeo.RegularSurface(
        3,
        4,
        55.0,
        133.0,
        xori=12345.0,
        yori=998877.0,
        values=np.ma.MaskedArray(SIMPLE_VALUES1, mask=SIMPLE_MASK1),
        rotation=33,
        yflip=1,
    )
    dfr = surf.get_dataframe(ij=True)
    target = tmp_path / "ijxyz.dat"
    dfr.to_csv(target, sep=" ", index=False, header=False)

    return (surf, np.loadtxt(target, comments=["@", "#", "EOB"]))


@pytest.fixture(name="simple2")
def fixture_simple2(tmp_path: Path):
    """Test somewhat larger case"""

    values = np.array(SIMPLE_VALUES2)
    values = np.ma.masked_where(np.isin(np.arange(values.size), SIMPLE_MASK2), values)

    surf = xtgeo.RegularSurface(
        30,
        10,
        50.0,
        100.0,
        xori=10000.0,
        yori=90000.0,
        values=values,
        rotation=0,
        yflip=1,
        ilines=np.array(range(233, 233 + 90, 3)),
        xlines=np.array(range(44, 44 + 20, 2)),
    )
    dfr = surf.get_dataframe(ij=True)
    ix = dfr.IX - 1
    jy = dfr.JY - 1
    ilines = surf.ilines[ix]
    xlines = surf.xlines[jy]
    dfr["IL"] = ilines
    dfr["XL"] = xlines
    dfr = dfr[["IL", "XL", "X_UTME", "Y_UTMN", "VALUES"]]
    target = tmp_path / "ijxyz2.dat"
    dfr.to_csv(target, sep=" ", index=False, header=False)

    return (surf, np.loadtxt(target, comments=["@", "#", "EOB"]))


def test_simple1_ijxyz_read(simple1: tuple):
    surf, data = simple1  # truth surf, and data arrays read from file
    inline = data[:, 0].astype("int32")
    xline = data[:, 1].astype("int32")
    x_arr = data[:, 2].astype("float64")
    y_arr = data[:, 3].astype("float64")
    z_arr = data[:, 4].astype("float64")

    ijxyz = SurfaceIJXYZ(x_arr, y_arr, z_arr, inline, xline)

    assert ijxyz.yflip == surf.yflip
    assert ijxyz.xori == pytest.approx(surf.xori)
    assert ijxyz.ncol == surf.ncol
    assert ijxyz.nrow == surf.nrow
    assert ijxyz.xinc == pytest.approx(surf.xinc)
    assert ijxyz.yinc == pytest.approx(surf.yinc)
    np.testing.assert_array_almost_equal(ijxyz.values, surf.values)
    np.testing.assert_array_almost_equal(ijxyz.values.mask, surf.values.mask)


def test_simple2_ijxyz_read(simple2: tuple):
    surf, data = simple2  # truth surf, and data arrays read from file
    inline = data[:, 0].astype("int32")
    xline = data[:, 1].astype("int32")
    x_arr = data[:, 2].astype("float64")
    y_arr = data[:, 3].astype("float64")
    z_arr = data[:, 4].astype("float64")

    ijxyz = SurfaceIJXYZ(x_arr, y_arr, z_arr, inline, xline)

    assert ijxyz.yflip == surf.yflip
    assert ijxyz.xori == pytest.approx(surf.xori)
    assert ijxyz.ncol == surf.ncol
    assert ijxyz.nrow == surf.nrow
    assert ijxyz.xinc == pytest.approx(surf.xinc)
    assert ijxyz.yinc == pytest.approx(surf.yinc)

    np.testing.assert_array_almost_equal(ijxyz.values, surf.values)
    np.testing.assert_array_almost_equal(ijxyz.values.mask, surf.values.mask)

    np.testing.assert_array_almost_equal(ijxyz.ilines, surf.ilines)
    np.testing.assert_array_almost_equal(ijxyz.xlines, surf.xlines)


def test_ijxyzfile1_read(testdata_path):
    """Test a real file exported from OW"""
    data = np.loadtxt(testdata_path / IJXYZFILE1, comments=["@", "#", "EOB"])
    inline = data[:, 0].astype("int32")
    xline = data[:, 1].astype("int32")
    x_arr = data[:, 2].astype("float64")
    y_arr = data[:, 3].astype("float64")
    z_arr = data[:, 4].astype("float64")

    ijxyz = SurfaceIJXYZ(x_arr, y_arr, z_arr, inline, xline)

    assert ijxyz.values.mean() == pytest.approx(5037.5840, abs=0.001)
    assert ijxyz.ncol == 51
    assert ijxyz.yflip == -1


def test_ijxyz_io_file_with_template(tmp_path, testdata_path):
    """Test i/o with and without template"""

    surf0 = xtgeo.surface_from_file(testdata_path / OTHERFILE1)

    surf0.ilines += 22
    surf0.ilines *= 2

    surf0.xlines += 33
    surf0.xlines *= 3

    file1 = tmp_path / "other_x1.dat"
    surf0.to_file(file1, fformat="ijxyz")

    surf1 = xtgeo.surface_from_file(file1, fformat="ijxyz")

    # assertions here are intentional; with undef cells, ijxy cannot reproduce
    # ncol, nrow and xori, yori for original file due to lack of information
    assert surf1.ncol == surf0.ncol - 29
    assert surf1.nrow == surf0.nrow - 9
    assert surf1.rotation == pytest.approx(surf0.rotation)
    assert surf1.xinc == pytest.approx(surf0.xinc)
    assert surf1.yinc == pytest.approx(surf0.yinc)
    assert surf1.xori != pytest.approx(surf0.xori)
    assert surf1.yori != pytest.approx(surf0.yori)

    # try ijxyz with template instead; here assertions are equal
    surf2 = xtgeo.surface_from_file(file1, fformat="ijxyz", template=surf0)
    assert surf2.ncol == surf0.ncol
    assert surf2.nrow == surf0.nrow
    assert surf2.rotation == pytest.approx(surf0.rotation)
    assert surf2.xinc == pytest.approx(surf0.xinc)
    assert surf2.yinc == pytest.approx(surf0.yinc)
    assert surf2.xori == pytest.approx(surf0.xori)
    assert surf2.yori == pytest.approx(surf0.yori)
    np.testing.assert_array_equal(surf0.ilines, surf2.ilines)
    np.testing.assert_array_equal(surf0.xlines, surf2.xlines)

    np.testing.assert_array_almost_equal(surf0.values, surf2.values)
    assert surf1.values.mean() == pytest.approx(surf2.values.mean())
