"""Tests 3D grids bytesio input/output for supported formats.

Currently tested format(s):
  * roff binary
"""
import io

import numpy as np

import xtgeo
from xtgeo.common import XTGeoDialog

xtg = XTGeoDialog()

if not xtg.testsetup():
    raise SystemExit

TPATH = xtg.testpathobj

BANAL6 = TPATH / "3dgrids/etc/banal6.roff"


def test_roff_binary_grid_bytesio_read():
    """Test reading ROFF binary from memory streams"""

    grd1 = xtgeo.grid_from_file(BANAL6)

    with open(BANAL6, "rb") as fhandle:
        stream = io.BytesIO(fhandle.read())
    stream.seek(0)

    grd2 = xtgeo.grid_from_file(stream, fformat="roff")
    assert grd2.dimensions == grd1.dimensions
    np.testing.assert_array_equal(grd2._zcornsv, grd1._zcornsv)


def test_roff_binary_grid_bytesio_read_write():
    """Test reading and writing ROFF binary to memory streams"""

    stream = io.BytesIO()
    grd1 = xtgeo.create_box_grid(dimension=(2, 3, 4))
    grd1.to_file(stream, fformat="roff")
    stream.seek(0)

    grd2 = xtgeo.grid_from_file(stream, fformat="roff")
    assert grd2.dimensions == grd1.dimensions
    np.testing.assert_array_equal(grd2._zcornsv, grd1._zcornsv)
