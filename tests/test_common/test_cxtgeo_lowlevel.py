"""Test some basic cxtgeo functions."""
import numpy as np
import pytest

import xtgeo
import xtgeo.cxtgeo._cxtgeo as _cxtgeo


@pytest.mark.parametrize(
    "depth, logseries, distance, expected",
    [
        (
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [1, 1, 1, 2, 2, 2],
            1.2,
            [False, False, True, True, False, False],
        ),
        (
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [1, 1, 1, 2, 2, 2],
            1.9,
            [False, True, True, True, True, False],
        ),
        (
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [1, 1, 1, 2, 2, 2],
            2.6,
            [True, True, True, True, True, True],
        ),
        (
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [1, 1, 1, 2, 2, 2],
            0.1,
            [False, False, False, False, False, False],
        ),
        (
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [1, 1, 1, 2, 3, 2],
            0.6,
            [False, False, True, True, True, True],
        ),
        (
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [1, 1, 1, 2, np.nan, np.nan],
            0.6,
            [False, False, True, True, False, False],
        ),
        (
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [1, 1, 1, 2, np.nan, np.nan],
            1.6,
            [False, True, True, True, False, False],
        ),
    ],
)
def test_well_mask_shoulder(depth, logseries, distance, expected):
    """Test well_mask_shoulder from cxtgeo."""

    depth = np.array(depth, dtype="float64")

    logseries = np.array(logseries, dtype="float64")
    try:
        logseries = np.nan_to_num(logseries, nan=xtgeo.UNDEF_INT).astype("int32")
    except TypeError:
        # for older numpy version
        logseries[np.isnan(logseries)] = xtgeo.UNDEF_INT
        logseries = logseries.astype("int32")

    print(logseries)

    expected = np.array(expected)

    mask = np.zeros(depth.size, dtype="int32")
    res = _cxtgeo.well_mask_shoulder(depth, logseries, mask, distance)

    mask = mask.astype("bool")
    print(mask)
    assert res == 0
    assert (mask == expected).all()


@pytest.mark.parametrize(
    "xcoords, ycoords, zcoords, xval, yval, method, zexpected",
    [
        (
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 2.0, 3.0, 4.0],
            0.5,
            0.5,
            2,  # bilinear, nonrotated
            2.5,
        ),
        (
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 2.0, 3.0, 4.0],
            0.0,
            0.5,
            2,  # bilinear, nonrotated
            2.0,
        ),
        (
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 2.0, 3.0, 4.0],
            0.4999,
            0.4999,
            4,  # nearest node
            1.0,
        ),
        (
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 2.0, 3.0, 4.0],
            0.50001,
            0.49999,
            4,  # nearest node
            2.0,
        ),
        (
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 2.0, 3.0, 4.0],
            0.49999,
            0.50001,
            4,  # nearest node
            3.0,
        ),
        (
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 2.0, 3.0, 4.0],
            0.50000,
            0.50000,
            4,  # nearest node
            [1.0, 2.0, 3.0, 4.0],  # singularity, can be any
        ),
    ],
)
def test_x_interp_map_nodes(xcoords, ycoords, zcoords, xval, yval, method, zexpected):
    xv = _cxtgeo.new_doublearray(4)
    yv = _cxtgeo.new_doublearray(4)
    zv = _cxtgeo.new_doublearray(4)

    for num, xvalue in enumerate(xcoords):
        _cxtgeo.doublearray_setitem(xv, num, xvalue)
    for num, yvalue in enumerate(ycoords):
        _cxtgeo.doublearray_setitem(yv, num, yvalue)
    for num, zvalue in enumerate(zcoords):
        _cxtgeo.doublearray_setitem(zv, num, zvalue)

    zresult = _cxtgeo.x_interp_map_nodes(xv, yv, zv, xval, yval, method)

    if isinstance(zexpected, list):
        assert zresult in zexpected
    else:
        assert zresult == zexpected
