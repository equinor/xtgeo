"""Test some basic cxtgeo functions."""
import numpy as np
import pytest

import xtgeo
import xtgeo.cxtgeo._cxtgeo as _cxtgeo

xtg = xtgeo.XTGeoDialog()
logger = xtg.basiclogger(__name__)


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
    logseries = np.nan_to_num(logseries, nan=xtgeo.UNDEF_INT).astype("int32")
    print(logseries)

    expected = np.array(expected)

    mask = np.zeros(depth.size, dtype="int32")
    res = _cxtgeo.well_mask_shoulder(depth, logseries, mask, distance)

    mask = mask.astype("bool")
    print(mask)
    assert res == 0
    assert (mask == expected).all()
