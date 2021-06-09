import pytest
import xtgeo.cxtgeo._cxtgeo as _cxtgeo  # type: ignore
import xtgeo
import numpy as np


class Surface:
    def __init__(self):
        self.ncol = 2
        self.nrow = 2
        self.xori = 10.0
        self.yori = 10.0
        self.xinc = 10.0
        self.yinc = 10.0
        self.yflip = 1
        self.rotation = 0
        self.values = np.array(range(self.nrow * self.ncol))


@pytest.mark.parametrize(
    "i, j, flag, expected_result",
    [
        (1, 1, 0, (10.0, 10.0, 0)),
        (1, 2, 0, (10.0, 20.0, 1)),
        (2, 1, 0, (20.0, 10.0, 2)),
        (2, 2, 0, (20.0, 20.0, 3)),
        (1, 1, 1, (10.0, 10.0, 999)),
        (1, 2, 1, (10.0, 20.0, 999)),
        (2, 1, 1, (20.0, 10.0, 999)),
        (2, 2, 1, (20.0, 20.0, 999)),
    ],
)
def test_xyz_from_ij_inside(i, j, expected_result, flag):
    surface = Surface()
    ier, xval, yval, value = _cxtgeo.surf_xyz_from_ij(
        i,
        j,
        surface.xori,
        surface.xinc,
        surface.yori,
        surface.yinc,
        surface.ncol,
        surface.nrow,
        surface.yflip,
        surface.rotation,
        surface.values,
        flag,
    )
    assert (xval, yval, value) == expected_result


@pytest.mark.parametrize(
    "i, j, expected_position",
    [
        (3, 3, (20.0, 20.0)),
        (0, 0, (10.0, 10.0)),
    ],
)
def test_xyz_from_ij_one_outside(i, j, expected_position):
    surface = Surface()
    ier, xval, yval, value = _cxtgeo.surf_xyz_from_ij(
        i,
        j,
        surface.xori,
        surface.xinc,
        surface.yori,
        surface.yinc,
        surface.ncol,
        surface.nrow,
        surface.yflip,
        surface.rotation,
        surface.values,
        0,
    )
    assert (xval, yval) == expected_position


@pytest.mark.parametrize(
    "i, j",
    [
        (-1, -1),
        (4, 4),
        (100, 100),
        (-1, 1),
        (4, 1),
        (100, 1),
    ],
)
def test_xyz_from_ij_far_outside(i, j):
    surface = Surface()
    with pytest.raises(xtgeo.XTGeoCLibError, match="Accessing value outside surface"):
        _cxtgeo.surf_xyz_from_ij(
            i,
            j,
            surface.xori,
            surface.xinc,
            surface.yori,
            surface.yinc,
            surface.ncol,
            surface.nrow,
            surface.yflip,
            surface.rotation,
            surface.values,
            0,
        )
