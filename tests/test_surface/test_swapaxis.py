import pytest
from xtgeo import RegularSurface


@pytest.fixture()
def default_surface():
    yield {
        "xori": 0.0,
        "yori": 0.0,
        "ncol": 2,
        "nrow": 2,
        "xinc": 1.0,
        "yinc": 1.0,
        "yflip": 1,
        "values": [1, 2, 3, 4],
    }


def test_swapaxis(default_surface):
    surface = RegularSurface(**default_surface)

    assert surface.values.flatten().tolist() == [1, 2, 3, 4]

    surface.swapaxes()

    assert surface.values.flatten().tolist() == [1.0, 3.0, 2.0, 4.0]


@pytest.mark.parametrize(
    "rotation, expected_rotation",
    [
        (-1, 89),
        (0, 90),
        (90, 180),
        (180, 270),
        (270, 0),
        (360, 90),
        (361, 91),
    ],
)
def test_swapaxis_rotation(rotation, expected_rotation, default_surface):
    default_surface["rotation"] = rotation
    surface = RegularSurface(**default_surface)

    surface.swapaxes()

    assert surface.rotation == expected_rotation


def test_swapaxis_ilines(default_surface):
    surface = RegularSurface(**default_surface)

    assert surface.ilines.tolist() == [1, 2]

    surface.swapaxes()

    assert surface.ilines.tolist() == [1, 2]


def test_swapaxis_ncol_nrow(default_surface):
    default_surface["nrow"] = 3
    default_surface["values"] = [1] * 6
    surface = RegularSurface(**default_surface)

    surface.swapaxes()

    assert (surface.nrow, surface.ncol) == (2, 3)


def test_swapaxis_xinc_yinc(default_surface):
    default_surface["yinc"] = 2.0
    surface = RegularSurface(**default_surface)

    surface.swapaxes()

    assert (surface.xinc, surface.yinc) == (2, 1)


@pytest.mark.parametrize("yflip, expected_result", [(1, -1), (-1, 1)])
def test_swapaxis_yflip(default_surface, yflip, expected_result):
    default_surface["yflip"] = yflip
    surface = RegularSurface(**default_surface)

    surface.swapaxes()

    assert surface.yflip == expected_result
