import pytest
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
        self.values1d = np.array(range(self.nrow * self.ncol))

    def get_values1d(self):
        return self.values1d


eps = 1e-6


@pytest.mark.parametrize(
    "coords, expected_val",
    [
        pytest.param((10.0, 10.0), 0.0, id="(xori, yori)"),
        pytest.param((10.0, 20.0), 1.0, id="(xori, ymax)"),
        pytest.param((20.0, 10.0), 2.0, id="(xmax, yori)"),
        pytest.param((20.0, 20.0), 3.0, id="(xmax, ymax)"),
        pytest.param((15.0, 10.0), 0.0, id="((xori + xmax) / 2, yori)"),
        pytest.param((10.0, 15.0), 0.0, id="(xori, (yori + ymax) / 2)"),
        pytest.param((15.0, 15.0), 0.0, id="((xori + xmax) / 2, (yori + ymax) / 2)"),
        pytest.param((15.0 - eps, 15.0 + eps), 1),
        pytest.param((15.0 + eps, 15.0 - eps), 2),
        pytest.param((15.0 + eps, 15.0 + eps), 3),
        pytest.param((10.0 - 1e-4, 10.0), 0.0, id="(xori - 1e-4, yori)"),
        pytest.param((10.0, 20.0 + 1e-4), 1.0, id="(xori, ymax + 1e-4)"),
        pytest.param((20.0 + 1e-4, 10.0), 2.0, id="(xmax - 1e-4, yori)"),
        pytest.param((20.0 + 1e-4, 20.0), 3.0, id="(xmax - 1e-4, ymax)"),
    ],
)
def test_ijk(coords, expected_val):
    surface = Surface()
    result = xtgeo.surface.regular_surface._regsurf_oper.get_value_from_xy(
        surface, coords, sampling="None"
    )
    assert result == expected_val


@pytest.mark.parametrize(
    "coords, expected_val",
    [
        pytest.param((10.0, 10.0), 0.0, id="(xori, yori)"),
        pytest.param((10.0, 20.0), 1.0, id="(xori, ymax)"),
        pytest.param((20.0, 10.0), 2.0, id="(xmax, yori)"),
        pytest.param((20.0, 20.0), 3.0, id="(xmax, ymax)"),
        pytest.param((15.0, 10.0), 1.0, id="((xori + xmax) / 2, yori)"),
        pytest.param((10.0, 15.0), 0.5, id="(xori, (yori + ymax) / 2)"),
        pytest.param((15.0, 15.0), 1.5, id="((xori + xmax) / 2, (yori + ymax) / 2)"),
    ],
)
def test_ijk_bilinear(coords, expected_val):
    surface = Surface()
    result = xtgeo.surface.regular_surface._regsurf_oper.get_value_from_xy(
        surface, coords, sampling="bilinear"
    )
    assert result == expected_val


@pytest.mark.parametrize("sampling", ["None", "bilinear"])
@pytest.mark.parametrize(
    "coords, expected_val",
    [
        pytest.param((10.0 - 9e-3, 10.0), None, id="(xori - 9e-3, yori)"),
        pytest.param((10.0, 20.0 + 9e-3), None, id="(xori, ymax - 9e-3)"),
        pytest.param((20.0 + 9e-3, 10.0), None, id="(xmax + 9e-3, yori)"),
        pytest.param((20.0 + 9e-3, 20.0), None, id="(xmax + 9e-3, ymax)"),
        pytest.param((9.0, 9.0), None, id="(xori - 1, yori - 1)"),
        pytest.param((21.0, 21.0 - 1e-10), None, id="(xmax + 1, ymax + 1)"),
    ],
)
def test_ijk_outside(coords, expected_val, sampling):
    surface = Surface()
    result = xtgeo.surface.regular_surface._regsurf_oper.get_value_from_xy(
        surface, coords, sampling=sampling
    )
    assert result == expected_val
