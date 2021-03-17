import pytest
import numpy as np
import hypothesis.strategies as st
from hypothesis import given, assume
from xtgeo import RegularSurface, Grid


def test_default_values():
    surf = RegularSurface(2, 2, 0.0, 0.0)
    assert surf.values.data.tolist() == [[0.0, 0.0], [0.0, 0.0]]


@pytest.mark.parametrize(
    "input_val, expected_result",
    [
        (1, [[1.0, 1.0], [1.0, 1.0]]),
        ([1, 2, 3, 4], [[1.0, 2.0], [3.0, 4.0]]),
        (np.zeros((2, 2)), [[0.0, 0.0], [0.0, 0.0]]),
        (None, [[0.0, 0.0], [0.0, 0.0]]),
        (
            np.ma.MaskedArray([[2, 2], [2, 2]], mask=[[True, False], [False, True]]),
            [[2.0, 2.0], [2.0, 2.0]],
        ),
    ],
)
def test_values_type(input_val, expected_result):
    surf = RegularSurface(2, 2, 0.0, 0.0, values=input_val)
    assert isinstance(surf.values, np.ma.MaskedArray)
    assert surf.values.data.tolist() == expected_result


@pytest.mark.parametrize("input_values", [None, 2])
@pytest.mark.parametrize(
    "input_val, expected_result",
    [
        (1, [[1.0, 1.0], [1.0, 1.0]]),
        ([1, 2, 3, 4], [[1.0, 2.0], [3.0, 4.0]]),
        (np.zeros((2, 2)), [[0.0, 0.0], [0.0, 0.0]]),
        (np.ma.zeros((2, 2)), [[0.0, 0.0], [0.0, 0.0]]),
        (np.ma.ones((2, 2)), [[1.0, 1.0], [1.0, 1.0]]),
    ],
)
def test_values_setter(input_values, input_val, expected_result):
    surf = RegularSurface(2, 2, 0.0, 0.0, values=input_values)
    surf.values = input_val
    assert surf.values.data.tolist() == expected_result


@pytest.mark.parametrize(
    "input_val, expected_data, expected_mask",
    [
        (1, [2.0, 1.0, 1.0, 2.0], [[True, False], [False, True]]),
        ([1, 2, 3, 4], [1.0, 2.0, 3.0, 4.0], [[False, False], [False, False]]),
        ([1, 1e33, 3, 4], [1.0, 1e33, 3.0, 4.0], [[False, True], [False, False]]),
        (np.zeros((2, 2)), [0.0, 0.0, 0.0, 0.0], [[False, False], [False, False]]),
        (np.ma.zeros((2, 2)), [0.0, 0.0, 0.0, 0.0], [[False, False], [False, False]]),
        (np.ma.ones((2, 2)), [1.0, 1.0, 1.0, 1.0], [[False, False], [False, False]]),
    ],
)
def test_values_mask_setter(input_val, expected_data, expected_mask):
    surf = RegularSurface(
        2,
        2,
        0.0,
        0.0,
        values=np.ma.MaskedArray([[2, 2], [2, 2]], mask=[[True, False], [False, True]]),
    )
    surf.values = input_val
    assert surf.values.mask.tolist() == expected_mask
    assert list(surf.values.data.flatten()) == expected_data


@given(data=st.lists(st.floats(allow_nan=False), min_size=100, max_size=100))
def test_input_lists(data):
    surf = RegularSurface(10, 10, 0.0, 0.0, values=data)
    assert surf.values.data.flatten().tolist() == pytest.approx(data)


@given(data=st.lists(st.floats(allow_nan=False)))
def test_wrong_size_input_lists(data):
    assume(len(data) != 100)
    with pytest.raises(ValueError, match=r"Cannot reshape array:"):
        RegularSurface(10, 10, 0.0, 0.0, values=data)


@given(data=st.one_of(st.floats(allow_nan=False), st.integers()))
def test_input_numbers(data):
    surf = RegularSurface(10, 10, 0.0, 0.0, values=data)
    assert set(surf.values.data.flatten().tolist()) == pytest.approx({data})


def test_read_grid3d():
    grid = Grid()  # this creates an example Grid (no other way to do it currently)
    surf = RegularSurface._read_grid3d(grid=grid)
    # There is some resolution changes between the grid and the surface, so we cant
    # expect identical sizes, even for regular grids.
    assert (surf.ncol, surf.nrow) == (3, 3)
