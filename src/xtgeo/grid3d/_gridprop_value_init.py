"""GridProperty (not GridProperies) some etc functions"""
from __future__ import annotations

import numbers
from typing import TYPE_CHECKING

import numpy as np

import xtgeo
from xtgeo.common import null_logger

logger = null_logger(__name__)

if TYPE_CHECKING:
    from xtgeo.grid3d import Grid, GridProperty


def gridproperty_non_dummy_values(
    gridlike: Grid | GridProperty | None,
    dimensions: tuple[int, int, int],
    values: np.ndarray | float | int | None,
    isdiscrete: bool,
) -> np.ma.MaskedArray:
    """
    Gives the initial values array of an gridprop.

    Args:
        gridlike: Either Grid or GridProperty, giving the mask to replicate.
        dimensions: The (ncol, nrow, nlay) dimensions of the grid property.
        values: The values parameter given to init.
        isdiscrete: The discrete parameter given to init.

    Returns:
        The array to be set to GridProperty._values.

    """
    if values is None:
        _values = initial_gridprop_values_zero(dimensions, isdiscrete)
    elif isinstance(values, numbers.Number):
        _values = initial_gridprop_values_from_scalar(dimensions, values, isdiscrete)
    elif isinstance(values, np.ndarray):
        _values = initial_gridprop_values_from_array(dimensions, values, isdiscrete)

    if gridlike:
        if isinstance(gridlike, xtgeo.grid3d.Grid):
            act = gridlike.get_actnum(asmasked=True)
            _values = np.ma.array(_values, mask=np.ma.getmaskarray(act.values))
        else:
            assert isinstance(gridlike, xtgeo.grid3d.GridProperty)
            _values = np.ma.array(_values, mask=np.ma.getmaskarray(gridlike.values))

    return _values


def gridproperty_dummy_values(isdiscrete: bool) -> np.ma.MaskedArray:
    """
    Given no parameters to init, these dummy values should be set for backwards
    compatability.

    Args:
        isdiscrete: If the grid property values are discrete.

    Returns:
        The array to be set to GridProperty._values.

    """
    values: np.ma.MaskedArray = np.ma.MaskedArray(
        np.full((4, 3, 5), 99.0), dtype=np.int32 if isdiscrete else np.float64
    )
    values[0:4, 0, 0:2] = np.ma.masked
    return values


def initial_gridprop_values_zero(
    dimensions: tuple[int, int, int], isdiscrete: bool
) -> np.ma.MaskedArray:
    """
    Initial values for an GridProperty with zeros.

    Given that the user supplies at least some parameters, but not a values array,
    values should be initialized to zero.

    Args:
        dimensions: The (ncol, nrow, nlay) dimensions of the grid property.

    Returns:
        Zero initialized values array.

    """
    return np.ma.zeros(dimensions, dtype=np.int32 if isdiscrete else np.float64)


def initial_gridprop_values_from_scalar(
    dimensions: tuple[int, int, int], value: float | int, isdiscrete: bool
) -> np.ma.MaskedArray:
    """
    Initial grid property values from scalar.

    Given scalar values, the gridproperties value array should be
    filled with that value, with possible conversion depending
    on the isdiscrete parameter.

    Args:
        dimensions: The (ncol, nrow, nlay) dimensions of the grid property.
        value: The scalar value to initialize with.
        isdiscrete: If the values are discrete.

    Returns:
        Filled array with given scalar value.

    """
    if not isinstance(value, numbers.Number):
        raise ValueError("Scalar input values of invalid type")
    return np.ma.zeros(dimensions, dtype=np.int32 if isdiscrete else np.float64) + value


def initial_gridprop_values_from_array(
    dimensions: tuple[int, int, int], values: np.ndarray, isdiscrete: bool
) -> np.ma.MaskedArray:
    """
    Initial GridProperty values from numpy array.

    Args:
        dimensions: The (ncol, nrow, nlay) dimensions of the grid property.
        value: The numpy array to initialize with.
        isdiscrete: If the values are discrete.

    Returns:
        GridProperty with values initialized from a numpy array.

    """
    return np.ma.MaskedArray(
        values.reshape(dimensions), dtype=np.int32 if isdiscrete else np.float64
    )
