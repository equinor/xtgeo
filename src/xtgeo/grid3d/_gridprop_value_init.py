"""GridProperty (not GridProperies) some etc functions"""

# from ._gridprop_import_eclrun import import_eclbinary
#  --> INIT, UNRST
#
# from ._gridprop_import_grdecl import import_grdecl_prop, import_bgrdecl_prop
#  --> ASCII and BINARY GRDECL format
# from ._gridprop_import_roff import import_roff
#  --> BINARY ROFF format


import numpy as np

import xtgeo

xtg = xtgeo.common.XTGeoDialog()

logger = xtg.functionlogger(__name__)


def gridproperty_non_dummy_values(gridlike, dimensions, values, isdiscrete):
    """Gives the initial values array of an gridprop.

    Param:
      gridlike: Either Grid or GridProperty, giving the mask to replicate.
      dimensions: The dimensions of the gridprop
      values: The values parameter given to init.
      isdiscrete: The discrete parameter given to init.

    Returns:
      The array to be set to GridProp._values.
    """
    if values is None:
        values = initial_gridprop_values_zero(dimensions, isdiscrete)
    elif np.isscalar(values):
        values = initial_gridprop_values_from_scalar(dimensions, values, isdiscrete)
    elif isinstance(values, np.ndarray):
        values = initial_gridprop_values_from_array(dimensions, values, isdiscrete)

    if gridlike is not None:
        if isinstance(gridlike, xtgeo.grid3d.Grid):
            act = gridlike.get_actnum(asmasked=True)
            values = np.ma.array(values, mask=np.ma.getmaskarray(act.values))
        else:
            values = np.ma.array(values, mask=np.ma.getmaskarray(gridlike.values))

    return values


def gridproperty_dummy_values(isdiscrete):
    """Given no parameters to init, these dummy values should be set for backwards
    compatability."""
    if isdiscrete:
        values = np.ma.MaskedArray(np.full((4, 3, 5), 99), dtype=np.int32)
    else:
        values = np.ma.MaskedArray(np.full((4, 3, 5), 99.0))
    values[0:4, 0, 0:2] = np.ma.masked
    return values


def initial_gridprop_values_zero(dimensions, isdiscrete):
    """Initial values for an GridProperty with zeros.

    Given that the user supplies at least some parameters, but not a values array,
    values should be initialized to zero.
    Param:
      dimensions: The dimensions of the gridproperty.


    Returns:
        zero initialized values array
    """
    if isdiscrete:
        return np.ma.zeros(dimensions, dtype=np.int32)
    return np.ma.zeros(dimensions)


def initial_gridprop_values_from_scalar(dimensions, value, isdiscrete):
    """Initial gridproperties values from scalar.

    Given scalar values, the gridproperties value array should be
    filled with that value, with possible conversion depending
    on the isdiscrete parameter.

    Returns:
        filled array with given scalar value.
    """
    if isinstance(value, (float, int)):
        dtype = np.float64
        if isdiscrete:
            dtype = np.int32
        return np.ma.zeros(dimensions, dtype=dtype) + value
    raise ValueError("Scalar input values of invalid type")


def initial_gridprop_values_from_array(dimensions, values, isdiscrete):
    """Initial gridproperties values from numpy array"""
    values = np.ma.MaskedArray(values.reshape(dimensions))
    if isdiscrete:
        return values.astype(np.int32)
    return values.astype(np.float64)
