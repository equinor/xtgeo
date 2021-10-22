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


def gridproperty_fromgrid(self, gridlike, linkgeometry=False, values=None):
    """Make GridProperty instance directly based on an existing grid or gridproperty.

    Args:
        gridlike (Grid or GridProperty): The grid(property) geometry instance.
        linkgeometry (bool): If True, connect the property.geometry to the input grid,
            this is only applicable if Grid is input.
        values: Input values (various data types)
    Example::

        import xtgeo
        grd = xtgeo.grid_from_file("my.roff")
        myporo = xtgeo.GridProperty(grd, name="PORO")
        myfacies = xtgeo.GridProperty(grd, name="Facies", linkgeometry=True)
        myfacies.geometry = grd  # alternative way to link geometry

    .. versionadded:: 2.6

    """
    self._ncol = gridlike.ncol
    self._nrow = gridlike.nrow
    self._nlay = gridlike.nlay

    gridvalues_fromspec(self, values)

    if isinstance(gridlike, xtgeo.grid3d.Grid):

        act = gridlike.get_actnum(asmasked=True)

        self._values = np.ma.array(self._values, mask=np.ma.getmaskarray(act.values))

        del act

        if linkgeometry:
            # assosiate this grid property with grid instance. This is not default
            # since sunch links may affect garbish collection
            self.geometry = gridlike

        gridlike.append_prop(self)

    else:
        self._values = np.ma.array(
            self._values, mask=np.ma.getmaskarray(gridlike.values)
        )


def default_gridprop(self):
    self._ncol = 4
    self._nrow = 3
    self._nlay = 5
    if self._isdiscrete:
        self._values = np.ma.MaskedArray(np.full((4, 3, 5), 99), dtype=np.int32)
    else:
        self._values = np.ma.MaskedArray(np.full((4, 3, 5), 99.0))
    self._values[0:4, 0, 0:2] = np.ma.masked


def gridvalues_fromspec(self, values):
    """Update or set values.

    Args:
        values: Values will be None, and array or a scalar
    """
    if self._ncol is None:
        self._ncol = 4
    if self._nrow is None:
        self._nrow = 3
    if self._nlay is None:
        self._nlay = 5
    if values is None:
        if self._isdiscrete:
            values = np.ma.zeros(self.dimensions, dtype=np.int32)
        else:
            values = np.ma.zeros(self.dimensions)

    elif np.isscalar(values):
        if isinstance(values, (float, int)):
            dtype = np.float64
            if self._isdiscrete:
                dtype = np.int32
            values = np.ma.zeros(self.dimensions, dtype=dtype) + values
        else:
            raise ValueError("Scalar input values of invalid type")

    elif isinstance(values, np.ndarray):
        values = np.ma.MaskedArray(values.reshape(self.dimensions))
        if self._isdiscrete:
            values = values.astype(np.int32)
        else:
            values = values.astype(np.float64)

    else:
        raise ValueError("Input values of invalid type")

    self._values = values  # numpy version of properties (as 3D array)
