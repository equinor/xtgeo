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


def gridvalues_fromspec(self, values):
    """Update or set values.

    Args:
        values: Values will be None, and array or a scalar
    """
    defaultvalues = False
    if values is None:
        values = np.ma.zeros(self.dimensions)

        if self.dimensions == (4, 3, 5):
            # looks like default input values
            values += 99
            defaultvalues = True
            self._isdiscrete = False

    elif np.isscalar(values):
        if isinstance(values, (float, int)):
            dtype = np.float64
            if self._isdiscrete:
                dtype = np.int32
                self._roxar_dtype = np.uint8
            values = np.ma.zeros(self.dimensions, dtype=dtype) + values
        else:
            raise ValueError("Scalar input values of invalid type")

    elif isinstance(values, np.ndarray):
        values = np.ma.zeros(self.dimensions) + values.reshape(self.dimensions)

    else:
        raise ValueError("Input values of invalid type")

    self._values = values  # numpy version of properties (as 3D array)

    if defaultvalues:
        # make some undef cells (when in default values mode)
        self._values[0:4, 0, 0:2] = xtgeo.UNDEF
        # make it masked
        self._values = np.ma.masked_greater(self._values, xtgeo.UNDEF_LIMIT)
