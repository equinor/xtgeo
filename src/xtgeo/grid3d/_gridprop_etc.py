"""GridProperty (not GridProperies) some etc functions"""

# from ._gridprop_import_eclrun import import_eclbinary
#  --> INIT, UNRST
#
# from ._gridprop_import_grdecl import import_grdecl_prop, import_bgrdecl_prop
#  --> ASCII and BINARY GRDECL format
# from ._gridprop_import_roff import import_roff
#  --> BINARY ROFF format

from __future__ import print_function, absolute_import
import numpy as np

import xtgeo

xtg = xtgeo.common.XTGeoDialog()

logger = xtg.functionlogger(__name__)


def gridproperty_fromgrid(self, grid, linkgeometry=False):

    """Make an simple GridProperty instance directly based on an existing grid or
    gridproperty.

    Args:
        grid (Grid or GridProperty): The grid(property) geometry instance
        linkgeometry (bool): If True, connect the property.geometry to the input grid
            which is only applicable if Grid is input
    Example::

        import xtgeo
        grd = xtgeo.grid_from_file("my.roff")
        myporo = xtgeo.GridProperty(grd, name="PORO")
        myfacies = xtgeo.GridProperty(grd, name="Facies", linkgeometry=grd)
        myfacies.geomery = grd  # alternative way to link geometry

    .. versionadded:: 2.6.0

    """
    self._ncol = grid.ncol
    self._nrow = grid.nrow
    self._nlay = grid.nlay

    vals = self._values
    if vals is None:
        vals = 0

    if isinstance(vals, (int, float)):
        dtype = np.float64
        if self._isdiscrete:
            dtype = np.int32
            self._roxar_dtype = np.uint8

        if isinstance(vals, int):
            vals = int(vals)
        else:
            vals = float(vals)

        vals = np.zeros(grid.dimensions, dtype=dtype) + vals
    else:
        vals = vals.copy()  # do copy do avoid potensial reference issues

    if isinstance(grid, xtgeo.grid3d.Grid):

        act = grid.get_actnum(asmasked=True)

        self._values = np.ma.array(vals, mask=np.ma.getmaskarray(act.values))

        del act

        if linkgeometry:
            # assosiate this grid property with grid instance. This is not default
            # since sunch links may affect garbish collection
            self.geometry = grid

        grid.append_prop(self)

    else:
        self._values = np.ma.array(vals, mask=np.ma.getmaskarray(grid.values))


def gridproperty_fromfile(self, pfile, **kwargs):

    """Make an GridProperty from file.

    Args:
        pfile (str): Name of file
        **kwargs: Various settings

    """
    logger.debug("Import from file...")
    fformat = kwargs.get("fformat", "guess")

    self.from_file(
        pfile,
        fformat=fformat,
        name=self._name,
        grid=self._geometry,
        gridlink=kwargs.get("gridlink"),
        date=self._date,
        fracture=self._fracture,
    )


def gridproperty_fromspec(self, **kwargs):

    """Make an GridProperty from kwargs spec.

    Args:
        pfile (str): Name of file
        **kwargs: Various settings

    """

    # self._geometry: this is a link to the Grid instance, _only if needed_. It may
    # potentially make trouble for garbage collection

    values = kwargs.get("values", None)

    testmask = False
    if values is None:
        values = np.ma.zeros(self.dimensions)
        values += 99
        testmask = True

    if values.shape != self.dimensions:
        values = values.reshape(self.dimensions, order="C")

    if not isinstance(values, np.ma.MaskedArray):
        values = np.ma.array(values)

    self._values = values  # numpy version of properties (as 3D array)

    if self._isdiscrete:
        self._values = self._values.astype(np.int32)
        self._roxar_dtype = np.uint8

    if testmask:
        # make some undef cells (for test)
        self._values[0:4, 0, 0:2] = xtgeo.UNDEF
        # make it masked
        self._values = np.ma.masked_greater(self._values, xtgeo.UNDEF_LIMIT)
