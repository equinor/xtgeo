"""Various grid property operations"""

from __future__ import print_function, absolute_import

import numpy as np

import xtgeo
from xtgeo.common import XTGeoDialog
from xtgeo.grid3d import _gridprop_lowlevel as gl
import xtgeo.cxtgeo._cxtgeo as _cxtgeo

xtg = XTGeoDialog()

logger = xtg.functionlogger(__name__)


# pylint: disable=protected-access


def get_xy_value_lists(self, **kwargs):
    """Get values for webportal format

    Two cells:
    [[[(x1,y1), (x2,y2), (x3,y3), (x4,y4)],
    [(x5,y5), (x6,y6), (x7,y7), (x8,y8)]]]

    If mask is True then inactive cells are ommited from the lists,
    else the active cells corners will be present while the property
    will have a -999 value.

    """

    grid = kwargs.get("grid", None)

    mask = kwargs.get("mask", True)

    if grid is None:
        raise RuntimeError("Missing grid object")

    if not isinstance(grid, xtgeo.grid3d.Grid):
        raise RuntimeError("The input grid is not a XTGeo Grid instance")

    if not isinstance(self, xtgeo.grid3d.GridProperty):
        raise RuntimeError("The property is not a XTGeo GridProperty instance")

    clist = grid.get_xyz_corners()
    actnum = grid.get_actnum()

    # set value 0 if actnum is 0 to facilitate later operations
    if mask:
        for cli in clist:
            cli.values[actnum.values == 0] = 0

    # now some numpy operations (coffee?, any?)
    xy0 = np.column_stack((clist[0].values1d, clist[1].values1d))
    xy1 = np.column_stack((clist[3].values1d, clist[4].values1d))
    xy2 = np.column_stack((clist[6].values1d, clist[7].values1d))
    xy3 = np.column_stack((clist[9].values1d, clist[10].values1d))

    xyc = np.column_stack((xy0, xy1, xy2, xy3))
    xyc = xyc.reshape(grid.nlay, grid.ncol * grid.nrow, 4, 2)

    coordlist = xyc.tolist()

    # remove cells that are undefined ("marked" as coordinate [0, 0] if mask)
    coordlist = [
        [[tuple(xy) for xy in cell if xy[0] > 0] for cell in lay] for lay in coordlist
    ]

    coordlist = [[cell for cell in lay if len(cell) > 1] for lay in coordlist]

    pval = self.values1d.reshape((grid.nlay, grid.ncol * grid.nrow))
    valuelist = pval.tolist(fill_value=-999.0)
    if mask:
        valuelist = [[val for val in lay if val != -999.0] for lay in valuelist]

    return coordlist, valuelist


def operation_polygons(self, poly, value, opname="add", inside=True):
    """A generic function for doing operations restricted to inside
    or outside polygon(s).
    """

    grid = self.geometry

    if not isinstance(poly, xtgeo.xyz.Polygons):
        raise ValueError("The poly input is not a Polygons instance")

    # make a copy of the array which is used a "filter" or "proxy"
    # value will be 1 inside polygons, 0 outside. Undef cells are kept as is
    dtype = self.dtype

    proxy = self.copy()
    proxy.discrete_to_continuous()

    proxy.values *= 0.0
    cvals = gl.update_carray(proxy)

    idgroups = poly.dataframe.groupby(poly.pname)

    for id_, grp in idgroups:
        xcor = grp[poly.xname].values
        ycor = grp[poly.yname].values

        ier = _cxtgeo.grd3d_setval_poly(
            xcor,
            ycor,
            self.ncol,
            self.nrow,
            self.nlay,
            grid._coordsv,
            grid._zcornsv,
            grid._actnumsv,
            cvals,
            1,
        )
        if ier == -9:
            print("## Polygon no {} is not closed".format(id_ + 1))

    gl.update_values_from_carray(proxy, cvals, np.float64, delete=True)

    proxyv = proxy.values.astype(np.int8)

    proxytarget = 1
    if not inside:
        proxytarget = 0

    if opname == "add":
        tmp = self.values.copy() + value
    elif opname == "sub":
        tmp = self.values.copy() - value
    elif opname == "mul":
        tmp = self.values.copy() * value
    elif opname == "div":
        # Dividing a map of zero is always a hazzle; try to obtain 0.0
        # as result in these cases
        if 0.0 in value:
            xtg.warn(
                "Dividing a surface with value or surface with zero "
                "elements; may get unexpected results, try to "
                "achieve zero values as result!"
            )
        with np.errstate(divide="ignore", invalid="ignore"):
            this = np.ma.filled(self.values, fill_value=1.0)
            that = np.ma.filled(value, fill_value=1.0)
            mask = np.ma.getmaskarray(self.values)
            tmp = np.true_divide(this, that)
            tmp = np.where(np.isinf(tmp), 0, tmp)
            tmp = np.nan_to_num(tmp)
            tmp = np.ma.array(tmp, mask=mask)

    elif opname == "set":
        tmp = self.values.copy() * 0 + value

    # convert tmp back to correct dtype
    tmp = tmp.astype(dtype)

    self.values[proxyv == proxytarget] = tmp[proxyv == proxytarget]
    del tmp
