"""Private module, Grid ETC 1 methods, info/modify/report."""

from collections import OrderedDict
from copy import deepcopy
from math import atan2, degrees
from typing import Tuple

import numpy as np
import numpy.ma as ma
import pandas as pd
import xtgeo
import xtgeo.cxtgeo._cxtgeo as _cxtgeo
from xtgeo.common import XTGeoDialog
from xtgeo.common.calc import find_flip
from xtgeo.xyz.polygons import Polygons

from . import _gridprop_lowlevel
from ._grid3d_fence import _update_tmpvars
from .grid_property import GridProperty

xtg = XTGeoDialog()


logger = xtg.functionlogger(__name__)


# Note that "self" is the grid instance


def create_box(dimension, origin, oricenter, increment, rotation, flip):
    """Create a shoebox grid from cubi'sh spec, xtgformat=2."""
    ncol, nrow, nlay = dimension
    nncol = ncol + 1
    nnrow = nrow + 1
    nnlay = nlay + 1

    coordsv = np.zeros((nncol, nnrow, 6), dtype=np.float64)
    zcornsv = np.zeros((nncol, nnrow, nnlay, 4), dtype=np.float32)
    actnumsv = np.zeros((ncol, nrow, nlay), dtype=np.int32)

    option = 0
    if oricenter:
        option = 1

    _cxtgeo.grdcp3d_from_cube(
        ncol,
        nrow,
        nlay,
        coordsv,
        zcornsv,
        actnumsv,
        origin[0],
        origin[1],
        origin[2],
        increment[0],
        increment[1],
        increment[2],
        rotation,
        flip,
        option,
    )

    return {
        "coordsv": coordsv,
        "zcornsv": zcornsv,
        "actnumsv": actnumsv,
    }


method_factory = {
    "euclid": _cxtgeo.euclid_length,
    "horizontal": _cxtgeo.horizontal_length,
    "east west vertical": _cxtgeo.east_west_vertical_length,
    "north south vertical": _cxtgeo.north_south_vertical_length,
    "x projection": _cxtgeo.x_projection,
    "y projection": _cxtgeo.y_projection,
    "z projection": _cxtgeo.z_projection,
}


def get_dz(
    self,
    name: str = "dZ",
    flip: bool = True,
    asmasked: bool = True,
    metric="z projection",
) -> GridProperty:
    """Get average cell height (dz) as property.

    Args:
        flip (bool): whether to flip the z direction, ie. increasing z is
            increasing depth (defaults to True)
        asmasked (bool): Whether to mask property by whether
        name (str): Name of resulting grid property, defaults to "dZ".
    """
    self._xtgformat2()
    nx, ny, nz = self.dimensions
    result = np.zeros((nx * ny * nz))
    try:
        metric_fun = method_factory[metric]
    except KeyError as err:
        raise ValueError(f"Unknown metric {metric}") from err
    _cxtgeo.grdcp3d_calc_dz(
        self._ncol,
        self._nrow,
        self._nlay,
        self._coordsv.ravel(),
        self._zcornsv.ravel(),
        result,
        metric_fun,
    )

    if not flip:
        result *= -1

    if asmasked:
        result = np.ma.masked_array(result, self._actnumsv == 0)
    else:
        result = np.ma.masked_array(result, False)

    return GridProperty(
        ncol=self._ncol,
        nrow=self._nrow,
        nlay=self._nlay,
        values=result.ravel(),
        name=name,
        discrete=False,
    )


def get_dx(self, name="dX", asmasked=False, metric="horizontal"):
    try:
        metric_fun = method_factory[metric]
    except KeyError as err:
        raise ValueError(f"Unknown metric {metric}") from err
    self._xtgformat2()
    nx, ny, nz = self.dimensions
    result = np.zeros((nx * ny * nz))
    _cxtgeo.grdcp3d_calc_dx(
        self._ncol,
        self._nrow,
        self._nlay,
        self._coordsv.ravel(),
        self._zcornsv.ravel(),
        result,
        metric_fun,
    )

    if asmasked:
        result = np.ma.masked_array(result, self._actnumsv == 0)
    else:
        result = np.ma.masked_array(result, False)
    return GridProperty(
        ncol=self._ncol,
        nrow=self._nrow,
        nlay=self._nlay,
        values=result.reshape((nx, ny, nz)),
        name=name,
        discrete=False,
    )


def get_dy(self, name="dX", asmasked=False, metric="horizontal"):
    try:
        metric_fun = method_factory[metric]
    except KeyError as err:
        raise ValueError(f"Unknown metric {metric}") from err
    self._xtgformat2()
    nx, ny, nz = self.dimensions
    result = np.zeros((nx * ny * nz))
    _cxtgeo.grdcp3d_calc_dy(
        self._ncol,
        self._nrow,
        self._nlay,
        self._coordsv.ravel(),
        self._zcornsv.ravel(),
        result,
        metric_fun,
    )

    if asmasked:
        result = np.ma.masked_array(result, self._actnumsv == 0)
    else:
        result = np.ma.masked_array(result, False)
    return GridProperty(
        ncol=self._ncol,
        nrow=self._nrow,
        nlay=self._nlay,
        values=result.reshape((nx, ny, nz)),
        name=name,
        discrete=False,
    )


def get_bulk_volume(self, name="bulkvol", asmasked=True, precision=2):
    """Get cell bulk volume as a GridProperty() instance."""
    self._xtgformat2()

    bulk = GridProperty(
        ncol=self._ncol,
        nrow=self._nrow,
        nlay=self._nlay,
        name=name,
        discrete=False,
    )

    bval = np.zeros((bulk.dimensions))

    if precision not in (1, 2, 4):
        raise ValueError("The precision key has an invalid entry, use 1, 2, or 4")

    _cxtgeo.grdcp3d_cellvol(
        self._ncol,
        self._nrow,
        self._nlay,
        self._coordsv,
        self._zcornsv,
        self._actnumsv,
        bval,
        precision,
        0 if asmasked else 1,
    )

    if asmasked:
        bval = np.ma.masked_greater(bval, xtgeo.UNDEF_LIMIT)

    bulk.values = bval

    return bulk


def get_ijk(self, names=("IX", "JY", "KZ"), asmasked=True, zerobased=False):
    """Get I J K as properties."""
    ashape = (self._ncol, self._nrow, self._nlay)

    ix, jy, kz = np.indices(ashape)

    ix = ix.ravel()
    jy = jy.ravel()
    kz = kz.ravel()

    if asmasked:
        actnum = self.get_actnum()

        ix = ma.masked_where(actnum.values1d == 0, ix)
        jy = ma.masked_where(actnum.values1d == 0, jy)
        kz = ma.masked_where(actnum.values1d == 0, kz)

    if not zerobased:
        ix += 1
        jy += 1
        kz += 1

    ix = GridProperty(
        ncol=self._ncol,
        nrow=self._nrow,
        nlay=self._nlay,
        values=ix.reshape(ashape),
        name=names[0],
        discrete=True,
    )
    jy = GridProperty(
        ncol=self._ncol,
        nrow=self._nrow,
        nlay=self._nlay,
        values=jy.reshape(ashape),
        name=names[1],
        discrete=True,
    )
    kz = GridProperty(
        ncol=self._ncol,
        nrow=self._nrow,
        nlay=self._nlay,
        values=kz.reshape(ashape),
        name=names[2],
        discrete=True,
    )

    # return the objects
    return ix, jy, kz


def get_ijk_from_points(
    self,
    points,
    activeonly=True,
    zerobased=False,
    dataframe=True,
    includepoints=True,
    columnnames=("IX", "JY", "KZ"),
    fmt="int",
    undef=-1,
):
    """Get I J K indices as a list of tuples or a dataframe.

    It is here tried to get fast execution. This requires a preprosessing
    of the grid to store a onlayer version, and maps with IJ positions
    """
    self._xtgformat1()
    logger.info("Getting IJK indices from Points...")

    actnumoption = 1
    if not activeonly:
        actnumoption = 0

    _update_tmpvars(self, force=True)

    arrsize = points.dataframe[points.xname].values.size

    useflip = 1
    if self.ijk_handedness == "left":
        useflip = -1

    logger.info("Grid FLIP for C code is %s", useflip)

    logger.info("Running C routine...")
    _, iarr, jarr, karr = _cxtgeo.grd3d_points_ijk_cells(
        points.dataframe[points.xname].values,
        points.dataframe[points.yname].values,
        points.dataframe[points.zname].values,
        self._tmp["topd"].ncol,
        self._tmp["topd"].nrow,
        self._tmp["topd"].xori,
        self._tmp["topd"].yori,
        self._tmp["topd"].xinc,
        self._tmp["topd"].yinc,
        self._tmp["topd"].rotation,
        self._tmp["topd"].yflip,
        self._tmp["topi_carr"],
        self._tmp["topj_carr"],
        self._tmp["basi_carr"],
        self._tmp["basj_carr"],
        self.ncol,
        self.nrow,
        self.nlay,
        self._coordsv,
        self._zcornsv,
        self._actnumsv,
        self._tmp["onegrid"]._zcornsv,
        actnumoption,
        arrsize,
        arrsize,
        arrsize,
    )
    logger.info("Running C routine... DONE")

    if zerobased:
        # zero based cell indexing
        iarr -= 1
        jarr -= 1
        karr -= 1

    proplist = OrderedDict()
    if includepoints:
        proplist["X_UTME"] = points.dataframe[points.xname].values
        proplist["Y_UTME"] = points.dataframe[points.yname].values
        proplist["Z_TVDSS"] = points.dataframe[points.zname].values

    proplist[columnnames[0]] = iarr
    proplist[columnnames[1]] = jarr
    proplist[columnnames[2]] = karr

    mydataframe = pd.DataFrame.from_dict(proplist)
    mydataframe.replace(xtgeo.UNDEF_INT, -1, inplace=True)

    if fmt == "float":
        mydataframe[columnnames[0]] = mydataframe[columnnames[0]].astype("float")
        mydataframe[columnnames[1]] = mydataframe[columnnames[1]].astype("float")
        mydataframe[columnnames[2]] = mydataframe[columnnames[2]].astype("float")

    if undef != -1:
        mydataframe[columnnames[0]].replace(-1, undef, inplace=True)
        mydataframe[columnnames[1]].replace(-1, undef, inplace=True)
        mydataframe[columnnames[2]].replace(-1, undef, inplace=True)

    result = mydataframe
    if not dataframe:
        result = list(mydataframe.itertuples(index=False, name=None))

    return result


def get_xyz(self, names=("X_UTME", "Y_UTMN", "Z_TVDSS"), asmasked=True):
    """Get X Y Z as properties."""
    # TODO: May be issues with asmasked vs activeonly here?

    self._xtgformat1()

    xv = np.zeros(self.ntotal, dtype=np.float64)
    yv = np.zeros(self.ntotal, dtype=np.float64)
    zv = np.zeros(self.ntotal, dtype=np.float64)

    option = 0
    if asmasked:
        option = 1

    _cxtgeo.grd3d_calc_xyz(
        self._ncol,
        self._nrow,
        self._nlay,
        self._coordsv,
        self._zcornsv,
        self._actnumsv,
        xv,
        yv,
        zv,
        option,
    )

    xv = np.ma.masked_greater(xv, xtgeo.UNDEF_LIMIT)
    yv = np.ma.masked_greater(yv, xtgeo.UNDEF_LIMIT)
    zv = np.ma.masked_greater(zv, xtgeo.UNDEF_LIMIT)

    xo = GridProperty(
        ncol=self._ncol,
        nrow=self._nrow,
        nlay=self._nlay,
        values=xv,
        name=names[0],
        discrete=False,
    )

    yo = GridProperty(
        ncol=self._ncol,
        nrow=self._nrow,
        nlay=self._nlay,
        values=yv,
        name=names[1],
        discrete=False,
    )

    zo = GridProperty(
        ncol=self._ncol,
        nrow=self._nrow,
        nlay=self._nlay,
        values=zv,
        name=names[2],
        discrete=False,
    )

    return xo, yo, zo


def get_xyz_cell_corners(self, ijk=(1, 1, 1), activeonly=True, zerobased=False):
    """Get X Y Z cell corners for one cell."""
    self._xtgformat1()

    i, j, k = ijk

    shift = 0
    if zerobased:
        shift = 1

    if activeonly:
        actnum = self.get_actnum()
        iact = actnum.values3d[i - 1 + shift, j - 1 + shift, k - 1 + shift]
        if iact == 0:
            return None

    pcorners = _cxtgeo.new_doublearray(24)

    if self._xtgformat == 1:
        logger.info("Use xtgformat 1...")
        _cxtgeo.grd3d_corners(
            i + shift,
            j + shift,
            k + shift,
            self.ncol,
            self.nrow,
            self.nlay,
            self._coordsv,
            self._zcornsv,
            pcorners,
        )
    else:
        logger.info("Use xtgformat 2...")
        _cxtgeo.grdcp3d_corners(
            i + shift - 1,
            j + shift - 1,
            k + shift - 1,
            self.ncol,
            self.nrow,
            self.nlay,
            self._coordsv,
            self._zcornsv,
            pcorners,
        )

    cornerlist = []
    for i in range(24):
        cornerlist.append(_cxtgeo.doublearray_getitem(pcorners, i))

    clist = tuple(cornerlist)
    return clist


def get_xyz_corners(self, names=("X_UTME", "Y_UTMN", "Z_TVDSS")):
    """Get X Y Z cell corners for all cells (as 24 GridProperty objects)."""
    self._xtgformat1()

    ntot = (self._ncol, self._nrow, self._nlay)

    grid_props = []

    for i in range(0, 8):
        xname = names[0] + str(i)
        yname = names[1] + str(i)
        zname = names[2] + str(i)
        x = GridProperty(
            ncol=self._ncol,
            nrow=self._nrow,
            nlay=self._nlay,
            values=np.zeros(ntot, dtype=np.float64),
            name=xname,
            discrete=False,
        )

        y = GridProperty(
            ncol=self._ncol,
            nrow=self._nrow,
            nlay=self._nlay,
            values=np.zeros(ntot, dtype=np.float64),
            name=yname,
            discrete=False,
        )

        z = GridProperty(
            ncol=self._ncol,
            nrow=self._nrow,
            nlay=self._nlay,
            values=np.zeros(ntot, dtype=np.float64),
            name=zname,
            discrete=False,
        )

        grid_props.append(x)
        grid_props.append(y)
        grid_props.append(z)

    ptr_coord = []
    for i in range(24):
        some = _cxtgeo.new_doublearray(self.ntotal)
        ptr_coord.append(some)

    for i, va in enumerate(ptr_coord):
        logger.debug("SWIG object %s   %s", i, va)

    option = 0

    # note, fool the argument list to unpack ptr_coord with * ...
    _cxtgeo.grd3d_get_all_corners(
        self._ncol,
        self._nrow,
        self._nlay,
        self._coordsv,
        self._zcornsv,
        self._actnumsv,
        *(ptr_coord + [option]),
    )

    for i in range(0, 24, 3):

        _gridprop_lowlevel.update_values_from_carray(
            grid_props[i], ptr_coord[i], np.float64, delete=True
        )

        _gridprop_lowlevel.update_values_from_carray(
            grid_props[i + 1], ptr_coord[i + 1], np.float64, delete=True
        )

        _gridprop_lowlevel.update_values_from_carray(
            grid_props[i + 2], ptr_coord[i + 2], np.float64, delete=True
        )

    # return the 24 objects (x1, y1, z1, ... x8, y8, z8)
    return tuple(grid_props)


def get_vtk_geometries(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return actnum, corners and dims arrays for VTK ExplicitStructuredGrid usage."""
    self._xtgformat2()

    narr = 8 * self.ncol * self.nrow * self.nlay
    xarr, yarr, zarr = _cxtgeo.grdcp3d_get_vtk_grid_arrays(
        self.ncol,
        self.nrow,
        self.nlay,
        self._coordsv,
        self._zcornsv,
        narr,
        narr,
        narr,
    )
    corners = np.stack((xarr, yarr, zarr))
    corners = corners.transpose()

    dims = np.asarray((self.ncol, self.nrow, self.nlay)) + 1

    actindices = self.get_actnum_indices(order="F", inverse=True)

    return dims, corners, actindices


def get_cell_volume(self, ijk=(1, 1, 1), activeonly=True, zerobased=False, precision=2):
    """Get bulk cell volume for one cell."""
    self._xtgformat1()

    i, j, k = ijk

    if precision not in (1, 2, 4):
        raise ValueError("The precision key has an invalid entry, use 1, 2, or 4")

    shift = 0
    if zerobased:
        shift = 1

    if activeonly:
        actnum = self.get_actnum()
        iact = actnum.values3d[i - 1 + shift, j - 1 + shift, k - 1 + shift]
        if iact == 0:
            return None

    pcorners = _cxtgeo.new_doublearray(24)

    if self._xtgformat == 1:
        logger.info("Use xtgformat 1...")
        _cxtgeo.grd3d_corners(
            i + shift,
            j + shift,
            k + shift,
            self.ncol,
            self.nrow,
            self.nlay,
            self._coordsv,
            self._zcornsv,
            pcorners,
        )
    else:
        logger.info("Use xtgformat 2...")
        _cxtgeo.grdcp3d_corners(
            i + shift - 1,
            j + shift - 1,
            k + shift - 1,
            self.ncol,
            self.nrow,
            self.nlay,
            self._coordsv,
            self._zcornsv,
            pcorners,
        )

    cellvol = _cxtgeo.x_hexahedron_volume(pcorners, 24, precision)
    return cellvol


def get_layer_slice(self, layer, top=True, activeonly=True):
    """Get X Y cell corners (XY per cell; 5 per cell) as array."""
    self._xtgformat1()
    ntot = self._ncol * self._nrow * self._nlay

    opt1 = 0
    if not top:
        opt1 = 1

    opt2 = 1
    if not activeonly:
        opt2 = 0

    icn, lay_array, ic_array = _cxtgeo.grd3d_get_lay_slice(
        self._ncol,
        self._nrow,
        self._nlay,
        self._coordsv,
        self._zcornsv,
        self._actnumsv,
        layer,
        opt1,
        opt2,
        10 * ntot,
        ntot,
    )

    lay_array = lay_array[: 10 * icn]
    ic_array = ic_array[:icn]

    lay_array = lay_array.reshape((icn, 5, 2))

    return lay_array, ic_array


def get_geometrics(self, allcells=False, cellcenter=True, return_dict=False, _ver=1):
    """Getting cell geometrics."""
    self._xtgformat1()

    if _ver == 1:
        res = _get_geometrics_v1(
            self, allcells=allcells, cellcenter=cellcenter, return_dict=return_dict
        )
    else:
        res = _get_geometrics_v2(
            self, allcells=allcells, cellcenter=cellcenter, return_dict=return_dict
        )
    return res


def _get_geometrics_v1(self, allcells=False, cellcenter=True, return_dict=False):

    ptr_x = []
    for i in range(13):
        ptr_x.append(_cxtgeo.new_doublepointer())

    option1 = 1
    if allcells:
        option1 = 0

    option2 = 1
    if not cellcenter:
        option2 = 0

    quality = _cxtgeo.grd3d_geometrics(
        self._ncol,
        self._nrow,
        self._nlay,
        self._coordsv,
        self._zcornsv,
        self._actnumsv,
        ptr_x[0],
        ptr_x[1],
        ptr_x[2],
        ptr_x[3],
        ptr_x[4],
        ptr_x[5],
        ptr_x[6],
        ptr_x[7],
        ptr_x[8],
        ptr_x[9],
        ptr_x[10],
        ptr_x[11],
        ptr_x[12],
        option1,
        option2,
    )

    glist = []
    for i in range(13):
        glist.append(_cxtgeo.doublepointer_value(ptr_x[i]))

    glist.append(quality)

    logger.info("Cell geometrics done")

    if return_dict:
        gdict = {}
        gkeys = [
            "xori",
            "yori",
            "zori",
            "xmin",
            "xmax",
            "ymin",
            "ymax",
            "zmin",
            "zmax",
            "avg_rotation",
            "avg_dx",
            "avg_dy",
            "avg_dz",
            "grid_regularity_flag",
        ]

        for i, key in enumerate(gkeys):
            gdict[key] = glist[i]

        return gdict

    return tuple(glist)


def _get_geometrics_v2(self, allcells=False, cellcenter=True, return_dict=False):
    # Currently a workaround as there seems to be bugs in v1
    # Will only work with allcells False and cellcenter True

    glist = []
    if cellcenter and allcells:
        xcor, ycor, zcor = self.get_xyz(asmasked=False)
        glist.append(xcor.values[0, 0, 0])
        glist.append(ycor.values[0, 0, 0])
        glist.append(zcor.values[0, 0, 0])
        glist.append(xcor.values.min())
        glist.append(xcor.values.max())
        glist.append(ycor.values.min())
        glist.append(ycor.values.max())
        glist.append(zcor.values.min())
        glist.append(zcor.values.max())

        # rotation (approx) for mid column
        midcol = int(self.nrow / 2)
        midlay = int(self.nlay / 2)
        x0 = xcor.values[0, midcol, midlay]
        y0 = ycor.values[0, midcol, midlay]
        x1 = xcor.values[self.ncol - 1, midcol, midlay]
        y1 = ycor.values[self.ncol - 1, midcol, midlay]
        glist.append(degrees(atan2(y1 - y0, x1 - x0)))

        dx, dy = self.get_dxdy(asmasked=False)
        dz = self.get_dz(asmasked=False)
        glist.append(dx.values.mean())
        glist.append(dy.values.mean())
        glist.append(dz.values.mean())
        glist.append(1)

    if return_dict:
        gdict = {}
        gkeys = [
            "xori",
            "yori",
            "zori",
            "xmin",
            "xmax",
            "ymin",
            "ymax",
            "zmin",
            "zmax",
            "avg_rotation",
            "avg_dx",
            "avg_dy",
            "avg_dz",
            "grid_regularity_flag",
        ]

        for i, key in enumerate(gkeys):
            gdict[key] = glist[i]

        return gdict

    return tuple(glist)


def inactivate_by_dz(self, threshold: float, flip: bool = True):
    """Set cell to inactive if dz does not exceed threshold.
    Args:
        threshold (float): The threshold for which the absolute value
            of dz should exceed.
        flip (bool): Whether the z-direction should be flipped.

    """
    self._xtgformat2()
    self._actnumsv[
        self.get_dz(asmasked=False, flip=flip).values.reshape(self._actnumsv.shape)
        < threshold
    ] = 0


def make_zconsistent(self, zsep):
    """Make consistent in z."""
    self._xtgformat1()

    if isinstance(zsep, int):
        zsep = float(zsep)

    if not isinstance(zsep, float):
        raise ValueError('The "zsep" is not a float or int')

    _cxtgeo.grd3d_make_z_consistent(
        self.ncol,
        self.nrow,
        self.nlay,
        self._zcornsv,
        zsep,
    )


def inactivate_inside(self, poly, layer_range=None, inside=True, force_close=False):
    """Inactivate inside a polygon (or outside)."""
    self._xtgformat1()

    if not isinstance(poly, Polygons):
        raise ValueError("Input polygon not a XTGeo Polygons instance")

    if layer_range is not None:
        k1, k2 = layer_range
    else:
        k1 = 1
        k2 = self.nlay

    method = 0
    if not inside:
        method = 1

    iforce = 0
    if force_close:
        iforce = 1

    # get dataframe where each polygon is ended by a 999 value
    dfxyz = poly.get_xyz_dataframe()

    xc = dfxyz["X_UTME"].values.copy()
    yc = dfxyz["Y_UTMN"].values.copy()

    ier = _cxtgeo.grd3d_inact_outside_pol(
        xc,
        yc,
        self.ncol,
        self.nrow,
        self.nlay,
        self._coordsv,
        self._zcornsv,
        self._actnumsv,  # is modified!
        k1,
        k2,
        iforce,
        method,
    )

    if ier == 1:
        raise RuntimeError("Problems with one or more polygons. " "Not closed?")


def collapse_inactive_cells(self):
    """Collapse inactive cells."""
    self._xtgformat1()

    _cxtgeo.grd3d_collapse_inact(
        self.ncol, self.nrow, self.nlay, self._zcornsv, self._actnumsv
    )


def copy(self):
    """Copy a grid instance (C pointers) and other props.

    Returns:
        A new instance (attached grid properties will also be unique)
    """
    self._xtgformat2()

    filesrc = None
    if self._filesrc is not None and "(copy)" not in self._filesrc:
        filesrc = self._filesrc + " (copy)"
    elif self._filesrc is not None:
        filesrc = self._filesrc

    return self.__class__(
        coordsv=self._coordsv.copy(),
        zcornsv=self._zcornsv.copy(),
        actnumsv=self._actnumsv.copy(),
        subgrids=deepcopy(self.subgrids),
        dualporo=self.dualporo,
        dualperm=self.dualperm,
        name=self.name + " (copy)" if self.name else None,
        roxgrid=self.roxgrid,
        roxindexer=self.roxindexer,
        props=self._props.copy() if self._props else None,
        filesrc=filesrc,
    )


def crop(self, spec, props=None):  # pylint: disable=too-many-locals
    """Do cropping of geometry (and properties).

    If props is 'all' then all properties assosiated (linked) to then
    grid are also cropped, and the instances are updated.

    Args:
        spec (tuple): A nested tuple on the form ((i1, i2), (j1, j2), (k1, k2))
            where 1 represents start number, and 2 reperesent end. The range
            is inclusive for both ends, and the number start index is 1 based.
        props (list or str): None is default, while properties can be listed.
            If 'all', then all GridProperty objects which are linked to the
            Grid instance are updated.

    Returns:
        The instance is updated (cropped)
    """
    self._xtgformat1()

    (ic1, ic2), (jc1, jc2), (kc1, kc2) = spec

    if (
        ic1 < 1
        or ic2 > self.ncol
        or jc1 < 1
        or jc2 > self.nrow
        or kc1 < 1
        or kc2 > self.nlay
    ):

        raise ValueError("Boundary for tuples not matching grid" "NCOL, NROW, NLAY")

    oldnlay = self._nlay

    # compute size of new cropped grid
    nncol = ic2 - ic1 + 1
    nnrow = jc2 - jc1 + 1
    nnlay = kc2 - kc1 + 1

    ntot = nncol * nnrow * nnlay
    ncoord = (nncol + 1) * (nnrow + 1) * 2 * 3
    nzcorn = nncol * nnrow * (nnlay + 1) * 4

    new_num_act = _cxtgeo.new_intpointer()
    new_coordsv = np.zeros(ncoord, dtype=np.float64)
    new_zcornsv = np.zeros(nzcorn, dtype=np.float64)
    new_actnumsv = np.zeros(ntot, dtype=np.int32)

    _cxtgeo.grd3d_crop_geometry(
        self.ncol,
        self.nrow,
        self.nlay,
        self._coordsv,
        self._zcornsv,
        self._actnumsv,
        new_coordsv,
        new_zcornsv,
        new_actnumsv,
        ic1,
        ic2,
        jc1,
        jc2,
        kc1,
        kc2,
        new_num_act,
        0,
    )

    self._coordsv = new_coordsv
    self._zcornsv = new_zcornsv
    self._actnumsv = new_actnumsv

    self._ncol = nncol
    self._nrow = nnrow
    self._nlay = nnlay

    if isinstance(self.subgrids, dict):
        newsub = OrderedDict()
        # easier to work with numpies than lists
        newarr = np.array(range(1, oldnlay + 1))
        newarr[newarr < kc1] = 0
        newarr[newarr > kc2] = 0
        newaxx = newarr.copy() - kc1 + 1
        for sub, arr in self.subgrids.items():
            arrx = np.array(arr)
            arrxmap = newaxx[arrx[0] - 1 : arrx[-1]]
            arrxmap = arrxmap[arrxmap > 0]
            if arrxmap.size > 0:
                newsub[sub] = arrxmap.astype(np.int32).tolist()

        self.subgrids = newsub

    # crop properties
    if props is not None:
        if props == "all":
            props = self.props

        for prop in props:
            logger.info("Crop %s", prop.name)
            prop.crop(spec)


def reduce_to_one_layer(self):
    """Reduce the grid to one single layer.

    This can be useful for algorithms that need to test if a point is within
    the full grid.

    Example::

        >>> import xtgeo
        >>> grid = xtgeo.grid_from_file(reek_dir + "/REEK.EGRID")
        >>> grid.nlay
        14
        >>> grid.reduce_to_one_layer()
        >>> grid.nlay
        1

    """
    # need new pointers in C (not for coord)
    # Note this could probably be done with pure numpy operations
    self._xtgformat1()

    ptr_new_num_act = _cxtgeo.new_intpointer()

    nnum = (1 + 1) * 4

    new_zcorn = np.zeros(self.ncol * self.nrow * nnum, dtype=np.float64)
    new_actnum = np.zeros(self.ncol * self.nrow * 1, dtype=np.int32)

    _cxtgeo.grd3d_reduce_onelayer(
        self.ncol,
        self.nrow,
        self.nlay,
        self._zcornsv,
        new_zcorn,
        self._actnumsv,
        new_actnum,
        ptr_new_num_act,
        0,
    )

    self._nlay = 1
    self._zcornsv = new_zcorn
    self._actnumsv = new_actnum
    self._props = None
    self._subgrids = None


def translate_coordinates(self, translate=(0, 0, 0), flip=(1, 1, 1)):
    """Translate grid coordinates."""
    self._xtgformat1()

    tx, ty, tz = translate
    fx, fy, fz = flip

    ier = _cxtgeo.grd3d_translate(
        self._ncol,
        self._nrow,
        self._nlay,
        fx,
        fy,
        fz,
        tx,
        ty,
        tz,
        self._coordsv,
        self._zcornsv,
    )
    if ier != 0:
        raise RuntimeError("Something went wrong in translate, code: {}".format(ier))

    logger.info("Translation of coords done")


def reverse_row_axis(self, ijk_handedness=None):
    """Reverse rows (aka flip) for geometry and assosiated properties."""
    self._xtgformat1()

    if ijk_handedness == self.ijk_handedness:
        return

    ier = _cxtgeo.grd3d_reverse_jrows(
        self._ncol,
        self._nrow,
        self._nlay,
        self._coordsv,
        self._zcornsv,
        self._actnumsv,
    )

    if ier != 0:
        raise RuntimeError("Something went wrong in jswapping, code: {}".format(ier))

    if self._props is None:
        return

    # do it for properties
    if self._props.props:
        for prp in self._props.props:
            prp.values = prp.values[:, ::-1, :]

    logger.info("Reversing of rows done")


def get_adjacent_cells(self, prop, val1, val2, activeonly=True):
    """Get adjacents cells."""
    self._xtgformat1()

    if not isinstance(prop, GridProperty):
        raise ValueError("The argument prop is not a xtgeo.GridPropery")

    if prop.isdiscrete is False:
        raise ValueError("The argument prop is not a discrete property")

    result = GridProperty(
        ncol=self._ncol,
        nrow=self._nrow,
        nlay=self._nlay,
        values=np.zeros(self.ntotal, dtype=np.int32),
        name="ADJ_CELLS",
        discrete=True,
    )

    p_prop1 = _gridprop_lowlevel.update_carray(prop)
    p_prop2 = _cxtgeo.new_intarray(self.ntotal)

    iflag1 = 1
    if activeonly:
        iflag1 = 0

    iflag2 = 1

    _cxtgeo.grd3d_adj_cells(
        self._ncol,
        self._nrow,
        self._nlay,
        self._coordsv,
        self._zcornsv,
        self._actnumsv,
        p_prop1,
        self.ntotal,
        val1,
        val2,
        p_prop2,
        self.ntotal,
        iflag1,
        iflag2,
    )

    _gridprop_lowlevel.update_values_from_carray(result, p_prop2, np.int32, delete=True)
    # return the property object
    return result


def estimate_design(self, nsubname):
    """Estimate (guess) (sub)grid design by examing DZ in median thickness column."""
    actv = self.get_actnum().values

    dzv = self.get_dz(asmasked=False).values

    # treat inactive thicknesses as zero
    dzv[actv == 0] = 0.0

    if nsubname is None:
        vrange = np.array(range(self.nlay))
    else:
        vrange = np.array(list(self.subgrids[nsubname])) - 1

    # find the dz for the actual subzone
    dzv = dzv[:, :, vrange]

    # find cumulative thickness as a 2D array
    dzcum = np.sum(dzv, axis=2, keepdims=False)

    # find the average thickness for nonzero thicknesses
    dzcum2 = dzcum.copy()
    dzcum2[dzcum == 0.0] = np.nan
    dzavg = np.nanmean(dzcum2) / dzv.shape[2]

    # find the I J indices for the median value
    argmed = np.stack(
        np.nonzero(dzcum == np.percentile(dzcum, 50, interpolation="nearest")), axis=1
    )
    im, jm = argmed[0]

    # find the dz stack of the median
    dzmedian = dzv[im, jm, :]
    logger.info("DZ median column is %s", dzmedian)

    # to compare thicknesses with (divide on 2 to assure)
    target = dzcum[im, jm] / (dzmedian.shape[0] * 2)
    eps = target / 100.0

    logger.info("Target and EPS values are %s, %s", target, eps)

    status = "X"  # unknown or cannot determine

    if dzmedian[0] > target and dzmedian[-1] <= eps:
        status = "T"
        dzavg = dzmedian[0]
    elif dzmedian[0] < eps and dzmedian[-1] > target:
        status = "B"
        dzavg = dzmedian[-1]
    elif dzmedian[0] > target and dzmedian[-1] > target:
        ratio = dzmedian[0] / dzmedian[-1]
        if 0.5 < ratio < 1.5:
            status = "P"
    elif dzmedian[0] < eps and dzmedian[-1] < eps:
        status = "M"
        middleindex = int(dzmedian.shape[0] / 2)
        dzavg = dzmedian[middleindex]

    return {"design": status, "dzsimbox": dzavg}


def estimate_flip(self):
    """Estimate if grid is left or right handed."""
    corners = self.get_xyz_cell_corners(activeonly=False)  # for cell 1, 1, 1

    v1 = (corners[3] - corners[0], corners[4] - corners[1], 0.0)
    v2 = (corners[6] - corners[0], corners[7] - corners[1], 0.0)

    flipvalue = find_flip(v1, v2)

    return flipvalue


def _convert_xtgformat2to1(self):
    """Convert arrays from new structure xtgformat=2 to legacy xtgformat=1."""
    if self._xtgformat == 1:
        logger.info("No conversion, format is already xtgformat == 1 or unset")
        return

    logger.info("Convert grid from new xtgformat to legacy format...")

    newcoordsv = np.zeros(((self._ncol + 1) * (self._nrow + 1) * 6), dtype=np.float64)
    newzcornsv = np.zeros(
        (self._ncol * self._nrow * (self._nlay + 1) * 4), dtype=np.float64
    )
    newactnumsv = np.zeros((self._ncol * self._nrow * self._nlay), dtype=np.int32)

    _cxtgeo.grd3cp3d_xtgformat2to1_geom(
        self._ncol,
        self._nrow,
        self._nlay,
        newcoordsv,
        self._coordsv,
        newzcornsv,
        self._zcornsv,
        newactnumsv,
        self._actnumsv,
    )

    self._coordsv = newcoordsv
    self._zcornsv = newzcornsv
    self._actnumsv = newactnumsv
    self._xtgformat = 1

    logger.info("Convert grid from new xtgformat to legacy format... done")


def _convert_xtgformat1to2(self):
    """Convert arrays from old structure xtgformat=1 to new xtgformat=2."""
    if self._xtgformat == 2 or self._coordsv is None:
        logger.info("No conversion, format is already xtgformat == 2 or unset")
        return

    logger.info("Convert grid from legacy xtgformat to new format...")

    newcoordsv = np.zeros((self._ncol + 1, self._nrow + 1, 6), dtype=np.float64)
    newzcornsv = np.zeros(
        (self._ncol + 1, self._nrow + 1, self._nlay + 1, 4), dtype=np.float32
    )
    newactnumsv = np.zeros((self._ncol, self._nrow, self._nlay), dtype=np.int32)

    _cxtgeo.grd3cp3d_xtgformat1to2_geom(
        self._ncol,
        self._nrow,
        self._nlay,
        self._coordsv,
        newcoordsv,
        self._zcornsv,
        newzcornsv,
        self._actnumsv,
        newactnumsv,
    )

    self._coordsv = newcoordsv
    self._zcornsv = newzcornsv
    self._actnumsv = newactnumsv
    self._xtgformat = 2

    logger.info("Convert grid from new xtgformat to legacy format... done")


def get_gridquality_properties(self):
    """Get the grid quality properties."""
    self._xtgformat2()

    qcnames = {
        0: "minangle_topbase",
        1: "maxangle_topbase",
        2: "minangle_topbase_proj",
        3: "maxangle_topbase_proj",
        4: "minangle_sides",
        5: "maxangle_sides",
        6: "collapsed",
        7: "faulted",
        8: "negative_thickness",
        9: "concave_proj",
    }

    # some of the properties shall be discrete:
    qcdiscrete = [6, 7, 8, 9]

    fresults = np.ones(
        (len(qcnames), self.ncol * self.nrow * self.nlay), dtype=np.float32
    )

    _cxtgeo.grdcp3d_quality_indicators(
        self.ncol,
        self.nrow,
        self.nlay,
        self._coordsv,
        self._zcornsv,
        self._actnumsv,
        fresults,
    )

    grdprops = xtgeo.GridProperties()

    for num, name in qcnames.items():
        prop = xtgeo.GridProperty(self, name=name)
        dtype = np.float32
        if num in qcdiscrete:
            dtype = np.int32
            prop.isdiscrete = True
            prop.codes = {0: "None", 1: name}
        prop.values = fresults[num, :].astype(dtype)
        grdprops.append_props([prop])

    return grdprops
