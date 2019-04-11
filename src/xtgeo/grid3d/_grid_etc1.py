"""Private module, Grid ETC 1 methods, info/modify/report ...."""

from __future__ import print_function, absolute_import

import inspect
import warnings
from copy import deepcopy
from collections import OrderedDict

import numpy as np
import numpy.ma as ma

import xtgeo.cxtgeo.cxtgeo as _cxtgeo
from xtgeo.common import XTGeoDialog
from xtgeo.xyz.polygons import Polygons
from xtgeo.well.well import Well
from . import _gridprop_lowlevel
from .grid_property import GridProperty

xtg = XTGeoDialog()

logger = xtg.functionlogger(__name__)

_cxtgeo.xtg_verbose_file("NONE")
XTGDEBUG = xtg.get_syslevel()

# Note that "self" is the grid instance


def get_dz(self, name="dZ", flip=True, asmasked=True):
    """Get dZ as property"""
    ntot = (self._ncol, self._nrow, self._nlay)

    dzv = GridProperty(
        ncol=self._ncol,
        nrow=self._nrow,
        nlay=self._nlay,
        values=np.zeros(ntot, dtype=np.float64),
        name=name,
        discrete=False,
    )

    ptr_dz_v = _cxtgeo.new_doublearray(self.ntotal)

    nflip = 1
    if not flip:
        nflip = -1

    option = 0
    if asmasked:
        option = 1

    _cxtgeo.grd3d_calc_dz(
        self._ncol,
        self._nrow,
        self._nlay,
        self._p_zcorn_v,
        self._p_actnum_v,
        ptr_dz_v,
        nflip,
        option,
        XTGDEBUG,
    )

    _gridprop_lowlevel.update_values_from_carray(dzv, ptr_dz_v, np.float64, delete=True)
    # return the property object

    logger.info("DZ mean value: %s", dzv.values.mean())

    return dzv


def get_dxdy(self, names=("dX", "dY"), asmasked=False):
    """Get dX, dY as properties"""
    ntot = self._ncol * self._nrow * self._nlay
    dx = GridProperty(
        ncol=self._ncol,
        nrow=self._nrow,
        nlay=self._nlay,
        values=np.zeros(ntot, dtype=np.float64),
        name=names[0],
        discrete=False,
    )
    dy = GridProperty(
        ncol=self._ncol,
        nrow=self._nrow,
        nlay=self._nlay,
        values=np.zeros(ntot, dtype=np.float64),
        name=names[1],
        discrete=False,
    )

    ptr_dx_v = _cxtgeo.new_doublearray(self.ntotal)
    ptr_dy_v = _cxtgeo.new_doublearray(self.ntotal)

    option1 = 0
    option2 = 0

    if asmasked:
        option1 = 1

    _cxtgeo.grd3d_calc_dxdy(
        self._ncol,
        self._nrow,
        self._nlay,
        self._p_coord_v,
        self._p_zcorn_v,
        self._p_actnum_v,
        ptr_dx_v,
        ptr_dy_v,
        option1,
        option2,
        XTGDEBUG,
    )

    _gridprop_lowlevel.update_values_from_carray(dx, ptr_dx_v, np.float64, delete=True)
    _gridprop_lowlevel.update_values_from_carray(dy, ptr_dy_v, np.float64, delete=True)

    # return the property objects
    return dx, dy


def get_ijk(self, names=("IX", "JY", "KZ"), asmasked=True, zerobased=False):
    """Get I J K as properties"""

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


def get_xyz(self, names=("X_UTME", "Y_UTMN", "Z_TVDSS"), asmasked=True):
    """Get X Y Z as properties... May be issues with asmasked vs activeonly here"""

    ntot = self.ntotal

    x = GridProperty(
        ncol=self._ncol,
        nrow=self._nrow,
        nlay=self._nlay,
        values=np.zeros(ntot, dtype=np.float64),
        name=names[0],
        discrete=False,
    )

    y = GridProperty(
        ncol=self._ncol,
        nrow=self._nrow,
        nlay=self._nlay,
        values=np.zeros(ntot, dtype=np.float64),
        name=names[1],
        discrete=False,
    )

    z = GridProperty(
        ncol=self._ncol,
        nrow=self._nrow,
        nlay=self._nlay,
        values=np.zeros(ntot, dtype=np.float64),
        name=names[2],
        discrete=False,
    )

    ptr_x_v = _cxtgeo.new_doublearray(self.ntotal)
    ptr_y_v = _cxtgeo.new_doublearray(self.ntotal)
    ptr_z_v = _cxtgeo.new_doublearray(self.ntotal)

    option = 0
    if asmasked:
        option = 1

    _cxtgeo.grd3d_calc_xyz(
        self._ncol,
        self._nrow,
        self._nlay,
        self._p_coord_v,
        self._p_zcorn_v,
        self._p_actnum_v,
        ptr_x_v,
        ptr_y_v,
        ptr_z_v,
        option,
        XTGDEBUG,
    )

    _gridprop_lowlevel.update_values_from_carray(x, ptr_x_v, np.float64, delete=True)
    _gridprop_lowlevel.update_values_from_carray(y, ptr_y_v, np.float64, delete=True)
    _gridprop_lowlevel.update_values_from_carray(z, ptr_z_v, np.float64, delete=True)

    # Note: C arrays are deleted in the update_values_from_carray()

    return x, y, z


def get_xyz_cell_corners(self, ijk=(1, 1, 1), activeonly=True, zerobased=False):
    """Get X Y Z cell corners for one cell."""
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

    _cxtgeo.grd3d_corners(
        i + shift,
        j + shift,
        k + shift,
        self.ncol,
        self.nrow,
        self.nlay,
        self._p_coord_v,
        self._p_zcorn_v,
        pcorners,
        XTGDEBUG,
    )

    cornerlist = []
    for i in range(24):
        cornerlist.append(_cxtgeo.doublearray_getitem(pcorners, i))

    clist = tuple(cornerlist)
    return clist


def get_xyz_corners(self, names=("X_UTME", "Y_UTMN", "Z_TVDSS")):
    """Get X Y Z cell corners for all cells (as 24 GridProperty objects)"""
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
        self._p_coord_v,
        self._p_zcorn_v,
        self._p_actnum_v,
        *(ptr_coord + [option] + [XTGDEBUG])
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


def get_geometrics(self, allcells=False, cellcenter=True, return_dict=False):

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
        self._p_coord_v,
        self._p_zcorn_v,
        self._p_actnum_v,
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
        XTGDEBUG,
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


def inactivate_by_dz(self, threshold):
    """Inactivate by DZ"""
    if isinstance(threshold, int):
        threshold = float(threshold)

    if not isinstance(threshold, float):
        raise ValueError("The threshold is not a float or int")

    # assumption (unless somebody finds a Petrel made grid):
    nflip = 1

    _cxtgeo.grd3d_inact_by_dz(
        self.ncol,
        self.nrow,
        self.nlay,
        self._p_zcorn_v,
        self._p_actnum_v,
        threshold,
        nflip,
        XTGDEBUG,
    )


def make_zconsistent(self, zsep):
    """Make consistent in z"""
    if isinstance(zsep, int):
        zsep = float(zsep)

    if not isinstance(zsep, float):
        raise ValueError('The "zsep" is not a float or int')

    _cxtgeo.grd3d_make_z_consistent(
        self.ncol,
        self.nrow,
        self.nlay,
        self._p_zcorn_v,
        self._p_actnum_v,
        zsep,
        XTGDEBUG,
    )


def inactivate_inside(self, poly, layer_range=None, inside=True, force_close=False):
    """Inactivate inside a polygon (or outside)"""
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
        self._p_coord_v,
        self._p_zcorn_v,
        self._p_actnum_v,
        k1,
        k2,
        iforce,
        method,
        XTGDEBUG,
    )

    if ier == 1:
        raise RuntimeError("Problems with one or more polygons. " "Not closed?")


def collapse_inactive_cells(self):
    """Collapse inactive cells"""
    _cxtgeo.grd3d_collapse_inact(
        self.ncol, self.nrow, self.nlay, self._p_zcorn_v, self._p_actnum_v, XTGDEBUG
    )


def copy(self):
    """Copy a grid instance (C pointers) and other props.

    Returns:
        A new instance (attached grid properties will also be unique)
    """

    other = self.__class__()

    ntot = self.ncol * self.nrow * self.nlay
    ncoord = (self.ncol + 1) * (self.nrow + 1) * 2 * 3
    nzcorn = self.ncol * self.nrow * (self.nlay + 1) * 4

    new_p_coord_v = _cxtgeo.new_doublearray(ncoord)
    new_p_zcorn_v = _cxtgeo.new_doublearray(nzcorn)
    new_p_actnum_v = _cxtgeo.new_intarray(ntot)

    _cxtgeo.grd3d_copy(
        self.ncol,
        self.nrow,
        self.nlay,
        self._p_coord_v,
        self._p_zcorn_v,
        self._p_actnum_v,
        new_p_coord_v,
        new_p_zcorn_v,
        new_p_actnum_v,
        0,
        XTGDEBUG,
    )

    other._p_coord_v = new_p_coord_v
    other._p_zcorn_v = new_p_zcorn_v
    other._p_actnum_v = new_p_actnum_v

    other._ncol = self.ncol
    other._nrow = self.nrow
    other._nlay = self.nlay

    if isinstance(self.subgrids, dict):
        other.subgrids = deepcopy(self.subgrids)

    # copy attached properties
    if self._props:
        other._props = self._props.copy()
        logger.info("Other vs self props %s vs %s", other._props, self._props)

    if self._filesrc is not None and "(copy)" not in self._filesrc:
        other._filesrc = self._filesrc + " (copy)"
    elif self._filesrc is not None:
        other._filesrc = self._filesrc

    return other


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
    new_p_coord_v = _cxtgeo.new_doublearray(ncoord)
    new_p_zcorn_v = _cxtgeo.new_doublearray(nzcorn)
    new_p_actnum_v = _cxtgeo.new_intarray(ntot)

    _cxtgeo.grd3d_crop_geometry(
        self.ncol,
        self.nrow,
        self.nlay,
        self._p_coord_v,
        self._p_zcorn_v,
        self._p_actnum_v,
        new_p_coord_v,
        new_p_zcorn_v,
        new_p_actnum_v,
        ic1,
        ic2,
        jc1,
        jc2,
        kc1,
        kc2,
        new_num_act,
        0,
        XTGDEBUG,
    )

    self._p_coord_v = new_p_coord_v
    self._p_zcorn_v = new_p_zcorn_v
    self._p_actnum_v = new_p_actnum_v

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

        >>> from xtgeo.grid3d import Grid
        >>> gf = Grid('gullfaks2.roff')
        >>> gf.nlay
        47
        >>> gf.reduce_to_one_layer()
        >>> gf.nlay
        1

    """

    # need new pointers in C (not for coord)

    ptr_new_num_act = _cxtgeo.new_intpointer()

    nnum = (1 + 1) * 4
    ptr_new_zcorn_v = _cxtgeo.new_doublearray(self.ncol * self.nrow * nnum)

    ptr_new_actnum_v = _cxtgeo.new_intarray(self.ncol * self.nrow * 1)

    _cxtgeo.grd3d_reduce_onelayer(
        self.ncol,
        self.nrow,
        self.nlay,
        self._p_zcorn_v,
        ptr_new_zcorn_v,
        self._p_actnum_v,
        ptr_new_actnum_v,
        ptr_new_num_act,
        0,
        XTGDEBUG,
    )

    self._nlay = 1
    self._p_zcorn_v = ptr_new_zcorn_v
    self._p_actnum_v = ptr_new_actnum_v
    self._props = None
    self._subgrids = None


def translate_coordinates(self, translate=(0, 0, 0), flip=(1, 1, 1)):
    """Translate grid coordinates"""
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
        self._p_coord_v,
        self._p_zcorn_v,
        XTGDEBUG,
    )
    if ier != 0:
        raise RuntimeError("Something went wrong in translate, code: {}".format(ier))

    logger.info("Translation of coords done")


def report_zone_mismatch(  # pylint: disable=too-many-statements
    self,
    well=None,
    zonelogname="ZONELOG",
    zoneprop=None,
    onelayergrid=None,
    zonelogrange=(0, 9999),
    zonelogshift=0,
    depthrange=None,
    option=0,
    perflogname=None,
):
    """Reports well to zone mismatch; this works together with a Well object."""

    this = inspect.currentframe().f_code.co_name

    # first do some trimming of the well dataframe
    if not well or not isinstance(well, Well):
        msg = "No well object in <{}> or invalid object; " "returns no result".format(
            this
        )
        warnings.warn(UserWarning(msg))
        return None

    # qperf = True
    if perflogname == "None" or perflogname is None:
        # qperf = False
        pass
    else:
        if perflogname not in well.lognames:
            msg = (
                "Asked for perf log <{}> but no such in <{}> for well {}; "
                "return None".format(perflogname, this, well.wellname)
            )
            warnings.warn(UserWarning(msg))
            return None

    logger.info("Process well object for %s...", well.wellname)
    df = well.dataframe.copy()

    if depthrange:
        logger.info("Filter depth...")
        df = df[df.Z_TVDSS > depthrange[0]]
        df = df[df.Z_TVDSS < depthrange[1]]
        df = df.copy()

    logger.info("Adding zoneshift %s", zonelogshift)
    if zonelogshift != 0:
        df[zonelogname] += zonelogshift

    logger.info("Filter ZONELOG...")
    df = df[df[zonelogname] > zonelogrange[0]]
    df = df[df[zonelogname] < zonelogrange[1]]
    df = df.copy()

    if perflogname:
        logger.info("Filter PERF...")
        df[perflogname].fillna(-999, inplace=True)
        df = df[df[perflogname] > 0]
        df = df.copy()

    df.reset_index(drop=True, inplace=True)
    well.dataframe = df

    # get the relevant well log C arrays...
    ptr_xc = well.get_carray("X_UTME")
    ptr_yc = well.get_carray("Y_UTMN")
    ptr_zc = well.get_carray("Z_TVDSS")
    ptr_zo = well.get_carray(zonelogname)

    nval = well.nrow

    ptr_results = _cxtgeo.new_doublearray(10)

    ptr_zprop = _gridprop_lowlevel.update_carray(zoneprop)

    cstatus = _cxtgeo.grd3d_rpt_zlog_vs_zon(
        self._ncol,
        self._nrow,
        self._nlay,
        self._p_coord_v,
        self._p_zcorn_v,
        self._p_actnum_v,
        ptr_zprop,
        nval,
        ptr_xc,
        ptr_yc,
        ptr_zc,
        ptr_zo,
        zonelogrange[0],
        zonelogrange[1],
        onelayergrid._p_zcorn_v,
        onelayergrid._p_actnum_v,
        ptr_results,
        option,
        XTGDEBUG,
    )

    _gridprop_lowlevel.delete_carray(zoneprop, ptr_zprop)

    if cstatus == 0:
        logger.debug("OK well")
    elif cstatus == 2:
        msg = "Well {} have no zonation?".format(well.wellname)
        warnings.warn(msg, UserWarning)
    else:
        msg = "Something is rotten with {}".format(well.wellname)
        raise SystemExit(msg)

    # extract the report
    perc = _cxtgeo.doublearray_getitem(ptr_results, 0)
    tpoi = _cxtgeo.doublearray_getitem(ptr_results, 1)
    mpoi = _cxtgeo.doublearray_getitem(ptr_results, 2)

    # returns percent match, then total numbers of well counts for zone,
    # then match count. perc = mpoi/tpoi
    return (perc, int(tpoi), int(mpoi))


def get_adjacent_cells(self, prop, val1, val2, activeonly=True):
    """Get adjacents cells"""
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
        self._p_coord_v,
        self._p_zcorn_v,
        self._p_actnum_v,
        p_prop1,
        self.ntotal,
        val1,
        val2,
        p_prop2,
        self.ntotal,
        iflag1,
        iflag2,
        XTGDEBUG,
    )

    _gridprop_lowlevel.update_values_from_carray(result, p_prop2, np.int32, delete=True)
    # return the property object
    return result
