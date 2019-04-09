# coding: utf-8
"""Various operations"""
from __future__ import print_function, absolute_import

import numpy as np
import numpy.ma as ma

from xtgeo.xyz import Polygons
import xtgeo.cxtgeo.cxtgeo as _cxtgeo
from xtgeo.common import XTGeoDialog

xtg = XTGeoDialog()

logger = xtg.functionlogger(__name__)


XTGDEBUG = xtg.get_syslevel()
if XTGDEBUG < 0:
    XTGDEBUG = 0

_cxtgeo.xtg_verbose_file("NONE")
# pylint: disable=protected-access


def operations_two(self, other, oper="add"):
    """General operations between two maps"""

    okstatus = self.compare_topology(other)

    if not okstatus:
        other.resample(self)

    if oper == "add":
        self.values = self.values + other.values

    if oper == "sub":
        self.values = self.values - other.values

    if oper == "mul":
        self.values = self.values * other.values

    if oper == "div":
        self.values = self.values / other.values


def resample(self, other):
    """Resample from other surface object to this surf"""

    logger.info("Resampling...")

    svalues = self.get_values1d()

    ier = _cxtgeo.surf_resample(
        other._ncol,
        other._nrow,
        other._xori,
        other._xinc,
        other._yori,
        other._yinc,
        other._yflip,
        other._rotation,
        other.get_values1d(),
        self._ncol,
        self._nrow,
        self._xori,
        self._xinc,
        self._yori,
        self._yinc,
        self._yflip,
        self._rotation,
        svalues,
        0,
        XTGDEBUG,
    )

    if ier != 0:
        raise RuntimeError("Resampling went wrong, " "code is {}".format(ier))

    self.set_values1d(svalues)


def distance_from_point(self, point=(0, 0), azimuth=0.0):
    """Find distance bwteen point and surface."""
    xpv, ypv = point

    svalues = self.get_values1d()

    # call C routine
    ier = _cxtgeo.surf_get_dist_values(
        self._xori,
        self._xinc,
        self._yori,
        self._yinc,
        self._ncol,
        self._nrow,
        self._rotation,
        xpv,
        ypv,
        azimuth,
        svalues,
        0,
        XTGDEBUG,
    )

    if ier != 0:
        logger.error("Something went wrong...")
        raise RuntimeError("Something went wrong in {}".format(__name__))

    self.set_values1d(svalues)


def get_value_from_xy(self, point=(0.0, 0.0)):
    """Find surface value for point X Y."""

    xcoord, ycoord = point

    zcoord = _cxtgeo.surf_get_z_from_xy(
        float(xcoord),
        float(ycoord),
        self.ncol,
        self.nrow,
        self.xori,
        self.yori,
        self.xinc,
        self.yinc,
        self.yflip,
        self.rotation,
        self.get_values1d(),
        XTGDEBUG,
    )

    if zcoord > self._undef_limit:
        return None

    return zcoord


def get_xy_value_from_ij(self, iloc, jloc, zvalues=None):
    """Find X Y value from I J index"""

    if zvalues is None:
        zvalues = self.get_values1d()

    if 1 <= iloc <= self.ncol and 1 <= jloc <= self.nrow:

        ier, xval, yval, value = _cxtgeo.surf_xyz_from_ij(
            iloc,
            jloc,
            self.xori,
            self.xinc,
            self.yori,
            self.yinc,
            self.ncol,
            self.nrow,
            self._yflip,
            self.rotation,
            zvalues,
            0,
            XTGDEBUG,
        )
        if ier != 0:
            logger.critical("Error code %s, contact the author", ier)
            raise SystemExit("Error code {}".format(ier))

    else:
        raise ValueError("Index i and/or j out of bounds")

    if value > self.undef_limit:
        value = None

    return xval, yval, value


def get_ij_values(self, zero_based=False, order="C", asmasked=False):
    """Get I J values as numpy 2D arrays.

    Args:
        zero_based (bool): If True, first index is 0. False (1) is default.
        order (str): 'C' or 'F' order (row vs column major)

    """

    ixn, jyn = np.indices((self._ncol, self._nrow))

    if order == "F":
        ixn = np.asfortranarray(ixn)
        jyn = np.asfortranarray(jyn)

    if not zero_based:
        ixn += 1
        jyn += 1

    if asmasked:
        ixn = ixn[~self.values.mask]
        jyn = ixn[~self.values.mask]

    return ixn, jyn


def get_ij_values1d(self, zero_based=False, activeonly=True, order="C"):
    """Get I J values as numpy 1D arrays.

    Args:
        zero_based (bool): If True, first index is 0. False (1) is default.
        activeonly (bool): If True, only for active nodes
        order (str): 'C' or 'F' order (row vs column major)

    """

    ixn, jyn = self.get_ij_values(zero_based=zero_based, order=order)

    ixn = ixn.ravel(order="K")
    jyn = jyn.ravel(order="K")

    if activeonly:
        tmask = ma.getmaskarray(self.get_values1d(order=order, asmasked=True))
        ixn = ma.array(ixn, mask=tmask)
        ixn = ixn[~ixn.mask]
        jyn = ma.array(jyn, mask=tmask)
        jyn = jyn[~jyn.mask]

    return ixn, jyn


def get_xy_values(self, order="C", asmasked=False):
    """Get X Y coordinate values as numpy 2D arrays."""
    nno = self.ncol * self.nrow

    ier, xvals, yvals = _cxtgeo.surf_xy_as_values(
        self.xori,
        self.xinc,
        self.yori,
        self.yinc * self.yflip,
        self.ncol,
        self.nrow,
        self.rotation,
        nno,
        nno,
        0,
        XTGDEBUG,
    )
    if ier != 0:
        logger.critical("Error code %s, contact the author", ier)

    # reshape
    xvals = xvals.reshape((self.ncol, self.nrow))
    yvals = yvals.reshape((self.ncol, self.nrow))

    if order == "F":
        xvals = np.array(xvals, order="F")
        yvals = np.array(yvals, order="F")

    if asmasked:
        tmpv = ma.filled(self.values, fill_value=np.nan)
        tmpv = np.array(tmpv, order=order)
        tmpv = ma.masked_invalid(tmpv)
        mymask = ma.getmaskarray(tmpv)
        xvals = ma.array(xvals, mask=mymask, order=order)
        yvals = ma.array(yvals, mask=mymask, order=order)

    return xvals, yvals


def get_xy_values1d(self, order="C", activeonly=True):
    """Get X Y coordinate values as numpy 1D arrays."""

    asmasked = False
    if activeonly:
        asmasked = True

    xvals, yvals = self.get_xy_values(order=order, asmasked=asmasked)

    xvals = xvals.ravel(order="K")
    yvals = yvals.ravel(order="K")

    if activeonly:
        xvals = xvals[~xvals.mask]
        yvals = yvals[~yvals.mask]

    return xvals, yvals


def get_fence(self, xyfence):
    """Get surface values along fence."""

    cxarr = xyfence[:, 0]
    cyarr = xyfence[:, 1]
    czarr = xyfence[:, 2].copy()

    # czarr will be updated "inplace":
    istat = _cxtgeo.surf_get_zv_from_xyv(
        cxarr,
        cyarr,
        czarr,
        self.ncol,
        self.nrow,
        self.xori,
        self.yori,
        self.xinc,
        self.yinc,
        self.yflip,
        self.rotation,
        self.get_values1d(),
        XTGDEBUG,
    )

    if istat != 0:
        logger.warning("Seem to be rotten")

    xyfence[:, 2] = czarr
    xyfence = ma.masked_greater(xyfence, self._undef_limit)
    xyfence = ma.mask_rows(xyfence)

    return xyfence


def operation_polygons(self, poly, value, opname="add", inside=True):
    """Operations restricted to polygons"""

    if not isinstance(poly, Polygons):
        raise ValueError("The poly input is not a Polygons instance")

    # make a copy of the RegularSurface which is used a "filter" or "proxy"
    # value will be 1 inside polygons, 0 outside. Undef cells are kept as is

    proxy = self.copy()
    proxy.values *= 0.0
    vals = proxy.get_values1d(fill_value=self.undef)

    # value could be a scalar or another surface; if another surface,
    # must ensure same topology

    if isinstance(value, type(self)):
        if not self.compare_topology(value):
            raise ValueError("Input is RegularSurface, but not same map " "topology")
        value = value.values.copy()
    else:
        # turn scalar value into numpy array
        value = self.values.copy() * 0 + value

    idgroups = poly.dataframe.groupby(poly.pname)

    for _id, grp in idgroups:
        xcor = grp[poly.xname].values
        ycor = grp[poly.yname].values

        ier = _cxtgeo.surf_setval_poly(
            proxy.xori,
            proxy.xinc,
            proxy.yori,
            proxy.yinc,
            proxy.ncol,
            proxy.nrow,
            proxy.yflip,
            proxy.rotation,
            vals,
            xcor,
            ycor,
            1.0,
            0,
            XTGDEBUG,
        )
        if ier == -9:
            xtg.warn("Polygon is not closed")

    proxy.set_values1d(vals)
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
            this = ma.filled(self.values, fill_value=1.0)
            that = ma.filled(value, fill_value=1.0)
            mask = ma.getmaskarray(self.values)
            tmp = np.true_divide(this, that)
            tmp[tmp == np.inf] = 0
            tmp = np.nan_to_num(tmp)
            tmp = ma.array(tmp, mask=mask)

    elif opname == "set":
        tmp = value
    elif opname == "eli":
        tmp = value * 0 + self.undef
        tmp = ma.masked_greater(tmp, self.undef_limit)

    self.values[proxyv == proxytarget] = tmp[proxyv == proxytarget]
    del tmp
