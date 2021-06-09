# coding: utf-8
"""Various operations"""

import numbers
import numpy as np
import numpy.ma as ma

import xtgeo
from xtgeo import XTGeoCLibError
from xtgeo.xyz import Polygons
import xtgeo.cxtgeo._cxtgeo as _cxtgeo  # type: ignore
from xtgeo.common import XTGeoDialog

xtg = XTGeoDialog()

logger = xtg.functionlogger(__name__)

VALID_OPER = (
    "add",
    "iadd",
    "sub",
    "isub",
    "mul",
    "imul",
    "div",
    "idiv",
    "lt",
    "gt",
    "le",
    "eq",
    "ne",
)

VALID_OPER_POLYS = (
    "add",
    "sub",
    "mul",
    "div",
    "set",
    "eli",
)


def operations_two(self, other, oper="add"):  # pylint: disable=too-many-branches
    """General operations between two maps"""

    other = _check_other(self, other)

    okstatus = self.compare_topology(other)

    useother = other
    if not okstatus:
        # to avoid that the "other" instance is changed
        useother = self.copy()
        useother.resample(other)

    if oper not in VALID_OPER:
        raise ValueError(f"Operation key oper has invalid value: {oper}")

    retvalue = None

    if oper == "add":
        self.values = self.values + useother.values
    elif oper == "iadd":
        self.values += useother.values
    elif oper == "sub":
        self.values = self.values - useother.values
    elif oper == "isub":
        self._values -= useother._values
    elif oper == "mul":
        self.values = self.values * useother.values
    elif oper == "imul":
        self._values *= useother._values
    elif oper == "div":
        self.values = self.values / useother.values
    elif oper == "idiv":
        self._values /= useother._values

    # comparisons:
    elif oper == "lt":
        retvalue = self.values < other.values
    elif oper == "gt":
        retvalue = self.values > other.values
    elif oper == "le":
        retvalue = self.values <= other.values
    elif oper == "ge":
        retvalue = self.values >= other.values
    elif oper == "eq":
        retvalue = self.values == other.values
    elif oper == "ne":
        retvalue = self.values != other.values

    if useother is not other:
        del useother

    self._filesrc = "Calculated"

    # return None or a boolean array
    return retvalue


def _check_other(self, other):
    """Will convert an other scalar to a valid numpy array"""

    if isinstance(other, numbers.Number):
        vals = other
        other = self.copy()
        other.values *= 0
        other.values += vals
        other._filesrc = None

    return other


def resample(self, other, mask=True):
    """Resample from other surface object to this surf."""

    logger.info("Resampling...")

    # a special case occur of the maps have same topology, but
    # different masks
    if self.compare_topology(other, strict=False):
        self.values = other.values.copy()
        return

    svalues = np.ma.filled(self.values, fill_value=xtgeo.UNDEF)
    ovalues = np.ma.filled(other.values, fill_value=xtgeo.UNDEF)

    _cxtgeo.surf_resample(
        other._ncol,
        other._nrow,
        other._xori,
        other._xinc,
        other._yori,
        other._yinc,
        other._yflip,
        other._rotation,
        ovalues,
        self._ncol,
        self._nrow,
        self._xori,
        self._xinc,
        self._yori,
        self._yinc,
        self._yflip,
        self._rotation,
        svalues,
        0 if not mask else 1,
    )

    self.values = np.ma.masked_greater(svalues, xtgeo.UNDEF_LIMIT)

    self.set_values1d(svalues)
    self._filesrc = "Resampled"


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
    )

    if ier != 0:
        logger.error("Something went wrong...")
        raise RuntimeError("Something went wrong in {}".format(__name__))

    self.set_values1d(svalues)


def get_value_from_xy(self, point=(0.0, 0.0), sampling="bilinear"):
    """Find surface value for point X Y."""

    xcoord, ycoord = point

    option = 0 if sampling == "bilinear" else 2

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
        option,
    )
    if zcoord > xtgeo.UNDEF_LIMIT:
        return None

    return zcoord


def get_xy_value_from_ij(self, iloc, jloc, zvalues=None):
    """Find X Y value from I J index"""

    if zvalues is None:
        zvalues = self.get_values1d()

    try:
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
        )
    except XTGeoCLibError:
        raise ValueError(f"Index i {iloc} and/or j {jloc} out of bounds")

    if value > xtgeo.UNDEF_LIMIT:
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
    )
    if ier != 0:
        raise XTGeoCLibError(f"Error in surf_xy_as_values, error code: {ier}")

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


def get_fence(self, xyfence, sampling="bilinear"):
    """Get surface values along fence."""

    cxarr = xyfence[:, 0]
    cyarr = xyfence[:, 1]
    czarr = xyfence[:, 2].copy()

    sampleoptions = {"bilinear": 0, "nearest": 2}

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
        sampleoptions.get(sampling, 0),
    )

    if istat != 0:
        logger.warning("Seem to be rotten")

    xyfence[:, 2] = czarr
    xyfence = ma.masked_greater(xyfence, xtgeo.UNDEF_LIMIT)
    xyfence = ma.mask_rows(xyfence)

    return xyfence


def get_randomline(
    self, fencespec, hincrement=None, atleast=5, nextend=2, sampling="bilinear"
):
    """Get surface values along fence."""

    if hincrement is None and isinstance(fencespec, xtgeo.Polygons):
        logger.info("Estimate hincrement from instance...")
        fencespec = _get_randomline_fence(self, fencespec, hincrement, atleast, nextend)
        logger.info("Estimate hincrement from instance... DONE")

    if fencespec is None or fencespec is False:
        return None

    sampleoptions = {"bilinear": 0, "nearest": 2}

    xcoords = fencespec[:, 0]
    ycoords = fencespec[:, 1]
    zcoords = fencespec[:, 2].copy()
    hcoords = fencespec[:, 3]

    # zcoords will be updated "inplace":
    istat = _cxtgeo.surf_get_zv_from_xyv(
        xcoords,
        ycoords,
        zcoords,
        self.ncol,
        self.nrow,
        self.xori,
        self.yori,
        self.xinc,
        self.yinc,
        self.yflip,
        self.rotation,
        self.get_values1d(),
        sampleoptions.get(sampling, 0),
    )

    if istat != 0:
        logger.warning("Seem to be rotten")

    zcoords[zcoords > xtgeo.UNDEF_LIMIT] = np.nan
    arr = np.vstack([hcoords, zcoords]).T

    return arr


def _get_randomline_fence(self, fencespec, hincrement, atleast, nextend):
    """Compute a resampled fence from a Polygons instance"""

    if hincrement is None:

        avgdxdy = 0.5 * (self.xinc + self.yinc)
        distance = 0.5 * avgdxdy
    else:
        distance = hincrement

    logger.info("Getting fence from a Polygons instance...")
    fspec = fencespec.get_fence(
        distance=distance, atleast=atleast, nextend=nextend, asnumpy=True
    )
    logger.info("Getting fence from a Polygons instance... DONE")
    return fspec


def operation_polygons(self, poly, value, opname="add", inside=True):
    """Operations restricted to polygons"""

    if not isinstance(poly, Polygons):
        raise ValueError("The poly input is not a Polygons instance")
    if opname not in VALID_OPER_POLYS:
        raise ValueError(f"Operation key opname has invalid value: {opname}")

    # make a copy of the RegularSurface which is used a "filter" or "proxy"
    # value will be 1 inside polygons, 0 outside. Undef cells are kept as is

    proxy = self.copy()
    proxy.values *= 0.0
    vals = proxy.get_values1d(fill_value=xtgeo.UNDEF)

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

    for _, grp in idgroups:
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
        )
        if ier == -9:
            xtg.warn("Polygon is not closed")

    proxy.set_values1d(vals)
    proxyv = proxy.values.astype(np.int8)

    proxytarget = 1
    if not inside:
        proxytarget = 0

    tmp = None
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
            tmp = np.where(np.isinf(tmp), 0, tmp)
            tmp = np.nan_to_num(tmp)
            tmp = ma.array(tmp, mask=mask)

    elif opname == "set":
        tmp = value
    elif opname == "eli":
        tmp = value * 0 + xtgeo.UNDEF
        tmp = ma.masked_greater(tmp, xtgeo.UNDEF_LIMIT)

    self.values[proxyv == proxytarget] = tmp[proxyv == proxytarget]
    del tmp
