"""Cube utilities (basic low level)"""
import sys
import warnings

import numpy as np

import xtgeo
import xtgeo.cxtgeo._cxtgeo as _cxtgeo
from xtgeo.common import XTGeoDialog

xtg = XTGeoDialog()


logger = xtg.functionlogger(__name__)
# pylint: disable=protected-access


def swapaxes(self, _algorithm=1):

    if _algorithm == 1:
        _swapaxes_v1(self)
    else:
        _swapaxes_v2(self)


def _swapaxes_v1(self):
    """Swap the axes inline vs xline, keep origin."""
    # TODO remove this when algo 2 works."""
    ncol = _cxtgeo.new_intpointer()
    nrow = _cxtgeo.new_intpointer()
    yflip = _cxtgeo.new_intpointer()
    xinc = _cxtgeo.new_doublepointer()
    yinc = _cxtgeo.new_doublepointer()
    rota = _cxtgeo.new_doublepointer()

    _cxtgeo.intpointer_assign(ncol, self._ncol)
    _cxtgeo.intpointer_assign(nrow, self._nrow)
    _cxtgeo.intpointer_assign(yflip, self._yflip)

    _cxtgeo.doublepointer_assign(xinc, self._xinc)
    _cxtgeo.doublepointer_assign(yinc, self._yinc)
    _cxtgeo.doublepointer_assign(rota, self._rotation)

    values1d = self.values.reshape(-1)
    traceid1d = self._traceidcodes.reshape(-1)

    ier = _cxtgeo.cube_swapaxes(
        ncol,
        nrow,
        self.nlay,
        yflip,
        xinc,
        yinc,
        rota,
        values1d,
        traceid1d,
    )
    if ier != 0:
        raise Exception

    self._ncol = _cxtgeo.intpointer_value(ncol)
    self._nrow = _cxtgeo.intpointer_value(nrow)
    self._yflip = _cxtgeo.intpointer_value(yflip)

    self._xinc = _cxtgeo.doublepointer_value(xinc)
    self._yinc = _cxtgeo.doublepointer_value(yinc)
    self._rotation = _cxtgeo.doublepointer_value(rota)

    ilines = self._xlines.copy()
    xlines = self._ilines.copy()
    self._xlines = xlines
    self._ilines = ilines

    self._traceidcodes = traceid1d.reshape((self._ncol, self._nrow))
    self._values = values1d.reshape((self._ncol, self._nrow, self.nlay))


def _swapaxes_v2(self):
    """Pure numpy/python version, still issues"""
    newncol = self.nrow
    newnrow = self.ncol

    yflip = self.yflip * -1

    rotation = self.rotation + self.yflip * 90
    if rotation < 0:
        rotation += 360
    if rotation >= 360:
        rotation -= 360

    ilines = self._xlines.copy()
    xlines = self._ilines.copy()

    self._ncol = newncol
    self._nrow = newnrow
    self.values = self._values.transpose(1, 0, 2)
    self._yflip = yflip
    self._rotation = rotation
    self._ilines = ilines
    self._xlines = xlines
    self._traceidcodes = np.array(self._traceidcodes.flatten(order="K")).reshape(
        (self._ncol, self._nrow)
    )


def thinning(self, icol, jrow, klay):

    inputs = [icol, jrow, klay]
    ranges = [self.nrow, self.ncol, self.nlay]

    for inum, ixc in enumerate(inputs):
        if not isinstance(ixc, int):
            raise ValueError("Some input is not integer: {}".format(inputs))
        if ixc > ranges[inum] / 2:
            raise ValueError(
                "Input numbers <{}> are too large compared to "
                "existing ranges <{}>".format(inputs, ranges)
            )

    # just simple numpy operations, and changing some cube props

    val = self.values.copy()

    val = val[::icol, ::jrow, ::klay]
    self._ncol = val.shape[0]
    self._nrow = val.shape[1]
    self._nlay = val.shape[2]
    self._xinc *= icol
    self._yinc *= jrow
    self._zinc *= klay
    self._ilines = self._ilines[::icol]
    self._xlines = self._xlines[::jrow]
    self._traceidcodes = self._traceidcodes[::icol, ::jrow]

    self.values = val


def cropping(self, icols, jrows, klays):
    """Cropping, where inputs are tuples"""

    icol1, icol2 = icols
    jrow1, jrow2 = jrows
    klay1, klay2 = klays

    val = self.values.copy()
    ncol = self.ncol
    nrow = self.nrow
    nlay = self.nlay

    val = val[
        0 + icol1 : ncol - icol2, 0 + jrow1 : nrow - jrow2, 0 + klay1 : nlay - klay2
    ]

    self._ncol = val.shape[0]
    self._nrow = val.shape[1]
    self._nlay = val.shape[2]

    self._ilines = self._ilines[0 + icol1 : ncol - icol2]
    self._xlines = self._xlines[0 + jrow1 : nrow - jrow2]
    self.traceidcodes = self.traceidcodes[
        0 + icol1 : ncol - icol2, 0 + jrow1 : nrow - jrow2
    ]

    # 1 + .., since the following routine as 1 as base for i j
    ier, xpp, ypp = _cxtgeo.cube_xy_from_ij(
        1 + icol1,
        1 + jrow1,
        self.xori,
        self.xinc,
        self.yori,
        self.yinc,
        ncol,
        nrow,
        self.yflip,
        self.rotation,
        0,
    )

    if ier != 0:
        raise RuntimeError("Unexpected error, code is {}".format(ier))

    # get new X Y origins
    self._xori = xpp
    self._yori = ypp
    self._zori = self.zori + klay1 * self.zinc

    self.values = val


def resample(self, other, sampling="nearest", outside_value=None):
    """Resample another cube to the current self"""
    # TODO: traceidcodes

    values1a = self.values.reshape(-1)
    values2a = other.values.reshape(-1)

    logger.info("Resampling, using %s...", sampling)

    ier = _cxtgeo.cube_resample_cube(
        self.ncol,
        self.nrow,
        self.nlay,
        self.xori,
        self.xinc,
        self.yori,
        self.yinc,
        self.zori,
        self.zinc,
        self.rotation,
        self.yflip,
        values1a,
        other.ncol,
        other.nrow,
        other.nlay,
        other.xori,
        other.xinc,
        other.yori,
        other.yinc,
        other.zori,
        other.zinc,
        other.rotation,
        other.yflip,
        values2a,
        1 if sampling == "trilinear" else 0,
        0 if outside_value is None else 1,
        0 if outside_value is None else outside_value,
    )

    if ier == -4:
        warnings.warn("Less than 10% of origonal cube sampled", RuntimeWarning)

    if ier == -5:
        raise ValueError("No cube overlap in sampling")

    logger.info("Resampling done!")


def get_xy_value_from_ij(self, iloc, jloc, ixline=False, zerobased=False):
    """Find X Y value from I J index, or corresponding inline/xline"""
    # assumes that inline follows I and xlines follows J

    iuse = iloc
    juse = jloc

    if zerobased:
        iuse = iuse + 1
        juse = juse + 1

    if ixline:
        ilst = self.ilines.tolist()
        jlst = self.xlines.tolist()
        iuse = ilst.index(iloc) + 1
        juse = jlst.index(jloc) + 1

    if 1 <= iuse <= self.ncol and 1 <= juse <= self.nrow:

        ier, xval, yval = _cxtgeo.cube_xy_from_ij(
            iuse,
            juse,
            self.xori,
            self.xinc,
            self.yori,
            self.yinc,
            self.ncol,
            self.nrow,
            self._yflip,
            self.rotation,
            0,
        )
        if ier != 0:
            logger.critical("Error code %s, contact the author", ier)
            raise SystemExit("Error code {}".format(ier))

    else:
        raise ValueError("Index i and/or j out of bounds")

    return xval, yval


def get_randomline(
    self,
    fencespec,
    zmin=None,
    zmax=None,
    zincrement=None,
    hincrement=None,
    atleast=5,
    nextend=2,
    sampling="nearest",
):
    """Get a random line from a fence spesification"""

    if isinstance(fencespec, xtgeo.Polygons):
        logger.info("Estimate hincrement from Polygons instance...")
        fencespec = _get_randomline_fence(self, fencespec, hincrement, atleast, nextend)
        logger.info("Estimate hincrement from Polygons instance... DONE")

    if not len(fencespec.shape) == 2:
        raise ValueError("Fence is not a 2D numpy")

    xcoords = fencespec[:, 0]
    ycoords = fencespec[:, 1]
    hcoords = fencespec[:, 3]

    for ino in range(hcoords.shape[0] - 1):
        dhv = hcoords[ino + 1] - hcoords[ino]
        logger.info("Delta H along well path: %s", dhv)

    zcubemax = self._zori + (self._nlay - 1) * self._zinc
    if zmin is None or zmin < self._zori:
        zmin = self._zori

    if zmax is None or zmax > zcubemax:
        zmax = zcubemax

    if zincrement is None:
        zincrement = self._zinc / 2.0

    nzsam = int((zmax - zmin) / zincrement) + 1

    nsamples = xcoords.shape[0] * nzsam

    option = 0
    if sampling == "trilinear":
        option = 1

    _ier, values = _cxtgeo.cube_get_randomline(
        xcoords,
        ycoords,
        zmin,
        zmax,
        nzsam,
        self._xori,
        self._xinc,
        self._yori,
        self._yinc,
        self._zori,
        self._zinc,
        self._rotation,
        self._yflip,
        self._ncol,
        self._nrow,
        self._nlay,
        self._values.reshape(-1),
        nsamples,
        option,
    )

    values[values > xtgeo.UNDEF_LIMIT] = np.nan
    arr = values.reshape((xcoords.shape[0], nzsam)).T

    return (hcoords[0], hcoords[-1], zmin, zmax, arr)


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


# copy (update) values from SWIG carray to numpy, 3D array, Fortran order
# to be DEPRECATED
def update_values(cube):

    if cube._cvalues is None and cube._values is None:
        logger.critical("Something is wrong. STOP!")
        sys.exit(9)

    elif cube._cvalues is None:
        return cube._values, None

    logger.debug("Updating numpy values...")
    ncrl = cube._ncol * cube._nrow * cube._nlay
    xvv = _cxtgeo.swig_carr_to_numpy_f1d(ncrl, cube._cvalues)

    xvv = np.reshape(xvv, (cube._ncol, cube._nrow, cube._nlay), order="F")

    logger.debug("Updating numpy values... done")

    xtype = xvv.dtype
    logger.info("VALUES of type %s", xtype)

    # free the C values (to save memory)
    _cxtgeo.delete_floatarray(cube._cvalues)

    return xvv, None
