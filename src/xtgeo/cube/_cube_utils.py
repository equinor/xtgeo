"""Cube utilities (basic low level)"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np

from xtgeo import _cxtgeo
from xtgeo._cxtgeo import XTGeoCLibError
from xtgeo.common.calc import _swap_axes
from xtgeo.common.constants import UNDEF_LIMIT
from xtgeo.common.log import null_logger
from xtgeo.xyz.polygons import Polygons

if TYPE_CHECKING:
    from .cube1 import Cube

logger = null_logger(__name__)


def swapaxes(cube: Cube) -> None:
    """Pure numpy/python version"""
    cube._rotation, cube._yflip, swapped_values = _swap_axes(
        cube._rotation,
        cube._yflip,
        values=cube._values,
        traceidcodes=cube._traceidcodes,
    )
    cube._ncol, cube._nrow = cube._nrow, cube._ncol
    cube._xinc, cube._yinc = cube._yinc, cube._xinc
    cube.values = swapped_values["values"]
    cube._traceidcodes = swapped_values["traceidcodes"]


def thinning(cube: Cube, icol: int, jrow: int, klay: int) -> None:
    inputs = [icol, jrow, klay]
    ranges = [cube.nrow, cube.ncol, cube.nlay]

    for inum, ixc in enumerate(inputs):
        if not isinstance(ixc, int):
            raise ValueError(f"Some input is not integer: {inputs}")
        if ixc > ranges[inum] / 2:
            raise ValueError(
                f"Input numbers <{inputs}> are too large compared to existing "
                f"ranges <{ranges}>"
            )

    # just simple numpy operations, and changing some cube props

    val = cube.values.copy()

    val = val[::icol, ::jrow, ::klay]
    cube._ncol = val.shape[0]
    cube._nrow = val.shape[1]
    cube._nlay = val.shape[2]
    cube._xinc *= icol
    cube._yinc *= jrow
    cube._zinc *= klay
    cube._ilines = cube._ilines[::icol]
    cube._xlines = cube._xlines[::jrow]
    cube._traceidcodes = cube._traceidcodes[::icol, ::jrow]

    cube.values = val


def cropping(
    cube: Cube,
    icols: tuple[int, int],
    jrows: tuple[int, int],
    klays: tuple[int, int],
) -> None:
    """Cropping, where inputs are tuples"""

    icol1, icol2 = icols
    jrow1, jrow2 = jrows
    klay1, klay2 = klays

    val = cube.values.copy()
    ncol = cube.ncol
    nrow = cube.nrow
    nlay = cube.nlay

    val = val[
        0 + icol1 : ncol - icol2, 0 + jrow1 : nrow - jrow2, 0 + klay1 : nlay - klay2
    ]

    cube._ncol = val.shape[0]
    cube._nrow = val.shape[1]
    cube._nlay = val.shape[2]

    cube._ilines = cube._ilines[0 + icol1 : ncol - icol2]
    cube._xlines = cube._xlines[0 + jrow1 : nrow - jrow2]
    cube.traceidcodes = cube.traceidcodes[
        0 + icol1 : ncol - icol2, 0 + jrow1 : nrow - jrow2
    ]

    # 1 + .., since the following routine as 1 as base for i j
    ier, xpp, ypp = _cxtgeo.cube_xy_from_ij(
        1 + icol1,
        1 + jrow1,
        cube.xori,
        cube.xinc,
        cube.yori,
        cube.yinc,
        ncol,
        nrow,
        cube.yflip,
        cube.rotation,
        0,
    )

    if ier != 0:
        raise RuntimeError(f"Unexpected error, code is {ier}")

    # get new X Y origins
    cube._xori = xpp
    cube._yori = ypp
    cube._zori = cube.zori + klay1 * cube.zinc

    cube.values = val


def resample(
    cube: Cube,
    other: Cube,
    sampling: str = "nearest",
    outside_value: float | None = None,
) -> None:
    """Resample another cube to the current cube"""
    # TODO: traceidcodes

    values1a = cube.values.reshape(-1)
    values2a = other.values.reshape(-1)

    logger.info("Resampling, using %s...", sampling)

    ier = _cxtgeo.cube_resample_cube(
        cube.ncol,
        cube.nrow,
        cube.nlay,
        cube.xori,
        cube.xinc,
        cube.yori,
        cube.yinc,
        cube.zori,
        cube.zinc,
        cube.rotation,
        cube.yflip,
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
        warnings.warn("Less than 10% of original cube sampled", RuntimeWarning)
    elif ier != 0:
        raise XTGeoCLibError("cube_resample_cube failed to complete")


def get_xy_value_from_ij(
    cube: Cube,
    iloc: int,
    jloc: int,
    ixline: bool = False,
    zerobased: bool = False,
) -> tuple[float, float]:
    """Find X Y value from I J index, or corresponding inline/xline"""
    # assumes that inline follows I and xlines follows J

    iuse = iloc
    juse = jloc

    if zerobased:
        iuse = iuse + 1
        juse = juse + 1

    if ixline:
        ilst = cube.ilines.tolist()
        jlst = cube.xlines.tolist()
        iuse = ilst.index(iloc) + 1
        juse = jlst.index(jloc) + 1

    if 1 <= iuse <= cube.ncol and 1 <= juse <= cube.nrow:
        ier, xval, yval = _cxtgeo.cube_xy_from_ij(
            iuse,
            juse,
            cube.xori,
            cube.xinc,
            cube.yori,
            cube.yinc,
            cube.ncol,
            cube.nrow,
            cube._yflip,
            cube.rotation,
            0,
        )
        if ier != 0:
            raise XTGeoCLibError(f"cube_xy_from_ij failed with error code: {ier}")

    else:
        raise ValueError("Index i and/or j out of bounds")

    return xval, yval


def get_randomline(
    cube: Cube,
    fencespec: np.ndarray | Polygons,
    zmin: float | None = None,
    zmax: float | None = None,
    zincrement: float | None = None,
    hincrement: float | None = None,
    atleast: int = 5,
    nextend: int = 2,
    sampling: str = "nearest",
) -> tuple[float, float, float, float, np.ndarray]:
    """Get a random line from a fence specification"""

    if isinstance(fencespec, Polygons):
        logger.info("Estimate hincrement from Polygons instance...")
        fencespec = _get_randomline_fence(cube, fencespec, hincrement, atleast, nextend)
        logger.info("Estimate hincrement from Polygons instance... DONE")

    if not len(fencespec.shape) == 2:
        raise ValueError("Fence is not a 2D numpy")

    xcoords = fencespec[:, 0]
    ycoords = fencespec[:, 1]
    hcoords = fencespec[:, 3]

    for ino in range(hcoords.shape[0] - 1):
        dhv = hcoords[ino + 1] - hcoords[ino]
        logger.info("Delta H along well path: %s", dhv)

    zcubemax = cube._zori + (cube._nlay - 1) * cube._zinc
    if zmin is None or zmin < cube._zori:
        zmin = cube._zori

    if zmax is None or zmax > zcubemax:
        zmax = zcubemax

    if zincrement is None:
        zincrement = cube._zinc / 2.0

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
        cube._xori,
        cube._xinc,
        cube._yori,
        cube._yinc,
        cube._zori,
        cube._zinc,
        cube._rotation,
        cube._yflip,
        cube._ncol,
        cube._nrow,
        cube._nlay,
        cube._values.reshape(-1),
        nsamples,
        option,
    )

    values[values > UNDEF_LIMIT] = np.nan
    arr = values.reshape((xcoords.shape[0], nzsam)).T

    return (hcoords[0], hcoords[-1], zmin, zmax, arr)


def _get_randomline_fence(
    cube: Cube,
    fencespec: Polygons,
    hincrement: float | None,
    atleast: int,
    nextend: int,
) -> np.ndarray:
    """Compute a resampled fence from a Polygons instance"""

    if hincrement is None:
        avgdxdy = 0.5 * (cube.xinc + cube.yinc)
        distance = 0.5 * avgdxdy
    else:
        distance = hincrement

    logger.info("Getting fence from a Polygons instance...")

    tempname = fencespec.name
    fspec = fencespec.get_fence(
        distance=distance, atleast=atleast, nextend=nextend, asnumpy=True
    )

    # get_fence() can return a bool, but a valid cube fence must be an np.ndarray.
    if isinstance(fspec, bool):
        raise ValueError(f"Too few points in polygons for fence, name: {tempname}")

    logger.info("Getting fence from a Polygons instance... DONE")

    if not isinstance(fspec, np.ndarray):
        raise ValueError(
            "Expected a numpy array from polygons.get_fence(),"
            f"but got type {type(fspec)}."
        )
    return fspec
