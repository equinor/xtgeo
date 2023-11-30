# -*- coding: utf-8 -*-

"""Some grid utilities, file scanning etc."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from xtgeo import _cxtgeo  # type: ignore
from xtgeo.common import null_logger
from xtgeo.common.constants import UNDEF_LIMIT
from xtgeo.grid3d import _gridprop_lowlevel as gl
from xtgeo.surface import _regsurf_lowlevel as rl
from xtgeo.surface.regular_surface import surface_from_grid3d
from xtgeo.xyz import Polygons

if TYPE_CHECKING:
    from xtgeo.grid3d import Grid, GridProperty


logger = null_logger(__name__)


def get_randomline(
    self: Grid,
    fencespec: np.ndarray | Polygons,
    prop: str | GridProperty,
    zmin: float | int | None = None,
    zmax: float | int | None = None,
    zincrement: float | int = 1.0,
    hincrement: float | int | None = None,
    atleast: int = 5,
    nextend: int = 2,
) -> tuple[float, float, float, float, np.ndarray]:
    """Extract a randomline from a 3D grid.

    This is a difficult task, in particular in terms of acceptable speed.
    """

    logger.info("Enter get_randomline from Grid...")

    _update_tmpvars(self)

    if hincrement is not None and not isinstance(hincrement, (float, int)):
        raise TypeError("'hincrement' can only be a number or None")

    if isinstance(fencespec, Polygons):
        fencespec = _get_randomline_fence(self, fencespec, hincrement, atleast, nextend)

    logger.info("Get property...")
    if isinstance(prop, str):
        grid_prop = self.get_prop_by_name(prop)
        if grid_prop is None:
            raise ValueError(f"No property with name {prop} was found in grid")
        prop = grid_prop

    assert isinstance(fencespec, np.ndarray)
    xcoords = fencespec[:, 0]
    ycoords = fencespec[:, 1]
    hcoords = fencespec[:, 3]

    if zmin is None:
        zmin = self._tmp["topd"].values.min()
    if zmax is None:
        zmax = self._tmp["basd"].values.max()

    nzsam = int((zmax - zmin) / float(zincrement)) + 1
    nsamples = xcoords.shape[0] * nzsam

    logger.info("Running C routine to get randomline...")
    self._xtgformat1()
    _ier, values = _cxtgeo.grd3d_get_randomline(
        xcoords,
        ycoords,
        zmin,
        zmax,
        nzsam,
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
        gl.update_carray(prop, dtype=np.float64),
        self._tmp["onegrid"]._zcornsv,
        self._tmp["onegrid"]._actnumsv,
        nsamples,
    )

    logger.info("Running C routine to get randomline... DONE")

    values[values > UNDEF_LIMIT] = np.nan
    arr = values.reshape((xcoords.shape[0], nzsam)).T

    logger.info("Getting randomline... DONE")
    return (hcoords[0], hcoords[-1], zmin, zmax, arr)


def _update_tmpvars(self: Grid, force: bool = False) -> None:
    """The self._tmp variables are needed to speed up calculations.

    If they are already created, the no need to recreate
    """
    if "onegrid" not in self._tmp or force:
        logger.info("Make a tmp onegrid instance...")
        self._tmp["onegrid"] = self.copy()
        self._tmp["onegrid"].reduce_to_one_layer()
        one = self._tmp["onegrid"]
        logger.info("Make a tmp onegrid instance... DONE")
        logger.info("Make a set of tmp surfaces for I J locations + depth...")
        self._tmp["topd"] = surface_from_grid3d(
            one, where="top", mode="depth", rfactor=4
        )
        self._tmp["topi"] = surface_from_grid3d(one, where="top", mode="i", rfactor=4)
        self._tmp["topj"] = surface_from_grid3d(one, where="top", mode="j", rfactor=4)
        self._tmp["basd"] = surface_from_grid3d(
            one, where="base", mode="depth", rfactor=4
        )
        self._tmp["basi"] = surface_from_grid3d(one, where="base", mode="i", rfactor=4)
        self._tmp["basj"] = surface_from_grid3d(one, where="base", mode="j", rfactor=4)

        self._tmp["topi"].fill()
        self._tmp["topj"].fill()
        self._tmp["basi"].fill()
        self._tmp["basj"].fill()

        self._tmp["topi_carr"] = rl.get_carr_double(self._tmp["topi"])
        self._tmp["topj_carr"] = rl.get_carr_double(self._tmp["topj"])
        self._tmp["basi_carr"] = rl.get_carr_double(self._tmp["basi"])
        self._tmp["basj_carr"] = rl.get_carr_double(self._tmp["basj"])

        logger.info("Make a set of tmp surfaces for I J locations + depth... DONE")
    else:
        logger.info("Re-use existing onegrid and tmp surfaces for I J")


def _get_randomline_fence(
    self: Grid,
    polygon: Polygons,
    distance: float | int | None,
    atleast: int,
    nextend: int,
) -> np.ndarray:
    """Compute a resampled fence from a Polygons instance."""
    if distance is None:
        logger.debug("Estimate fence distance from grid resolution...")
        geom = self.get_geometrics()

        avgdxdy = 0.5 * (geom[10] + geom[11])
        distance = 0.5 * avgdxdy
        logger.debug("Estimate fence distance from grid resolution... DONE")

    logger.debug("Getting fence from a Polygons instance...")
    fspec = polygon.get_fence(
        distance=distance, atleast=atleast, nextend=nextend, asnumpy=True
    )
    logger.debug("Getting fence from a Polygons instance... DONE")
    return fspec
