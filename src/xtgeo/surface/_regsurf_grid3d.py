"""Regular surface vs Grid3D"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from xtgeo import _cxtgeo, _internal
from xtgeo.common.log import null_logger
from xtgeo.grid3d import _gridprop_lowlevel
from xtgeo.grid3d.grid_property import GridProperty

if TYPE_CHECKING:
    from xtgeo.grid3d.grid import Grid
    from xtgeo.surface.regular_surface import RegularSurface


# self = RegularSurface instance!

logger = null_logger(__name__)


def slice_grid3d(
    self: RegularSurface,
    grid: Grid,
    prop: GridProperty,
    zsurf: RegularSurface | None = None,
    sbuffer: int = 1,
):
    """Private function for the Grid3D slicing."""

    grid._xtgformat1()
    if zsurf is not None:
        other = zsurf
    else:
        logger.info('The current surface is copied as "other"')
        other = self.copy()
    if not self.compare_topology(other, strict=False):
        raise RuntimeError("Topology of maps differ. Stop!")

    zslice = other.copy()

    nsurf = self.ncol * self.nrow

    p_prop = _gridprop_lowlevel.update_carray(prop, discrete=False)

    istat, updatedval = _cxtgeo.surf_slice_grd3d(
        self.ncol,
        self.nrow,
        self.xori,
        self.xinc,
        self.yori,
        self.yinc,
        self.rotation,
        self.yflip,
        zslice.get_values1d(),
        nsurf,
        grid.ncol,
        grid.nrow,
        grid.nlay,
        grid._coordsv,
        grid._zcornsv,
        grid._actnumsv,
        p_prop,
        sbuffer,
    )

    if istat != 0:
        logger.warning("Problem, ISTAT = %s", istat)

    self.set_values1d(updatedval)

    return istat


@dataclass
class DeriveSurfaceFromGrid3D:
    """Private class; derive a surface from a 3D grid / gridproperty."""

    grid: Grid
    template: RegularSurface | str | None = None
    where: str | int = "top"
    property: str | GridProperty = "depth"
    rfactor: float = 1.0

    # private state variables
    _args: dict | None = None
    _tempsurf: RegularSurface | None = None
    _klayer: int | None = None
    _option: int | None = None

    def __post_init__(self) -> None:
        logger.debug("DeriveSurfaceFromGrid3D: __post_init__")

        self._parametrize_regsurf()
        self._generate_working_surface()
        self._eval_where()

        xtgformat_convert = self.grid._xtgformat == 1
        if xtgformat_convert:
            self.grid._xtgformat2()

        self._sample_regsurf_from_grid3d()

        # convert back to original xtgformat due to unforseen consequences
        if xtgformat_convert:
            self.grid._xtgformat1()

    def result(self) -> dict:
        args = {}
        args["xori"] = self._tempsurf.xori
        args["yori"] = self._tempsurf.yori
        args["xinc"] = self._tempsurf.xinc
        args["yinc"] = self._tempsurf.yinc
        args["ncol"] = self._tempsurf.ncol
        args["nrow"] = self._tempsurf.nrow
        args["rotation"] = self._tempsurf.rotation
        args["yflip"] = self._tempsurf.yflip
        args["values"] = self._tempsurf.values.copy()

        return args

    def _parametrize_regsurf(self) -> None:
        """Parametrize setting for the regular surface based on template and grid."""
        args = {}
        if self.template is None:
            # need to estimate map settings from the existing 3D grid
            geom = self.grid.get_geometrics(
                allcells=True, cellcenter=True, return_dict=True, _ver=2
            )

            xlen = 1.1 * (geom["xmax"] - geom["xmin"])
            ylen = 1.1 * (geom["ymax"] - geom["ymin"])
            xori = geom["xmin"] - 0.05 * xlen
            yori = geom["ymin"] - 0.05 * ylen
            # take same xinc and yinc

            xinc = yinc = (1.0 / self.rfactor) * 0.5 * (geom["avg_dx"] + geom["avg_dy"])
            ncol = int(xlen / xinc)
            nrow = int(ylen / yinc)

            args["xori"] = xori
            args["yori"] = yori
            args["xinc"] = xinc
            args["yinc"] = yinc
            args["ncol"] = ncol
            args["nrow"] = nrow
            args["values"] = np.ma.zeros((ncol, nrow), dtype=np.float64)
        elif isinstance(self.template, str) and self.template == "native":
            geom = self.grid.get_geometrics(
                allcells=True, cellcenter=True, return_dict=True
            )

            args["ncol"] = self.grid.ncol
            args["nrow"] = self.grid.nrow
            args["xori"] = geom["xori"]
            args["yori"] = geom["yori"]
            args["xinc"] = geom["avg_dx"]
            args["yinc"] = geom["avg_dy"]
            args["rotation"] = geom["avg_rotation"]
            args["values"] = np.ma.zeros(
                (self.grid.ncol, self.grid.nrow), dtype=np.float64
            )
            args["yflip"] = -1 if self.grid.ijk_handedness == "right" else 1

        else:
            args["xori"] = self.template.xori
            args["yori"] = self.template.yori
            args["xinc"] = self.template.xinc
            args["yinc"] = self.template.yinc
            args["ncol"] = self.template.ncol
            args["nrow"] = self.template.nrow
            args["rotation"] = self.template.rotation
            args["yflip"] = self.template.yflip
            args["values"] = self.template.values.copy()

        self._args = args

    def _generate_working_surface(self) -> None:
        from xtgeo.surface.regular_surface import RegularSurface

        self._tempsurf = RegularSurface(**self._args)
        # ensure that this is a subsurface left-handed system
        self._tempsurf.make_lefthanded()

    def _eval_where(self):
        """Set klayer and option based on ``where`` parameter.

        Note that klayer is zero-based, while where is one-based.
        """
        if isinstance(self.where, str):
            where = self.where.lower()
            if where == "top":
                self._klayer = 0
                self._option = 0
            elif where == "base":
                self._klayer = self.grid.nlay - 1
                self._option = 1
            else:
                klayer, what = where.split("_")
                self._klayer = int(klayer) - 1
                if self._klayer < 0 or self._klayer >= self.grid.nlay:
                    raise ValueError(f"Klayer out of range in where={where}")
                self._option = 0
                if what == "base":
                    self._option = 1
        else:
            self._klayer = self.where - 1
            self._option = 0

    def _sample_regsurf_from_grid3d(self) -> None:
        """Sample the grid3d to get the values for the regular surface."""

        iindex, jindex, depth = _internal.regsurf.sample_grid3d_layer(
            self._tempsurf.ncol,
            self._tempsurf.nrow,
            self._tempsurf.xori,
            self._tempsurf.yori,
            self._tempsurf.xinc,
            self._tempsurf.yinc,
            self._tempsurf.rotation,
            self.grid.ncol,
            self.grid.nrow,
            self.grid.nlay,
            self.grid._coordsv,
            self.grid._zcornsv,
            self.grid._actnumsv,
            self._klayer,
            self._option,
            0 if isinstance(self.property, str) else 1,  # activeonly flag
        )

        mask = iindex == -1
        iindex = np.where(iindex == -1, 0, iindex)
        jindex = np.where(jindex == -1, 0, jindex)

        if isinstance(self.property, str):
            if self.property == "depth":
                self._tempsurf.values = depth
            elif self.property == "i":
                self._tempsurf.values = iindex
            elif self.property == "j":
                self._tempsurf.values = jindex
                self._tempsurf.values.mask = mask
            else:
                raise ValueError(f"Unknown property: {self.property}")

        elif isinstance(self.property, GridProperty):
            self._tempsurf.values = self.property.values[iindex, jindex, self._klayer]

        self._tempsurf.values.mask = mask


def from_grid3d(grid, template=None, where="top", property="depth", rfactor=1) -> dict:
    """Private function for deriving a surface from a 3D grid / gridproperty."""

    return DeriveSurfaceFromGrid3D(grid, template, where, property, rfactor).result()
