"""Roxar API functions for XTGeo Grid Geometry."""

from __future__ import annotations

import os
import tempfile
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

import xtgeo._internal as _internal
from xtgeo.common import XTGeoDialog, null_logger
from xtgeo.roxutils._roxar_loader import roxar, roxar_grids
from xtgeo.roxutils.roxutils import RoxUtils

xtg = XTGeoDialog()

logger = null_logger(__name__)

if TYPE_CHECKING:
    from xtgeo.grid3d.grid import Grid
    # from xtgeo.roxutils._roxar_loader import RoxarGrid3DType

    if roxar is not None:
        from roxar import grids as RoxarGridType
        from roxar.grids import Grid3D as RoxarGrid3DType


# self is Grid() instance


# ======================================================================================
# Load/import
# ======================================================================================


def load_grid_from_rms(
    projectname: str, gname: str, realisation: int, info: bool
) -> dict[str, Any]:
    """Load a Grid via ROXAR API spec and convert to XTGeo internal storage.

    Returns:
        dictionary of parameters to be used in the Grid constructor function.

    """

    rox = RoxUtils(projectname, readonly=True)
    return _load_grid_from_rms_viaroxapi(rox, gname, realisation, info)


def _display_roxapi_grid_info(roxgrid: RoxarGrid3DType) -> None:
    """Push info to screen (mostly for debugging), experimental."""

    indexer = roxgrid.grid_indexer
    ncol, nrow, _ = indexer.dimensions

    xtg.say("ROXAPI with support for CornerPointGridGeometry")
    geom = roxgrid.get_geometry()
    defined_cells = geom.get_defined_cells()
    xtg.say(f"Defined cells \n{defined_cells}")

    xtg.say(f"IJK handedness: {geom.ijk_handedness}")
    for ipi in range(ncol + 1):
        for jpi in range(nrow + 1):
            tpi, bpi, zco = geom.get_pillar_data(ipi, jpi)
            xtg.say(f"For pillar {ipi}, {jpi}\n")
            xtg.say(f"Tops\n{tpi}")
            xtg.say(f"Bots\n{bpi}")
            xtg.say(f"Depths\n{zco}")


def _load_grid_from_rms_viaroxapi(
    rox: RoxUtils, gname: str, realisation: int, info: bool
) -> dict[str, Any]:
    """Import a Grid via ROXAR API spec."""
    proj = rox.project

    logger.info("Loading grid with realisation %s...", realisation)
    try:
        if gname not in proj.grid_models:
            raise KeyError(f"No such gridmodel: {gname}")

        logger.info("Get roxgrid...")
        roxgrid = proj.grid_models[gname].get_grid(realisation=realisation)

        if roxgrid.has_dual_index_system:
            xtg.warnuser(
                f"The roxar grid {gname} has dual index system.\n"
                "XTGeo does not implement extraction of simbox grid\n"
                "and only considers physical index."
            )

        if info:
            _display_roxapi_grid_info(roxgrid)

        result = _convert_to_xtgeo_grid(roxgrid, gname)

    except KeyError as keyerror:
        raise RuntimeError(keyerror)

    if rox._roxexternal:
        rox.safe_close()

    return result


def _convert_to_xtgeo_grid(roxgrid: RoxarGrid3DType, gname: str) -> dict[str, Any]:
    """Convert from roxar CornerPointGeometry to xtgeo, version 2 using _xtgformat=2."""
    indexer = roxgrid.grid_indexer

    ncol, nrow, nlay = indexer.dimensions

    # update other attributes
    result: dict[str, Any] = {}

    nncol = ncol + 1
    nnrow = nrow + 1
    nnlay = nlay + 1

    result["name"] = gname

    geom = roxgrid.get_geometry()

    coordsv = np.zeros((nncol, nnrow, 6), dtype=np.float64)
    zcornsv = np.zeros((nncol, nnrow, nnlay, 4), dtype=np.float32)
    actnumsv = np.zeros((ncol, nrow, nlay), dtype=np.int32)
    for icol in range(nncol):
        for jrow in range(nnrow):
            topc, basc, zcorn = geom.get_pillar_data(icol, jrow)
            coordsv[icol, jrow, 0:3] = topc
            coordsv[icol, jrow, 3:6] = basc
            zcorn = np.ma.filled(zcorn, fill_value=np.nan)
            zcornsv[icol, jrow, :, :] = zcorn.T

    median_zcornsv = np.nanmedian(zcornsv)
    # replace nan with median of the zorner values
    zcornsv[np.isnan(zcornsv)] = median_zcornsv

    _internal.grid3d.process_edges_rmsapi(zcornsv)

    result["coordsv"] = coordsv
    result["zcornsv"] = zcornsv

    actnumsv[geom.get_defined_cells()] = 1
    result["actnumsv"] = actnumsv

    # subgrids
    if len(indexer.zonation) > 1:
        logger.debug("Zonation length (N subzones) is %s", len(indexer.zonation))
        subz = {}
        for inum, zrange in indexer.zonation.items():
            logger.debug("inum: %s, zrange: %s", inum, zrange)
            zname = roxgrid.zone_names[inum]
            logger.debug("zname is: %s", zname)
            zra = [nn + 1 for ira in zrange for nn in ira]  # nested lists
            subz[zname] = zra

        result["subgrids"] = subz

    result["roxgrid"] = roxgrid
    result["roxindexer"] = indexer

    return result


# ======================================================================================
# Save/export
# ======================================================================================


def save_grid_to_rms(
    self: Grid,
    projectname: str,
    gname: str,
    realisation: int,
    info: bool = False,
    method: str | Literal["cpg", "roff"] = "cpg",
) -> None:
    """Save (i.e. store in RMS) via RMSAPI (former ROXAPI) spec.

    Using method 'cpg' means that the CPG method is applied (from ROXAPI 1.3).
    This is possible from version ROXAPI ver 1.3, where the CornerPointGeometry
    class is defined.

    An alternative is to use simple roff import (via some /tmp area),
    can be used from version 1.2. The "roff" method is also better if the user
    want to activate undefined cells as a part of the work flow.

    """
    rox = RoxUtils(projectname, readonly=False)

    if method == "cpg":
        self._set_xtgformat2()
        _save_grid_to_rms_cornerpoint(self, rox, gname, realisation, info)

    else:
        _save_grid_to_rms_viaroff(self, rox, gname, realisation)

    if rox._roxexternal:
        rox.project.save()

    rox.safe_close()


def _save_grid_to_rms_cornerpoint(
    self: Grid, rox: RoxUtils, gname: str, realisation: int, info: bool
) -> None:
    """Convert xtgeo geometry to pillar spec in ROXAPI and store _xtgformat=2."""

    grid_model = rox.project.grid_models.create(gname)
    grid_model.set_empty(realisation)
    grid = grid_model.get_grid(realisation)

    roxar_grids_: RoxarGridType = roxar_grids  # for mypy
    geom = roxar_grids_.CornerPointGridGeometry.create(self.dimensions)

    grid_cpp = self._get_grid_cpp()
    tpi, bpi, zco, zma = grid_cpp.convert_xtgeo_to_rmsapi()
    zco = np.ma.array(zco, mask=zma)

    # NOTE (KEEP), it is a bit unclear if mask is needed. A possible simpilification is
    # zco = self._zcornsv.astype(np.float64)
    # zco = np.moveaxis(zco, 2, 3)
    # tpi = self._coordsv[:, :, 0:3]
    # bpi = self._coordsv[:, :, 3:6]
    # This would be 10-20% faster, but the mask is kept for now.

    for ipi in range(self.ncol + 1):
        for jpi in range(self.nrow + 1):
            geom.set_pillar_data(
                ipi,
                jpi,
                top_point=tpi[ipi, jpi],
                bottom_point=bpi[ipi, jpi],
                depth_values=zco[ipi, jpi],
            )

    geom.set_defined_cells(self._actnumsv.astype(bool))
    grid.set_geometry(geom)
    _set_subgrids(self, rox, grid)


def _set_subgrids(self: Grid, rox: RoxUtils, grid: RoxarGrid3DType) -> None:
    """Export the subgrid index (zones) to Roxar API.

    From roxar API:
        set_zonation(zone_dict)

            zone_dict A dictionary with start-layer (zero based) and name for each zone.

    """

    if not self.subgrids:
        return

    if rox.version_required("1.6"):
        subs = self.subgrids
        roxar_subs = {}
        for name, zrange in subs.items():
            roxar_subs[int(zrange[0] - 1)] = name

        grid.set_zonation(roxar_subs)

    else:
        xtg.warnuser(
            "Implementation of subgrids is lacking in Roxar API for this "
            "RMS version. Will continue to store in RMS but without subgrid index."
        )


def _save_grid_to_rms_viaroff(
    self: Grid, rox: RoxUtils, gname: str, realisation: int
) -> None:
    """Save xtgeo geometry to internal RMS via i/o ROFF tricks."""
    logger.info("Realisation is %s", realisation)

    # make a temporary folder and work within the with.. block
    with tempfile.TemporaryDirectory() as tmpdir:
        logger.info("Made a tmp folder: %s", tmpdir)

        fname = os.path.join(tmpdir, gname)
        self.to_file(fname)

        grd = rox.project.grid_models

        try:
            del grd[gname]
            logger.info("Overwriting existing grid in RMS")
        except KeyError:
            logger.info("Grid in RMS is new")

        grd.load(fname)
