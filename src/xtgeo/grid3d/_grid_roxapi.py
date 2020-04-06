# -*- coding: utf-8 -*-
"""Roxar API functions for XTGeo Grid Geometry"""
import os
import tempfile
from collections import OrderedDict

import numpy as np

import xtgeo
import xtgeo.cxtgeo._cxtgeo as _cxtgeo

from xtgeo.common import XTGeoDialog
from xtgeo import RoxUtils

xtg = XTGeoDialog()

logger = xtg.functionlogger(__name__)

# logger.info(roxmsg)

# self is Grid() instance


# =============================================================================
# Load/import
# =============================================================================


def import_grid_roxapi(
    self, projectname, gname, realisation, dimonly, info
):  # pragma: no cover
    """Import a Grid via ROXAR API spec."""

    rox = RoxUtils(projectname, readonly=True)

    proj = rox.project

    logger.info("Loading grid with realisation %s...", realisation)
    try:
        if gname not in proj.grid_models:
            raise KeyError("No such gridmodel: {}".format(gname))

        logger.info("Get roxgrid...")
        roxgrid = proj.grid_models[gname].get_grid()

        if dimonly:
            corners = None
        else:
            logger.info("Get corners...")
            corners = roxgrid.get_cell_corners_by_index()
            logger.info("Get corners... done")

        if info:
            _display_roxapi_grid_info(rox, roxgrid)

        _convert_to_xtgeo_grid(self, rox, roxgrid, corners, gname)

    except KeyError as keyerror:
        raise RuntimeError(keyerror)

    rox.safe_close()


def _display_roxapi_grid_info(rox, roxgrid):  # pragma: no cover
    # in prep!
    """Push info to screen (mostly for debugging)"""
    cpgeom = False
    if rox.version_required("1.3"):
        cpgeom = True

    indexer = roxgrid.grid_indexer
    ncol, nrow, _nlay = indexer.dimensions

    if cpgeom:
        xtg.say("ROXAPI with support for CornerPointGeometry")
        geom = roxgrid.get_geometry()
        defined_cells = geom.get_defined_cells()
        xtg.say("Defined cells \n{}".format(defined_cells))

        xtg.say("IJK handedness: {}".format(geom.ijk_handedness))
        for ipi in range(ncol + 1):
            for jpi in range(nrow + 1):
                tpi, bpi, zco = geom.get_pillar_data(ipi, jpi)
                xtg.say("For pillar {}, {}\n".format(ipi, jpi))
                xtg.say("Tops\n{}".format(tpi))
                xtg.say("Bots\n{}".format(bpi))
                xtg.say("Depths\n{}".format(zco))


def _convert_to_xtgeo_grid(self, rox, roxgrid, corners, gname):  # pragma: no cover
    """Convert from RMS API to XTGeo API"""

    # pylint: disable=too-many-statements

    logger.info("Converting to XTGeo internals...")
    logger.info("Call the ROXAPI grid indexer")
    indexer = roxgrid.grid_indexer

    ncol, nrow, nlay = indexer.dimensions
    ntot = ncol * nrow * nlay

    # update other attributes
    self._ncol = ncol
    self._nrow = nrow
    self._nlay = nlay

    self.name = gname

    if corners is None:
        logger.info("Asked for dimensions_only: No geometry read!")
        return

    logger.info("Get active cells")
    mybuffer = np.ndarray(indexer.dimensions, dtype=np.int32)

    mybuffer.fill(0)

    logger.info("Get cell numbers")
    cellno = indexer.get_cell_numbers_in_range((0, 0, 0), indexer.dimensions)

    logger.info("Reorder...")
    ijk = indexer.get_indices(cellno)

    iind = ijk[:, 0]
    jind = ijk[:, 1]
    kind = ijk[:, 2]

    pvalues = np.ones(len(cellno))
    pvalues[cellno] = 1
    mybuffer[iind, jind, kind] = pvalues[cellno]

    actnum = mybuffer

    if rox.version_required("1.3"):
        logger.info("Handedness (new) %s", indexer.ijk_handedness)
    else:
        logger.info("Handedness (old) %s", indexer.handedness)

    corners = corners.ravel(order="K")
    actnum = actnum.ravel(order="K")

    ntot = ncol * nrow * nlay
    ncoord = (ncol + 1) * (nrow + 1) * 2 * 3
    nzcorn = ncol * nrow * (nlay + 1) * 4

    self._coordsv = np.zeros(ncoord, dtype=np.float64)
    self._zcornsv = np.zeros(nzcorn, dtype=np.float64)
    self._actnumsv = np.zeros(ntot, dtype=np.int32)

    # next task is to convert geometry to cxtgeo internal format
    logger.info("Run XTGeo C code...")
    _cxtgeo.grd3d_conv_roxapi_grid(
        ncol,
        nrow,
        nlay,
        ntot,
        actnum,
        corners,
        self._coordsv,
        self._zcornsv,
        self._actnumsv,
    )
    logger.info("Run XTGeo C code... done")
    logger.info("Converting to XTGeo internals... done")

    del corners
    del actnum

    # subgrids
    if len(indexer.zonation) > 1:
        logger.debug("Zonation length (N subzones) is %s", len(indexer.zonation))
        subz = OrderedDict()
        for inum, zrange in indexer.zonation.items():
            logger.debug("inum: %s, zrange: %s", inum, zrange)
            zname = roxgrid.zone_names[inum]
            logger.debug("zname is: %s", zname)
            zra = [nn + 1 for ira in zrange for nn in ira]  # nested lists
            subz[zname] = zra

        self.subgrids = subz


# =============================================================================
# Save/export
# =============================================================================


def export_grid_roxapi(
    self, projectname, gname, realisation, info=False, method="cpg"
):  # pragma: no cover
    """Export (i.e. store in RMS) via ROXAR API spec.

    Using method 'cpg' means that the CPG method is applied (from ROXAPI 1.3).
    This is possible from version ROXAPI ver 1.3, where the CornerPointGeometry
    class is defined.

    An alternative is to use simple roff import (via some /tmp area),
    can be used from version 1.2

    """
    rox = RoxUtils(projectname, readonly=False)

    if method != "cpg" and not rox.version_required("1.2"):
        minimumrms = rox.rmsversion("1.2")
        raise NotImplementedError(
            "Not supported in this ROXAPI version. Grid load/import requires "
            "RMS version {}".format(minimumrms)
        )

    if method == "cpg" and not rox.version_required("1.3"):
        xtg.warn(
            "Export method=cpg is not implemented for ROXAPI "
            "version {}. Change to workaround...".format(rox.roxversion)
        )
        method = "other"

    if method == "cpg":
        _export_grid_cornerpoint_roxapi(self, rox, gname, realisation, info)
    else:
        _export_grid_viaroff_roxapi(self, rox, gname, realisation)

    rox.safe_close()


def _export_grid_cornerpoint_roxapi(
    self, rox, gname, realisation, info
):  # pragma: no cover
    """Convert xtgeo geometry to pillar spec in ROXAPI and store"""

    try:
        from roxar.grids import CornerPointGridGeometry as CPG
    except ImportError:
        raise RuntimeError("Cannot load Roxar module")

    logger.info("Load grid via CornerPointGridGeometry...")

    grid_model = rox.project.grid_models.create(gname)
    grid_model.set_empty(realisation)
    grid = grid_model.get_grid(realisation)

    geom = CPG.create(self.dimensions)

    logger.info(geom)

    npill = (self.ncol + 1) * (self.nrow + 1) * 3
    nzcrn = (self.ncol + 1) * (self.nrow + 1) * 4 * (self.nlay + 1)

    _ier, tpi, bpi, zco = _cxtgeo.grd3d_conv_grid_roxapi(
        self.ncol,
        self.nrow,
        self.nlay,
        self._coordsv,
        self._zcornsv,
        self._actnumsv,
        npill,
        npill,
        nzcrn,
    )

    tpi = tpi.reshape(self.ncol + 1, self.nrow + 1, 3)
    bpi = bpi.reshape(self.ncol + 1, self.nrow + 1, 3)

    zco = np.ma.masked_greater(zco, xtgeo.UNDEF_LIMIT)
    zco = zco.reshape((self.ncol + 1, self.nrow + 1, 4, self.nlay + 1))

    for ipi in range(self.ncol + 1):
        for jpi in range(self.nrow + 1):
            zzco = zco[ipi, jpi].reshape((self.nlay + 1), 4).T
            geom.set_pillar_data(
                ipi,
                jpi,
                top_point=tpi[ipi, jpi],
                bottom_point=bpi[ipi, jpi],
                depth_values=zzco,
            )
            if info and ipi < 5 and jpi < 5:
                if ipi == 0 and jpi == 0:
                    xtg.say("Showing info for i<5 and j<5 only!")
                xtg.say("XTGeo pillar {}, {}\n".format(ipi, jpi))
                xtg.say("XTGeo Tops\n{}".format(tpi[ipi, jpi]))
                xtg.say("XTGeo Bots\n{}".format(bpi[ipi, jpi]))
                xtg.say("XTGeo Depths\n{}".format(zzco))

    geom.set_defined_cells(self.get_actnum().values.astype(np.bool))
    grid.set_geometry(geom)


def _export_grid_viaroff_roxapi(self, rox, gname, realisation):  # pragma: no cover
    """Convert xtgeo geometry to internal RMS via i/o ROFF tricks"""

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
