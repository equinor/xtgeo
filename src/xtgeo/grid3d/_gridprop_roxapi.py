# coding: utf-8
"""Roxar API functions for XTGeo Grid Property"""

import numpy as np
import numpy.ma as ma

import xtgeo
from xtgeo.common import XTGeoDialog

try:
    import roxar
except ImportError:
    pass

xtg = XTGeoDialog()

logger = xtg.functionlogger(__name__)

# self is the XTGeo GridProperty instance


def import_prop_roxapi(self, project, gname, pname, realisation):
    """Import a Property via ROXAR API spec."""

    logger.info("Opening RMS project ...")
    rox = xtgeo.RoxUtils(project, readonly=True)

    _get_gridprop_data(self, rox, gname, pname, realisation)

    rox.safe_close()


def _get_gridprop_data(self, rox, gname, pname, realisation):
    # inside a RMS project

    logger.info("Realisation key not applied yet: %s", realisation)

    if gname not in rox.project.grid_models:
        raise ValueError("No gridmodel with name {}".format(gname))
    if pname not in rox.project.grid_models[gname].properties:
        raise ValueError("No property in {} with name {}".format(gname, pname))

    try:
        roxgrid = rox.project.grid_models[gname]
        roxprop = roxgrid.properties[pname]

        if str(roxprop.type) == "discrete":
            self._isdiscrete = True

        self._roxorigin = True
        _convert_to_xtgeo_prop(self, rox, pname, roxgrid, roxprop)

    except KeyError as keyerror:
        raise RuntimeError(keyerror)


def _convert_to_xtgeo_prop(self, rox, pname, roxgrid, roxprop):

    indexer = roxgrid.get_grid().grid_indexer
    self._ncol, self._nrow, self._nlay = indexer.dimensions

    if rox.version_required("1.3"):
        logger.info(indexer.ijk_handedness)
    else:
        logger.info(indexer.handedness)

    pvalues = roxprop.get_values()
    self._roxar_dtype = pvalues.dtype

    if self._isdiscrete:
        mybuffer = np.ndarray(indexer.dimensions, dtype=np.int32)
    else:
        mybuffer = np.ndarray(indexer.dimensions, dtype=np.float64)

    mybuffer.fill(self.undef)  # self.undef dynamic based on self._isdiscrete

    cellno = indexer.get_cell_numbers_in_range((0, 0, 0), indexer.dimensions)

    ijk = indexer.get_indices(cellno)

    iind = ijk[:, 0]
    jind = ijk[:, 1]
    kind = ijk[:, 2]

    mybuffer[iind, jind, kind] = pvalues[cellno]

    mybuffer = ma.masked_greater(mybuffer, self.undef_limit)

    self._values = mybuffer
    self._name = pname


def export_prop_roxapi(self, project, gname, pname, saveproject=False, realisation=0):
    """Export (i.e. store) to a Property in RMS via ROXAR API spec."""

    rox = xtgeo.RoxUtils(project, readonly=False)

    logger.info("Realisation key not applied yet: %s", realisation)

    try:
        roxgrid = rox.project.grid_models[gname]
        _store_in_roxar(self, pname, roxgrid)

        if saveproject:
            try:
                rox.project.save()
            except RuntimeError:
                xtg.warn("Could not save project!")

    except KeyError as keyerror:
        raise RuntimeError(keyerror)

    rox.safe_close()


def _store_in_roxar(self, pname, roxgrid):

    indexer = roxgrid.get_grid().grid_indexer

    logger.info("Store in RMS...")

    val3d = self.values.copy()

    cellno = indexer.get_cell_numbers_in_range((0, 0, 0), indexer.dimensions)

    ijk = indexer.get_indices(cellno)

    iind = ijk[:, 0]
    jind = ijk[:, 1]
    kind = ijk[:, 2]

    dtype = self._roxar_dtype
    logger.info("DTYPE is %s for %s", dtype, pname)
    if self.isdiscrete:
        pvalues = roxgrid.get_grid().generate_values(data_type=dtype)
    else:
        pvalues = roxgrid.get_grid().generate_values(data_type=dtype)

    pvalues[cellno] = val3d[iind, jind, kind]

    properties = roxgrid.properties

    if self.isdiscrete:
        rprop = properties.create(
            pname, property_type=roxar.GridPropertyType.discrete, data_type=dtype
        )
    else:
        rprop = properties.create(
            pname, property_type=roxar.GridPropertyType.continuous, data_type=dtype
        )

    rprop.set_values(pvalues.astype(dtype))

    if self.isdiscrete:
        rprop.code_names = self.codes.copy()
