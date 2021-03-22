# coding: utf-8
"""Roxar API functions for XTGeo Grid Property."""

import numpy as np
import numpy.ma as ma

import xtgeo
from xtgeo.common import XTGeoDialog

try:
    import roxar  # type: ignore
except ImportError:
    pass

xtg = XTGeoDialog()

logger = xtg.functionlogger(__name__)

# self is the XTGeo GridProperty instance
# pragma: no cover

VALID_ROXAR_DTYPES = [np.uint8, np.uint16, np.float32]


def import_prop_roxapi(
    self, project, gname, pname, realisation, faciescodes
):  # pragma: no cover
    """Import a Property via ROXAR API spec."""
    logger.info("Opening RMS project ...")

    rox = xtgeo.RoxUtils(project, readonly=True)

    _get_gridprop_data(self, rox, gname, pname, realisation, faciescodes)

    rox.safe_close()


def _get_gridprop_data(
    self, rox, gname, pname, realisation, faciescodes
):  # pragma: no cover
    # inside a RMS project

    if gname not in rox.project.grid_models:
        raise ValueError("No gridmodel with name {}".format(gname))
    if pname not in rox.project.grid_models[gname].properties:
        raise ValueError("No property in {} with name {}".format(gname, pname))

    try:
        roxgrid = rox.project.grid_models[gname]
        roxprop = roxgrid.properties[pname]

        if str(roxprop.type) in ("discrete", "body_facies"):
            self._isdiscrete = True

        self._roxorigin = True
        _convert_to_xtgeo_prop(
            self, rox, pname, roxgrid, roxprop, realisation, faciescodes
        )

    except KeyError as keyerror:
        raise RuntimeError(keyerror)


def _convert_to_xtgeo_prop(
    self, rox, pname, roxgrid, roxprop, realisation, faciescodes
):  # pragma: no cover
    """Collect numpy array and convert to XTGeo internal format."""
    indexer = roxgrid.get_grid(realisation=realisation).grid_indexer
    self._ncol, self._nrow, self._nlay = indexer.dimensions

    if rox.version_required("1.3"):
        logger.info(indexer.ijk_handedness)
    else:
        logger.info(indexer.handedness)

    pvalues = roxprop.get_values(realisation=realisation)

    if str(roxprop.type) == "body_facies" and faciescodes:
        fmap = roxprop.get_facies_map(realisation=realisation)
        pvalues = fmap[pvalues]  # numpy magics

    self._roxar_dtype = pvalues.dtype

    if self._isdiscrete:
        mybuffer = np.ndarray(indexer.dimensions, dtype=np.int32)
        mybuffer.fill(xtgeo.UNDEF_INT)
    else:
        mybuffer = np.ndarray(indexer.dimensions, dtype=np.float64)
        mybuffer.fill(xtgeo.UNDEF)

    cellno = indexer.get_cell_numbers_in_range((0, 0, 0), indexer.dimensions)

    ijk = indexer.get_indices(cellno)

    iind = ijk[:, 0]
    jind = ijk[:, 1]
    kind = ijk[:, 2]

    mybuffer[iind, jind, kind] = pvalues[cellno]

    if self._isdiscrete:
        mybuffer = ma.masked_greater(mybuffer, xtgeo.UNDEF_INT_LIMIT)
    else:
        mybuffer = ma.masked_greater(mybuffer, xtgeo.UNDEF_LIMIT)

    self._values = mybuffer
    self._name = pname

    if self._isdiscrete:
        self.codes = _fix_codes(self, roxprop.code_names)
        logger.info("Fixed codes: %s", self.codes)


def export_prop_roxapi(
    self, project, gname, pname, realisation=0, casting="unsafe"
):  # pragma: no cover
    """Export (i.e. store or save) to a Property icon in RMS via ROXAR API spec."""
    rox = xtgeo.RoxUtils(project, readonly=False)

    try:
        roxgrid = rox.project.grid_models[gname]
        _store_in_roxar(self, pname, roxgrid, realisation, casting)

    except KeyError as keyerror:
        raise RuntimeError(keyerror)

    if rox._roxexternal:
        rox.project.save()

    rox.safe_close()


def _store_in_roxar(self, pname, roxgrid, realisation, casting):  # pragma: no cover
    """Store property in RMS."""
    indexer = roxgrid.get_grid(realisation=realisation).grid_indexer

    logger.info("Store in RMS...")

    val3d = self.values.copy()

    cellno = indexer.get_cell_numbers_in_range((0, 0, 0), indexer.dimensions)

    ijk = indexer.get_indices(cellno)

    iind = ijk[:, 0]
    jind = ijk[:, 1]
    kind = ijk[:, 2]

    dtype = self._roxar_dtype
    logger.info("DTYPE is %s for %s", dtype, pname)

    # casting will secure correct types
    if dtype not in VALID_ROXAR_DTYPES:
        raise TypeError(
            f"Roxar dtype is not valid: {dtype} must be in {VALID_ROXAR_DTYPES}"
        )

    if self.isdiscrete:
        pvalues = roxgrid.get_grid(realisation=realisation).generate_values(
            data_type=dtype
        )
        roxar_property_type = roxar.GridPropertyType.discrete

    else:
        pvalues = roxgrid.get_grid(realisation=realisation).generate_values(
            data_type=dtype
        )
        roxar_property_type = roxar.GridPropertyType.continuous

    pvalues[cellno] = val3d[iind, jind, kind]

    properties = roxgrid.properties

    if pname not in properties:
        rprop = properties.create(
            pname, property_type=roxar_property_type, data_type=dtype
        )
    else:
        rprop = properties[pname]
        dtype = rprop.data_type

    rprop.set_values(pvalues.astype(dtype, casting=casting), realisation=realisation)

    if self.isdiscrete:
        rprop.code_names = self.codes.copy()


def _fix_codes(self, codes):  # pragma: no cover
    """Roxar may provide a code list with empty strings values, fix this issue here.

    Roxar may also interpolate code values which are actually not present in the
    property. Here, the presence of actual codes is also checked.
    """
    newcodes = {}
    codes_data = {val: str(val) for val in np.unique(self.get_active_npvalues1d())}

    for code, name in codes.items():
        if not isinstance(code, int):
            code = int(code)

        if not name:
            name = str(code)

        if code not in codes_data.keys():
            continue

        newcodes[code] = name

    return newcodes
