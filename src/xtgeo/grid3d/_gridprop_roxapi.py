# coding: utf-8
"""Roxar API functions for XTGeo Grid Property."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy import ma

from xtgeo.common import null_logger
from xtgeo.common.constants import UNDEF, UNDEF_INT, UNDEF_INT_LIMIT, UNDEF_LIMIT
from xtgeo.roxutils import RoxUtils

try:
    import roxar
except ImportError:
    pass

if TYPE_CHECKING:
    from xtgeo.grid3d.grid_property import GridProperty

logger = null_logger(__name__)

# self is the XTGeo GridProperty instance

VALID_ROXAR_DTYPES = [np.uint8, np.uint16, np.float32]


def import_prop_roxapi(
    project: roxar.Project, gname: str, pname: str, realisation: int, faciescodes: bool
) -> dict[str, Any]:
    """Import a Property via ROXAR API spec."""
    logger.info("Opening RMS project ...")

    rox = RoxUtils(project, readonly=True)

    result = _get_gridprop_data(rox, gname, pname, realisation, faciescodes)

    rox.safe_close()
    return result


def _get_gridprop_data(
    rox: RoxUtils, gname: str, pname: str, realisation: int, faciescodes: bool
) -> dict[str, Any]:
    # inside a RMS project

    if gname not in rox.project.grid_models:
        raise ValueError(f"No gridmodel with name {gname}")
    if pname not in rox.project.grid_models[gname].properties:
        raise ValueError(f"No property in {gname} with name {pname}")

    try:
        return _convert_to_xtgeo_prop(rox, gname, pname, realisation, faciescodes)
    except KeyError as keyerror:
        raise RuntimeError(keyerror) from keyerror


def _convert_to_xtgeo_prop(
    rox: RoxUtils, gname: str, pname: str, realisation: int, faciescodes: bool
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    roxgrid = rox.project.grid_models[gname]
    roxprop = roxgrid.properties[pname]

    discrete = str(roxprop.type) in ("discrete", "body_facies")
    result["discrete"] = discrete

    result["roxorigin"] = True
    indexer = roxgrid.get_grid(realisation=realisation).grid_indexer
    result["ncol"], result["nrow"], result["nlay"] = indexer.dimensions

    logger.info(indexer.ijk_handedness)

    pvalues = roxprop.get_values(realisation=realisation)

    if str(roxprop.type) == "body_facies" and faciescodes:
        fmap = roxprop.get_facies_map(realisation=realisation)
        pvalues = fmap[pvalues]  # numpy magics

    result["roxar_dtype"] = pvalues.dtype

    mybuffer: np.ndarray = np.ndarray(
        indexer.dimensions, dtype=np.int32 if discrete else np.float64
    )
    mybuffer.fill(UNDEF_INT if discrete else UNDEF)

    cellno = indexer.get_cell_numbers_in_range((0, 0, 0), indexer.dimensions)

    ijk = indexer.get_indices(cellno)

    iind = ijk[:, 0]
    jind = ijk[:, 1]
    kind = ijk[:, 2]

    mybuffer[iind, jind, kind] = pvalues[cellno]

    mybuffer = ma.masked_greater(mybuffer, UNDEF_INT_LIMIT if discrete else UNDEF_LIMIT)

    result["values"] = mybuffer
    result["name"] = pname

    if discrete:
        result["codes"] = _fix_codes(
            result["values"].reshape(-1).compressed(), roxprop.code_names
        )
        logger.info("Fixed codes: %s", result["codes"])
    return result


def export_prop_roxapi(
    self: GridProperty,
    project: roxar.Project,
    gname: str,
    pname: str,
    realisation: int = 0,
    casting: Literal["no", "equiv", "safe", "same_kind", "unsafe"] | None = "unsafe",
) -> None:
    """Export (i.e. store or save) to a Property icon in RMS via ROXAR API spec."""
    rox = RoxUtils(project, readonly=False)

    try:
        roxgrid = rox.project.grid_models[gname]
        _store_in_roxar(self, pname, roxgrid, realisation, casting)

    except KeyError as keyerror:
        raise RuntimeError(keyerror)

    if rox._roxexternal:
        rox.project.save()

    rox.safe_close()


def _store_in_roxar(
    self: GridProperty,
    pname: str,
    roxgrid: roxar.grid.Grid3D,
    realisation: int,
    casting: Literal["no", "equiv", "safe", "same_kind", "unsafe"] | None,
) -> None:
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

    pvalues = roxgrid.get_grid(realisation=realisation).generate_values(data_type=dtype)

    roxar_property_type = (
        roxar.GridPropertyType.discrete
        if self.isdiscrete
        else roxar.GridPropertyType.continuous
    )

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
        rprop.code_names = _rox_compatible_codes(self.codes)


def _fix_codes(
    active_values: np.ndarray, codes: dict[str | int, str]
) -> dict[int, str]:
    """Roxar may provide a code list with empty strings values, fix this issue here.

    Roxar may also interpolate code values which are actually not present in the
    property. Here, the presence of actual codes is also checked.
    """
    newcodes = {}
    codes_data = {val: str(val) for val in np.unique(active_values)}

    for code, name in codes.items():
        if not isinstance(code, int):
            code = int(code)

        if not name:
            name = str(code)

        if code not in codes_data.keys():
            continue

        newcodes[code] = name

    return newcodes


def _rox_compatible_codes(codes: dict) -> dict:
    """Ensure that keys in codes are int's prior to storage in RMS."""

    newcodes = {}
    for code, name in codes.items():
        if code is None:
            continue  # skip codes of type None; assumed to be spurious
        if not isinstance(code, int):
            try:
                code = int(code)
            except ValueError:
                raise ValueError(
                    "The keys in codes must be an integer prior to RMS "
                    f"storage. Actual key found here is '{code}' of type {type(code)}"
                )

        newcodes[code] = name
    return newcodes
