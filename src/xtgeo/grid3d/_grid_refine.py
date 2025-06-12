"""Private module for refinement of a grid."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import xtgeo._internal as _internal  # type: ignore
from xtgeo.common import XTGeoDialog, null_logger
from xtgeo.grid3d import _gridprop_op1

xtg = XTGeoDialog()
logger = null_logger(__name__)

if TYPE_CHECKING:
    from xtgeo.grid3d import Grid, GridProperty


def refine(
    self: Grid,
    refine_col: int | dict[int, int],
    refine_row: int | dict[int, int],
    refine_layer: int | dict[int, int],
    zoneprop: GridProperty | None = None,
) -> Grid:
    """Refine in all direction, proportionally.

    See details in caller.
    """
    self._set_xtgformat1()
    self.make_zconsistent()

    if isinstance(refine_col, int):
        refine_factor_column = [refine_col] * self.dimensions[0]
    else:
        refine_factor_column = [1] * self.dimensions[0]
        for col, factor in refine_col.items():
            if 0 < col < self.dimensions[0] and isinstance(factor, int) and factor > 0:
                refine_factor_column[col - 1] = factor
            else:
                xtg.warning("Invalid refine_col item {col}:{factor}")

    if isinstance(refine_row, int):
        refine_factor_row = [refine_row] * self.dimensions[1]
    else:
        refine_factor_row = [1] * self.dimensions[1]
        for row, factor in refine_row.items():
            if 0 < row < self.dimensions[1] and isinstance(factor, int) and factor > 0:
                refine_factor_row[row - 1] = factor
            else:
                xtg.warning("Invalid refine_row item {row}:{factor}")

    refine_factor_layer_dict = {}
    # case 1 rfactor as scalar value.
    if isinstance(refine_layer, int):
        if self.subgrids:
            subgrids = self.get_subgrids()
            for i, _ in enumerate(self.subgrids.keys()):
                refine_factor_layer_dict[i + 1] = refine_layer
        else:
            refine_factor_layer_dict[0] = refine_layer
            subgrids = {}
            subgrids[1] = self.nlay

    # case 2 rfactor is a dict
    else:
        refine_factor_layer_dict = dict(
            sorted(refine_layer.items())
        )  # redefined to ordered
        # 2a: zoneprop is present
        if zoneprop is not None:
            oldsubgrids = None
            if self.subgrids:
                oldsubgrids = self.get_subgrids()

            subgrids = self.subgrids_from_zoneprop(zoneprop)

            if oldsubgrids and subgrids.values() != oldsubgrids.values():
                xtg.warn(
                    "Subgrid definitions from zone property do not match existing "
                    "subgrids. Proceeding with new subgrid definitions from zone "
                    "property."
                )

        # 2b: zoneprop is not present
        elif zoneprop is None and self.subgrids:
            subgrids = self.get_subgrids()

        elif zoneprop is None and not self.subgrids:
            raise ValueError(
                "You gave in a dict, but no zoneprops and "
                "subgrids are not present in the grid"
            )
        else:
            raise ValueError("Some major unexpected issue in routine...")

    if len(subgrids) != len(refine_factor_layer_dict):
        raise RuntimeError("Subgrids and refinements: different definition!")

    self.set_subgrids(subgrids)

    # Now, based on dict, give a value per subgrid for key, val in rfactor
    newsubgrids = {}
    newnlay = 0
    for (_x, rfi), (snam, sran) in zip(
        refine_factor_layer_dict.items(), subgrids.items()
    ):
        newsubgrids[snam] = sran * rfi
        newnlay += newsubgrids[snam]

    logger.debug("New layers: %s", newnlay)

    refine_factor_layer = []

    for (_, rfi), (_, arr) in zip(
        refine_factor_layer_dict.items(), self.subgrids.items()
    ):
        for _ in range(len(arr)):
            refine_factor_layer.append(rfi)

    self._set_xtgformat2()

    refine_factor_column = np.array(refine_factor_column, dtype=np.int8)
    refine_factor_row = np.array(refine_factor_row, dtype=np.int8)
    refine_factor_layer = np.array(refine_factor_layer, dtype=np.int8)

    if refine_factor_column.sum() > self.dimensions[0]:
        grid_cpp = _internal.grid3d.Grid(self)
        ref_coordsv, ref_zcornsv, ref_actnumsv = grid_cpp.refine_columns(
            refine_factor_column
        )
        self._coordsv = ref_coordsv
        self._zcornsv = ref_zcornsv
        self._actnumsv = ref_actnumsv.astype(np.int32)
        self._ncol = int(refine_factor_column.sum())

    if refine_factor_row.sum() > self.dimensions[1]:
        grid_cpp = _internal.grid3d.Grid(self)
        ref_coordsv, ref_zcornsv, ref_actnumsv = grid_cpp.refine_rows(refine_factor_row)
        self._coordsv = ref_coordsv
        self._zcornsv = ref_zcornsv
        self._actnumsv = ref_actnumsv.astype(np.int32)
        self._nrow = int(refine_factor_row.sum())

    if refine_factor_layer.sum() > self.dimensions[2]:
        grid_cpp = _internal.grid3d.Grid(self)
        ref_zcornsv, ref_actnumsv = grid_cpp.refine_vertically(refine_factor_layer)
        self._zcornsv = ref_zcornsv
        self._actnumsv = ref_actnumsv.astype(np.int32)
        self._nlay = newnlay
        if self.subgrids is None or len(self.subgrids) <= 1:
            self.subgrids = None
        else:
            self.set_subgrids(newsubgrids)

    # Check if grid has any properties
    if self._props and self._props.props and len(self._props.props) > 0:
        for prop in self._props.props:
            _gridprop_op1.refine(
                prop, refine_factor_column, refine_factor_row, refine_factor_layer
            )

    return self


def refine_vertically(
    self: Grid,
    rfactor: int | dict[int, int],
    zoneprop: GridProperty | None = None,
) -> Grid:
    """Refine vertically, proportionally.

    See details in caller.
    """
    self._set_xtgformat1()
    self.make_zconsistent()

    rfactord = {}

    # case 1 rfactor as scalar value.
    if isinstance(rfactor, int):
        if self.subgrids:
            subgrids = self.get_subgrids()
            for i, _ in enumerate(self.subgrids.keys()):
                rfactord[i + 1] = rfactor
        else:
            rfactord[0] = rfactor
            subgrids = {}
            subgrids[1] = self.nlay

    # case 2 rfactor is a dict
    else:
        rfactord = dict(sorted(rfactor.items()))  # redefined to ordered
        # 2a: zoneprop is present
        if zoneprop is not None:
            oldsubgrids = None
            if self.subgrids:
                oldsubgrids = self.get_subgrids()

            subgrids = self.subgrids_from_zoneprop(zoneprop)

            if oldsubgrids and subgrids.values() != oldsubgrids.values():
                xtg.warn("ISSUES!!!")

        # 2b: zoneprop is not present
        elif zoneprop is None and self.subgrids:
            subgrids = self.get_subgrids()

        elif zoneprop is None and not self.subgrids:
            raise ValueError(
                "You gave in a dict, but no zoneprops and "
                "subgrids are not present in the grid"
            )
        else:
            raise ValueError("Some major unexpected issue in routine...")

    if len(subgrids) != len(rfactord):
        raise RuntimeError("Subgrids and refinements: different definition!")

    self.set_subgrids(subgrids)

    # Now, based on dict, give a value per subgrid for key, val in rfactor
    newsubgrids = {}
    newnlay = 0
    for (_x, rfi), (snam, sran) in zip(rfactord.items(), subgrids.items()):
        newsubgrids[snam] = sran * rfi
        newnlay += newsubgrids[snam]

    logger.debug("New layers: %s", newnlay)

    refine_factors = []

    for (_, rfi), (_, arr) in zip(rfactord.items(), self.subgrids.items()):
        for _ in range(len(arr)):
            refine_factors.append(rfi)

    self._set_xtgformat2()
    grid_cpp = _internal.grid3d.Grid(self)
    refine_factors = np.array(refine_factors, dtype=np.int8)
    ref_zcornsv, ref_actnumsv = grid_cpp.refine_vertically(refine_factors)

    # update instance:
    self._nlay = newnlay
    self._zcornsv = ref_zcornsv
    self._actnumsv = ref_actnumsv

    if self.subgrids is None or len(self.subgrids) <= 1:
        self.subgrids = None
    else:
        self.set_subgrids(newsubgrids)

    # Check if grid has any properties
    if self._props and self._props.props and len(self._props.props) > 0:
        for prop in self._props.props:
            _gridprop_op1.refine(prop, 1, 1, refine_factors)

    return self
