"""Private module for refinement of a grid."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

import xtgeo._internal as _internal  # type: ignore
from xtgeo.common import XTGeoDialog, null_logger
from xtgeo.grid3d import _gridprop_op1

xtg = XTGeoDialog()
logger = null_logger(__name__)

if TYPE_CHECKING:
    from xtgeo.grid3d import Grid, GridProperty


def _is_non_text_sequence(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    )


def _expand_section_factors(
    ncells: int, factors: Sequence[int], name: str
) -> list[int]:
    """Expand a per-section list of refinement factors to a per-cell list.

    The cells along the chosen direction are split into ``len(factors)``
    approximately equal sections, mirroring :func:`numpy.array_split` semantics:
    when ``ncells`` is not divisible by the number of sections, the earlier
    sections receive one extra cell each. Each section is then assigned the
    refinement factor at the matching position of ``factors``.

    Example:
        ``_expand_section_factors(10, [2, 3, 1], "refine_col")`` produces
        ``[2, 2, 2, 2, 3, 3, 3, 1, 1, 1]`` (sections of size 4, 3, 3).

    Args:
        ncells: Number of cells in the direction being refined.
        factors: Per-section refinement factors. Must contain at least one
            entry, contain only positive integers and have no more entries
            than ``ncells``.
        name: Argument name used in error messages.

    Returns:
        A list of length ``ncells`` holding the per-cell refinement factor.

    Raises:
        ValueError: If ``factors`` is empty, contains non-positive values or
            has more entries than ``ncells``.
        TypeError: If any factor is not an ``int``.
    """
    n_sections = len(factors)
    if n_sections == 0:
        raise ValueError(f"{name} list must contain at least one factor")
    if n_sections > ncells:
        raise ValueError(
            f"{name} list has {n_sections} sections, but the grid only has "
            f"{ncells} cells in that direction"
        )
    for idx, factor in enumerate(factors):
        # bool is a subclass of int; reject it explicitly to avoid surprises
        if not isinstance(factor, int) or isinstance(factor, bool):
            raise TypeError(
                f"{name}[{idx}]={factor!r} must be int, got {type(factor).__name__}"
            )
        if not 1 <= factor <= np.iinfo(np.uint16).max:
            raise ValueError(
                f"{name}[{idx}]={factor} must be in range "
                f"[1, {np.iinfo(np.uint16).max}] "
                f"(use 1 to leave a section unchanged)"
            )

    base, extra = divmod(ncells, n_sections)
    expanded: list[int] = []
    for k, factor in enumerate(factors):
        size = base + (1 if k < extra else 0)
        expanded.extend([factor] * size)
    return expanded


def _validate_refine_dict(name: str, factor: dict[int, int], max_refine: int) -> None:
    for key, value in factor.items():
        if isinstance(key, bool) or not isinstance(key, int):
            raise TypeError(
                f"{name}[{key!r}] key must be int, got {type(key).__name__}"
            )
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(
                f"{name}[{key}]={value!r} must be int, got {type(value).__name__}"
            )
        if not 1 <= value <= max_refine:
            raise ValueError(
                f"{name}[{key}]={value} is out of valid range [1, {max_refine}]"
            )


def _validate_refine_factor(
    name: str, factor: int | Sequence[int] | dict[int, int], max_refine: int
) -> None:
    if isinstance(factor, bool):
        expected = (
            "int or dict[int, int]"
            if name == "refine_layer"
            else "int, sequence of int or dict[int, int]"
        )
        raise TypeError(f"{name} must be {expected}, got {type(factor).__name__}")
    if isinstance(factor, int):
        if not 1 <= factor <= max_refine:
            raise ValueError(f"{name}={factor} is out of valid range [1, {max_refine}]")
        return
    if _is_non_text_sequence(factor):
        if name == "refine_layer":
            raise TypeError(
                f"{name} must be int or dict[int, int], got {type(factor).__name__}"
            )
        return
    if isinstance(factor, dict):
        _validate_refine_dict(name, factor, max_refine)
        return
    if name == "refine_layer":
        raise TypeError(
            f"{name} must be int or dict[int, int], got {type(factor).__name__}"
        )
    raise TypeError(
        f"{name} must be int, sequence of int or dict[int, int], got "
        f"{type(factor).__name__}"
    )


def _build_lateral_refine_factors(
    name: str, factor: int | Sequence[int] | dict[int, int], ncells: int
) -> list[int]:
    if isinstance(factor, int):
        return [factor] * ncells
    if _is_non_text_sequence(factor):
        return _expand_section_factors(ncells, factor, name)

    refine_factors = [1] * ncells
    for index, item_factor in factor.items():
        if not 0 < index <= ncells:
            raise ValueError(f"{name} key {index} is out of valid range [1, {ncells}]")
        refine_factors[index - 1] = item_factor
    return refine_factors


def refine(
    self: Grid,
    refine_col: int | Sequence[int] | dict[int, int],
    refine_row: int | Sequence[int] | dict[int, int],
    refine_layer: int | dict[int, int],
    zoneprop: GridProperty | None = None,
) -> Grid:
    """Refine in all direction, proportionally.

    See details in caller.
    """
    self._set_xtgformat1()
    self.make_zconsistent()

    max_refine = np.iinfo(np.uint16).max

    # Validate refinement factors are within valid range
    for name, factor in (
        ("refine_col", refine_col),
        ("refine_row", refine_row),
        ("refine_layer", refine_layer),
    ):
        _validate_refine_factor(name, factor, max_refine)

    refine_factor_column = _build_lateral_refine_factors(
        "refine_col", refine_col, self.dimensions[0]
    )
    refine_factor_row = _build_lateral_refine_factors(
        "refine_row", refine_row, self.dimensions[1]
    )

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

    refine_factor_column = np.array(refine_factor_column, dtype=np.uint16)
    refine_factor_row = np.array(refine_factor_row, dtype=np.uint16)
    refine_factor_layer = np.array(refine_factor_layer, dtype=np.uint16)

    # Copy properties BEFORE refining the grid, while dimensions still match
    properties_to_refine = []
    if self._props and self._props.props and len(self._props.props) > 0:
        for prop in self._props.props:
            properties_to_refine.append(prop.copy())

    # Now refine the grid
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

    # Refine the copied properties and update the grid with them
    if properties_to_refine:
        refined_props = []
        for newprop in properties_to_refine:
            newprop.geometry = None
            _gridprop_op1.refine(
                newprop, refine_factor_column, refine_factor_row, refine_factor_layer
            )
            newprop.geometry = self
            refined_props.append(newprop)
        self._props.props = refined_props

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

    max_refine = np.iinfo(np.uint16).max

    # Validate refinement factor is within valid range
    if isinstance(rfactor, int):
        if isinstance(rfactor, bool):
            raise TypeError(
                f"rfactor must be int or dict[int, int], got {type(rfactor).__name__}"
            )
        if not 1 <= rfactor <= max_refine:
            raise ValueError(
                f"rfactor={rfactor} is out of valid range [1, {max_refine}]"
            )
    elif isinstance(rfactor, dict):
        for key, value in rfactor.items():
            if isinstance(key, bool) or not isinstance(key, int):
                raise TypeError(
                    f"rfactor[{key!r}] key must be int, got {type(key).__name__}"
                )
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    f"rfactor[{key}]={value!r} must be int, got {type(value).__name__}"
                )
            if not 1 <= value <= max_refine:
                raise ValueError(
                    f"rfactor[{key}]={value} is out of valid range [1, {max_refine}]"
                )
    else:
        raise TypeError(
            f"rfactor must be int or dict[int, int], got {type(rfactor).__name__}"
        )

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

    # Copy properties BEFORE refining the grid, while dimensions still match
    properties_to_refine = []
    if self._props and self._props.props and len(self._props.props) > 0:
        for prop in self._props.props:
            properties_to_refine.append(prop.copy())

    grid_cpp = _internal.grid3d.Grid(self)
    refine_factors = np.array(refine_factors, dtype=np.uint16)
    ref_zcornsv, ref_actnumsv = grid_cpp.refine_vertically(refine_factors)

    # update instance:
    self._nlay = newnlay
    self._zcornsv = ref_zcornsv
    self._actnumsv = ref_actnumsv.astype(np.int32)

    if self.subgrids is None or len(self.subgrids) <= 1:
        self.subgrids = None
    else:
        self.set_subgrids(newsubgrids)

    # Refine the copied properties and update the grid with them
    if properties_to_refine:
        refined_props = []
        for newprop in properties_to_refine:
            newprop.geometry = None
            _gridprop_op1.refine(newprop, 1, 1, refine_factors)
            newprop.geometry = self
            refined_props.append(newprop)
        self._props.props = refined_props

    return self
