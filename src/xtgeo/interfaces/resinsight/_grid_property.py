"""ResInsight API helpers for grid properties via rips."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from xtgeo.common.constants import UNDEF, UNDEF_INT
from xtgeo.common.log import null_logger

from ._resinsight_base import _BaseResInsightDataRW
from ._rips_package import PropertyDataType, PropertyType, rips

if TYPE_CHECKING:
    import numpy.typing as npt

    from xtgeo.grid3d.grid import Grid
    from xtgeo.grid3d.grid_property import GridProperty

    from ._rips_package import ResInsightInstanceOrPortType

logger = null_logger(__name__)


def _validate_property_type(property_type: str | PropertyType) -> PropertyType:
    """Coerce *property_type* to ``rips.PropertyType``.

    Raises ``RuntimeError`` if rips is not installed, ``ValueError`` on invalid input.
    """
    if rips is None:
        raise RuntimeError("rips package is not available")
    if isinstance(property_type, PropertyType):
        return property_type
    try:
        return PropertyType(property_type)
    except ValueError:
        valid = ", ".join(f'"{m.value}"' for m in PropertyType)
        raise ValueError(
            f"Invalid property_type {property_type!r}. Must be one of: {valid}"
        ) from None


@dataclass(frozen=True, eq=False)
class GridPropertyDataResInsight:
    """Immutable data container for a ResInsight grid property.

    Stores property values for all cells (nx x ny x nz), together with the
    grid dimensions and actnum needed to reconstruct a full 3-D XTGeo
    ``GridProperty``.
    """

    name: str
    nx: int
    ny: int
    nz: int
    values: npt.NDArray[np.float64] | npt.NDArray[np.int32] = field(repr=False)
    actnumsv: npt.NDArray[np.int32] = field(repr=False)
    property_type: str | PropertyType
    time_step_index: int
    discrete: bool
    codes: dict[int, str]
    filesrc: str

    __hash__ = None  # type: ignore[assignment]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GridPropertyDataResInsight):
            return NotImplemented
        return (
            self.name == other.name
            and self.nx == other.nx
            and self.ny == other.ny
            and self.nz == other.nz
            and np.array_equal(self.values, other.values)
            and np.array_equal(self.actnumsv, other.actnumsv)
            and self.property_type == other.property_type
            and self.time_step_index == other.time_step_index
            and self.discrete == other.discrete
            and self.codes == other.codes
            and self.filesrc == other.filesrc
        )

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "property_type", _validate_property_type(self.property_type)
        )

        total_cells = self.nx * self.ny * self.nz
        if self.actnumsv.size != total_cells:
            raise ValueError(
                f"actnumsv should have length {total_cells}, but got "
                f"length {self.actnumsv.size}"
            )
        if self.values.size != total_cells:
            raise ValueError(
                f"values should have length {total_cells}, but got "
                f"length {self.values.size}"
            )

    def to_xtgeo_gridproperty(self) -> GridProperty:
        """Convert this data container to an XTGeo ``GridProperty``."""
        from xtgeo.grid3d.grid_property import GridProperty

        shape = (self.nx, self.ny, self.nz)
        dtype = np.int32 if self.discrete else np.float64
        vals = np.asarray(self.values, dtype=dtype).reshape(shape, order="F")
        mask = (self.actnumsv == 0).reshape(shape, order="F")

        return GridProperty(
            ncol=self.nx,
            nrow=self.ny,
            nlay=self.nz,
            values=np.ma.MaskedArray(vals, mask=mask),
            name=self.name,
            discrete=self.discrete,
            codes=dict(self.codes) if self.codes else None,
            filesrc=self.filesrc,
        )

    @classmethod
    def from_xtgeo_gridproperty(
        cls,
        prop: GridProperty,
        property_type: str | PropertyType,
        time_step_index: int = 0,
        grid: Grid | None = None,
    ) -> GridPropertyDataResInsight:
        """Create from an XTGeo ``GridProperty``."""
        validated_type = _validate_property_type(property_type)

        # Active = unmasked in property, intersected with grid actnum if given.
        # Ravel in Fortran order (i fastest) to match Eclipse/ResInsight convention.
        active = ~np.ma.getmaskarray(prop.values).ravel(order="F")
        if grid is not None:
            grid_actnum = np.asarray(
                grid.get_actnum().values.data, dtype=np.int32
            ).ravel(order="F")
            active &= grid_actnum != 0

        dtype = np.int32 if prop.isdiscrete else np.float64
        fill = UNDEF_INT if prop.isdiscrete else UNDEF

        values = np.asarray(prop.values.filled(fill), dtype=dtype).ravel(order="F")
        values = values.copy()  # type: ignore[assignment]
        values[~active] = fill

        return cls(
            name=prop.name or "unknown",
            nx=prop.ncol,
            ny=prop.nrow,
            nz=prop.nlay,
            values=values,  # type: ignore[arg-type]
            actnumsv=active.astype(np.int32),
            property_type=validated_type,
            time_step_index=time_step_index,
            discrete=prop.isdiscrete,
            codes=dict(prop.codes) if prop.isdiscrete and prop.codes else {},
            filesrc=prop.filesrc or "",
        )


def _read_actnum(case: object, expected_size: int) -> npt.NDArray[np.int32]:
    """Read ACTNUM from ResInsight as a grid property.

    Tries STATIC_NATIVE first, then INPUT_PROPERTY as fallback.
    Active cells have value 1, inactive cells have value 0.  Non-finite
    values (inf at inactive cells) are treated as inactive.
    """
    for ptype in (PropertyType.STATIC_NATIVE, PropertyType.INPUT_PROPERTY):
        try:
            raw = np.asarray(
                case.grid_property(ptype, "ACTNUM", 0),  # type: ignore[attr-defined]
                dtype=np.float64,
            )
            if raw.size == expected_size:
                return (np.isfinite(raw) & (raw != 0)).astype(np.int32)
        except Exception:  # noqa: BLE001
            continue

    raise RuntimeError(
        f"Could not read ACTNUM from case (tried STATIC_NATIVE and "
        f"INPUT_PROPERTY, expected {expected_size} cells)"
    )


class GridPropertyReader(_BaseResInsightDataRW):
    """Read grid properties from ResInsight using rips."""

    def __init__(
        self,
        instance_or_port: ResInsightInstanceOrPortType | None = None,
    ) -> None:
        super().__init__(instance_or_port)

    def load(
        self,
        case_name: str,
        property_name: str,
        property_type: str | PropertyType = "STATIC_NATIVE",
        time_step_index: int = 0,
        find_last: bool = True,
    ) -> GridPropertyDataResInsight:
        """Load a grid property from selected ResInsight case."""
        validated_type = _validate_property_type(property_type)

        case = self.get_case(case_name=case_name, find_last=find_last)
        if case is None:
            raise RuntimeError(f"Cannot find any case with name '{case_name}'")

        dims = case.grids()[0].dimensions()  # type: ignore[attr-defined]
        nx, ny, nz = dims.i, dims.j, dims.k

        actnum = _read_actnum(case, nx * ny * nz)

        discrete = (
            case.property_data_type(  # type: ignore[attr-defined]
                property_type=validated_type,
                property_name=property_name,
            )
            == PropertyDataType.INTEGER
        )

        raw = np.asarray(
            case.grid_property(  # type: ignore[attr-defined]
                validated_type, property_name, time_step_index
            ),
            dtype=np.float64,
        )
        # Replace non-finite values (inf at inactive cells) with fill sentinel.
        fill = UNDEF_INT if discrete else UNDEF
        non_finite = ~np.isfinite(raw)
        if np.any(non_finite):
            raw[non_finite] = fill

        values = raw.astype(np.int32) if discrete else raw.astype(np.float64)

        codes: dict[int, str] = {}
        if discrete:
            raw_codes = (
                case.discrete_property_category_names(  # type: ignore[attr-defined]
                    property_name
                )
                or {}
            )
            codes = {int(k): str(v) for k, v in raw_codes.items()}

        return GridPropertyDataResInsight(
            name=property_name,
            nx=nx,
            ny=ny,
            nz=nz,
            values=values,
            actnumsv=actnum,
            property_type=validated_type,
            time_step_index=time_step_index,
            discrete=discrete,
            codes=codes,
            filesrc=getattr(case, "file_path", "") or "",
        )


class GridPropertyWriter(_BaseResInsightDataRW):
    """Write grid properties to ResInsight using rips."""

    def __init__(
        self,
        instance_or_port: ResInsightInstanceOrPortType | None = None,
    ) -> None:
        super().__init__(instance_or_port)

    def save(
        self,
        data: GridPropertyDataResInsight,
        case_name: str,
        find_last: bool = True,
    ) -> None:
        """Save a grid property into selected ResInsight case."""
        case = self.get_case(case_name=case_name, find_last=find_last)
        if case is None:
            raise RuntimeError(
                f"Cannot find any case with name '{case_name}' for property export"
            )

        # Validate grid dimensions before writing.
        dims = case.grids()[0].dimensions()  # type: ignore[attr-defined]
        if (data.nx, data.ny, data.nz) != (dims.i, dims.j, dims.k):
            raise ValueError(
                f"Property '{data.name}' has dimensions "
                f"{data.nx}x{data.ny}x{data.nz}, but case '{case_name}' grid "
                f"has {dims.i}x{dims.j}x{dims.k}"
            )

        validated_type = _validate_property_type(data.property_type)
        try:
            extra = {"data_type": PropertyDataType.INTEGER} if data.discrete else {}

            # ResInsight expects NaN for inactive cells.
            send_values = data.values.astype(np.float64)
            send_values[data.actnumsv == 0] = np.nan

            case.set_grid_property(  # type: ignore[attr-defined]
                send_values.tolist(),
                validated_type,
                data.name,
                data.time_step_index,
                **extra,
            )

            if data.discrete and data.codes:
                case.set_discrete_property_category_names(  # type: ignore[attr-defined]
                    property_name=data.name,
                    value_names={int(k): str(v) for k, v in data.codes.items()},
                )

            logger.debug(
                "Saved property '%s' (type=%s, time_step=%d) to case '%s'",
                data.name,
                validated_type,
                data.time_step_index,
                case_name,
            )

        except Exception as exc:
            raise RuntimeError(
                f"Failed to save property '{data.name}' to ResInsight case "
                f"'{case_name}': {exc}"
            ) from exc
