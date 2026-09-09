"""ResInsight API helpers for XTGeo regular surfaces via rips."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from xtgeo.common.log import null_logger

from ._resinsight_base import (
    _BaseResInsightDataRW,
    resolve_folder,
    select_by_name,
)
from ._rips_package import NameConflictPolicy, require_rips

if TYPE_CHECKING:
    from typing import Any

    import numpy.typing as npt

    from xtgeo.surface.regular_surface import RegularSurface


logger = null_logger(__name__)

_DEPTH_PROPERTY_NAME = "Depth"
# ResInsight marks a surface whose depth is a constant instead of a value array
_FIXED_DEPTH_PROPERTY = "FIXED_DEPTH"
_NAME_ATTR = "surface_user_description"


def _find_regular_surface(folder: object, name: str) -> Any | None:
    """Find the ``RegularSurface`` named *name* directly in *folder*.

    Surface names are unique within a folder, so at most one can match.
    """
    surface_cls = require_rips().RegularSurface
    surfaces = folder.surfaces_field()  # type: ignore[attr-defined]
    return select_by_name(
        (surf for surf in surfaces if isinstance(surf, surface_cls)),
        name,
        find_last=False,
        name_attr=_NAME_ATTR,
    )


@dataclass(frozen=True, eq=False)
class RegularSurfaceDataResInsight:
    """Immutable data container for ResInsight regular-surface metadata.

    ``values`` is flat and in X-fastest (Fortran) order, i.e. equivalent to
    ``RegularSurface.values.ravel(order="F")``.
    """

    name: str
    ncol: int
    nrow: int
    xori: float
    yori: float
    xinc: float
    yinc: float
    rotation: float
    values: npt.NDArray[np.float64] = field(repr=False)

    __hash__ = None  # type: ignore[assignment]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RegularSurfaceDataResInsight):
            return NotImplemented
        return self._geometry() == other._geometry() and np.array_equal(
            self.values, other.values, equal_nan=True
        )

    def _geometry(self) -> tuple:
        """Return the scalar fields, i.e. everything except ``values``."""
        return (
            self.name,
            self.ncol,
            self.nrow,
            self.xori,
            self.yori,
            self.xinc,
            self.yinc,
            self.rotation,
        )

    def __post_init__(self) -> None:
        expected_size = self.ncol * self.nrow
        if self.values.size != expected_size:
            raise ValueError(
                f"values should have length {expected_size} (ncol={self.ncol} * "
                f"nrow={self.nrow}), but got length {self.values.size}"
            )

    def to_xtgeo_surface(self) -> RegularSurface:
        """Convert this data container to an XTGeo ``RegularSurface``."""
        from xtgeo.surface.regular_surface import RegularSurface

        return RegularSurface(
            ncol=self.ncol,
            nrow=self.nrow,
            xori=self.xori,
            yori=self.yori,
            xinc=self.xinc,
            yinc=self.yinc,
            rotation=self.rotation,
            values=self.values.reshape((self.ncol, self.nrow), order="F"),  # type: ignore[arg-type]
            name=self.name,
        )

    @classmethod
    def from_xtgeo_surface(
        cls, surface: RegularSurface, name: str
    ) -> RegularSurfaceDataResInsight:
        """Create regular-surface data from an XTGeo ``RegularSurface``."""
        values = np.ma.filled(surface.values, fill_value=np.nan)

        return cls(
            name=name,
            ncol=surface.ncol,
            nrow=surface.nrow,
            xori=surface.xori,
            yori=surface.yori,
            xinc=surface.xinc,
            yinc=surface.yinc,
            rotation=surface.rotation,
            values=np.asarray(values, dtype=np.float64).ravel(order="F"),
        )


class RegularSurfaceReader(_BaseResInsightDataRW):
    """Read an XTGeo regular surface from ResInsight using rips."""

    def load(
        self,
        surface_name: str,
        property_name: str | None = None,
        folder_name: str = "",
    ) -> RegularSurfaceDataResInsight:
        """Load a regular surface from ResInsight by name.

        Args:
            surface_name: Name of the surface, unique within its folder.
            property_name: Property to read; ``None`` (default) reads the
                surface's current depth property.
            folder_name: ``/``-separated folder path, empty (default) for the
                root surface folder. Sub-folders are not searched recursively.

        Raises:
            RuntimeError: If the folder or the surface cannot be found.
        """
        target_folder = resolve_folder(
            self.get_project().surface_folder(),  # type: ignore[attr-defined]
            folder_name,
            _NAME_ATTR,
        )
        if target_folder is None:
            raise RuntimeError(f"Cannot find surface folder '{folder_name}'")

        surface = _find_regular_surface(target_folder, surface_name)
        if surface is None:
            raise RuntimeError(
                f"Cannot find any surface with name '{surface_name}' in folder "
                f"'{folder_name or '<root>'}'"
            )

        return self._read_regular_surface(surface, property_name=property_name)

    @staticmethod
    def _read_regular_surface(
        surface: object, property_name: str | None = None
    ) -> RegularSurfaceDataResInsight:
        """Extract geometry and values from a rips ``RegularSurface``."""
        name = surface.surface_user_description  # type: ignore[attr-defined]
        ncol = int(surface.nx)  # type: ignore[attr-defined]
        nrow = int(surface.ny)  # type: ignore[attr-defined]

        prop = (
            property_name
            or getattr(surface, "depth_property", None)
            or _DEPTH_PROPERTY_NAME
        )
        if prop == _FIXED_DEPTH_PROPERTY:
            # Constant-depth surfaces hold a scalar depth, not a value array.
            values = np.full(ncol * nrow, float(surface.depth))  # type: ignore[attr-defined]
        else:
            values_list = surface.get_property(prop)  # type: ignore[attr-defined]
            if values_list is None or len(values_list) == 0:
                raise RuntimeError(
                    f"Property '{prop}' on surface '{name}' is missing or empty"
                )
            values = np.asarray(values_list, dtype=np.float64)

        return RegularSurfaceDataResInsight(
            name=name,
            ncol=ncol,
            nrow=nrow,
            xori=float(surface.origin_x),  # type: ignore[attr-defined]
            yori=float(surface.origin_y),  # type: ignore[attr-defined]
            xinc=float(surface.increment_x),  # type: ignore[attr-defined]
            yinc=float(surface.increment_y),  # type: ignore[attr-defined]
            rotation=float(surface.rotation),  # type: ignore[attr-defined]
            values=values,
        )


class RegularSurfaceWriter(_BaseResInsightDataRW):
    """Write an XTGeo regular surface to ResInsight using rips."""

    def save(
        self,
        data: RegularSurfaceDataResInsight,
        surface_name: str,
        folder_name: str = "",
        property_name: str = _DEPTH_PROPERTY_NAME,
        set_as_depth: bool = True,
        replace: bool = False,
    ) -> None:
        """Save a regular surface to the ResInsight project.

        An existing surface of the same name is updated in place when its
        geometry matches ``data``; otherwise it must be recreated, which
        requires ``replace=True``. See :meth:`RegularSurface.to_resinsight`
        for the user-facing description of the arguments.

        Args:
            data: The regular-surface data to save.
            surface_name: Display name for the surface in ResInsight.
            folder_name: ``/``-separated folder path; missing folders are created.
            property_name: Property to write the values under.
            set_as_depth: Mark the written property as the depth property.
            replace: Allow recreating an existing surface of differing geometry.

        Raises:
            RuntimeError: If the save fails, or if a differing surface would
                have to be replaced while ``replace`` is ``False``.
        """
        target_folder = resolve_folder(
            self.get_project().surface_folder(),  # type: ignore[attr-defined]
            folder_name,
            _NAME_ATTR,
            create=True,
        )
        existing = _find_regular_surface(target_folder, surface_name)
        reuse = existing is not None and self._regular_surface_geometry_matches(
            existing, data
        )

        if existing is not None and not reuse and not replace:
            raise RuntimeError(
                f"A surface named '{surface_name}' already exists in folder "
                f"'{folder_name or '<root>'}' with a different geometry. "
                "Pass replace=True to overwrite it."
            )

        logger.debug(
            "Saving surface '%s' (reuse existing=%s, overwrite=%s)",
            surface_name,
            reuse,
            existing is not None and not reuse,
        )

        try:
            surface = (
                existing
                if reuse
                else self._create_regular_surface(
                    target_folder, surface_name, data, overwrite=replace
                )
            )
            self._set_regular_surface_property(
                surface, data, property_name=property_name, set_as_depth=set_as_depth
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to save ResInsight regular surface data: {exc}"
            ) from exc

    @staticmethod
    def _regular_surface_geometry_matches(
        surface: object, data: RegularSurfaceDataResInsight
    ) -> bool:
        """Check whether an existing surface has the same geometry."""
        if int(surface.nx) != data.ncol or int(surface.ny) != data.nrow:  # type: ignore[attr-defined]
            return False

        # ResInsight stores coordinates as float32, so compare against the
        # float32-rounded expected value rather than a relative tolerance,
        # which would grow too large at typical (large) map origins and
        # accept distinct, materially different coordinates as equal.
        return all(
            np.float32(actual) == np.float32(expected)
            for actual, expected in (
                (surface.origin_x, data.xori),  # type: ignore[attr-defined]
                (surface.origin_y, data.yori),  # type: ignore[attr-defined]
                (surface.increment_x, data.xinc),  # type: ignore[attr-defined]
                (surface.increment_y, data.yinc),  # type: ignore[attr-defined]
                (surface.rotation, data.rotation),  # type: ignore[attr-defined]
            )
        )

    @staticmethod
    def _set_regular_surface_property(
        surface: object,
        data: RegularSurfaceDataResInsight,
        property_name: str = _DEPTH_PROPERTY_NAME,
        set_as_depth: bool = True,
    ) -> None:
        """Set a property on an existing regular surface."""
        values_list = data.values.astype(np.float32).tolist()
        surface.set_property(property_name, values_list)  # type: ignore[attr-defined]
        if set_as_depth:
            surface.set_property_as_depth(property_name)  # type: ignore[attr-defined]

    @staticmethod
    def _create_regular_surface(
        folder: object,
        surface_name: str,
        data: RegularSurfaceDataResInsight,
        overwrite: bool = False,
    ) -> Any:
        """Create an empty ResInsight ``RegularSurface`` inside the folder.

        With *overwrite*, ResInsight deletes any existing item carrying the same
        name in *folder* — including a non-regular surface, which the
        ``RegularSurface`` lookup does not see; otherwise a name clash raises.
        """
        new_surf = folder.new_regular_surface(  # type: ignore[attr-defined]
            name=surface_name,
            on_name_conflict=(
                NameConflictPolicy.OVERWRITE if overwrite else NameConflictPolicy.FAIL
            ),
            origin_x=data.xori,
            origin_y=data.yori,
            depth=0.0,
            nx=data.ncol,
            ny=data.nrow,
            increment_x=data.xinc,
            increment_y=data.yinc,
            rotation=data.rotation,
        )
        new_surf.update()
        return new_surf
