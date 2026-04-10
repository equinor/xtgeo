"""ResInsight API helpers for grid metadata via rips."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from xtgeo.common.log import null_logger
from xtgeo.grid3d._egrid import EGrid

from ._resinsight_base import _BaseResInsightDataRW

if TYPE_CHECKING:
    import numpy.typing as npt

    from xtgeo.grid3d.grid import Grid

    from ._rips_package import ResInsightInstanceOrPortType

logger = null_logger(__name__)


@dataclass(frozen=True, eq=False)
class GridDataResInsight:
    """Immutable data container for ResInsight grid metadata."""

    name: str
    nx: int
    ny: int
    nz: int
    coordsv: npt.NDArray[np.float64] = field(repr=False)
    zcornsv: npt.NDArray[np.float32] = field(repr=False)
    actnumsv: npt.NDArray[np.int32] = field(repr=False)
    filesrc: str

    # Explicitly unhashable: numpy array fields cannot be hashed reliably.
    __hash__ = None  # type: ignore[assignment]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GridDataResInsight):
            return NotImplemented
        return (
            self.name == other.name
            and self.nx == other.nx
            and self.ny == other.ny
            and self.nz == other.nz
            and np.array_equal(self.coordsv, other.coordsv)
            and np.array_equal(self.zcornsv, other.zcornsv)
            and np.array_equal(self.actnumsv, other.actnumsv)
            and self.filesrc == other.filesrc
        )

    def __post_init__(self) -> None:
        # Validate that the array sizes match the expected dimensions
        expected_coordsv_size = (self.nx + 1) * (self.ny + 1) * 6
        expected_zcornsv_size = self.nx * self.ny * self.nz * 8
        expected_actnumsv_size = self.nx * self.ny * self.nz

        if self.coordsv.size != expected_coordsv_size:
            raise ValueError(
                f"coordsv should have length {expected_coordsv_size}, but got "
                f"length {self.coordsv.size}"
            )
        if self.zcornsv.size != expected_zcornsv_size:
            raise ValueError(
                f"zcornsv should have length {expected_zcornsv_size}, but got "
                f"length {self.zcornsv.size}"
            )
        if self.actnumsv.size != expected_actnumsv_size:
            raise ValueError(
                f"actnumsv should have length {expected_actnumsv_size}, but got "
                f"length {self.actnumsv.size}"
            )

    def to_xtgeo_grid(self) -> Grid:
        """Convert this ResInsight grid data back to an XTGeo Grid."""
        from xtgeo.grid3d import _grid_import_ecl
        from xtgeo.grid3d._ecl_grid import GridRelative
        from xtgeo.grid3d.grid import Grid

        # Build XTGeo Grid from flat corner-point arrays and dimensions
        egrid = EGrid.default_settings_grid(
            coord=np.asarray(self.coordsv, dtype=np.float64),
            zcorn=np.asarray(self.zcornsv, dtype=np.float32),
            actnum=np.asarray(self.actnumsv, dtype=np.int32),
            size=(self.nx, self.ny, self.nz),
        )

        kwargs = _grid_import_ecl.grid_from_ecl_grid(
            egrid, relative_to=GridRelative.MAP
        )
        kwargs["filesrc"] = self.filesrc
        return Grid(**kwargs)

    @classmethod
    def from_xtgeo_grid(
        cls, grid: Grid, name: str, filesrc: str = ""
    ) -> GridDataResInsight:
        """Create a GridDataResInsight from an XTGeo Grid."""
        egrid = EGrid.from_xtgeo_grid(grid)
        nx, ny, nz = egrid.dimensions
        coord = np.asarray(egrid.coord, dtype=np.float64)
        zcorn = np.asarray(egrid.zcorn, dtype=np.float32)
        if egrid.actnum is None:
            actnum = np.ones(nx * ny * nz, dtype=np.int32)
        else:
            actnum = np.asarray(egrid.actnum, dtype=np.int32)

        return cls(
            name=name,
            nx=nx,
            ny=ny,
            nz=nz,
            coordsv=coord,
            zcornsv=zcorn,
            actnumsv=actnum,
            filesrc=filesrc,
        )


class GridReader(_BaseResInsightDataRW):
    """Read grid from ResInsight using rips."""

    def __init__(
        self,
        instance_or_port: ResInsightInstanceOrPortType | None = None,
    ) -> None:
        super().__init__(instance_or_port)

    def load(self, case_name: str, find_last: bool = True) -> GridDataResInsight:
        """Load metadata from selected ResInsight case."""
        case = self.get_case(case_name=case_name, find_last=find_last)
        if case is None:
            raise RuntimeError(f"Cannot find any case with name '{case_name}'")

        zcorn, coord, actnum, nx, ny, nz = case.export_corner_point_grid()  # type: ignore[attr-defined]
        return GridDataResInsight(
            name=case.name,
            nx=nx,
            ny=ny,
            nz=nz,
            coordsv=np.asarray(coord, dtype=np.float64),
            zcornsv=np.asarray(zcorn, dtype=np.float32),
            actnumsv=np.asarray(actnum, dtype=np.int32),
            filesrc=case.file_path or "",
        )


class GridWriter(_BaseResInsightDataRW):
    """Write grid to ResInsight using rips."""

    def __init__(
        self,
        instance_or_port: ResInsightInstanceOrPortType | None = None,
    ) -> None:
        super().__init__(instance_or_port)

    def save(
        self, data: GridDataResInsight, gname: str, find_last: bool = True
    ) -> None:
        """Save grid to selected ResInsight case.

        Args:
            data: The grid metadata to save.
            gname: The name of the case to create or replace in ResInsight.
                If a case with the same name already exists, it will be replaced with
                the new grid data.
                If multiple cases share the same name, the one to replace is determined
                by the `find_last` parameter.

            find_last: Controls which existing case to replace when multiple cases
                share the same `gname`. If `True` (default), the last matching case is
                replaced; if `False`, the first matching case is replaced.

        Note: If an existing case with the same name is found but is not replaceable
        (e.g. it's loaded from grid file), a warning is logged and a new case with the
        same name is created instead of replacing the existing one.
        """
        try:
            case = self.get_case(case_name=gname, find_last=find_last)
            if case:
                grid_type = type(case).__name__.split(".")[-1]
                if grid_type == "CornerPointCase":
                    logger.debug(
                        "Found existing case named '%s'. Replacing its grid data.",
                        gname,
                    )
                    # Replace existing case
                    case.replace_corner_point_grid(  # type: ignore[attr-defined]
                        data.nx,
                        data.ny,
                        data.nz,
                        data.coordsv,
                        data.zcornsv,
                        data.actnumsv,
                    )
                    case.file_path = data.filesrc
                    case.update()
                    return
                logger.warning(
                    "Existing case named '%s' is of type '%s', which is not "
                    "compatible with grid data. Creating a new case with same name "
                    "instead.",
                    gname,
                    grid_type,
                )
            else:
                logger.debug(
                    "No existing case named '%s' found. Creating a new case.", gname
                )

            # Create a new case
            new_case = self.get_project().create_corner_point_grid(  # type: ignore[attr-defined]
                name=gname,
                nx=data.nx,
                ny=data.ny,
                nz=data.nz,
                coord=data.coordsv,
                zcorn=data.zcornsv,
                actnum=data.actnumsv,
            )
            new_case.file_path = data.filesrc
            new_case.update()

        except Exception as exc:
            raise RuntimeError(f"Failed to save ResInsight case data: {exc}") from exc
