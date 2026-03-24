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


@dataclass(frozen=True)
class GridDataResInsight:
    """Immutable data container for ResInsight grid metadata."""

    name: str
    nx: int
    ny: int
    nz: int
    coordsv: npt.NDArray[np.float32] = field(repr=False)
    zcornsv: npt.NDArray[np.float32] = field(repr=False)
    actnumsv: npt.NDArray[np.int32] = field(repr=False)
    filesrc: str

    def __post_init__(self) -> None:
        # Validate that the array sizes match the expected dimensions
        expected_coordsv_size = (self.nx + 1) * (self.ny + 1) * 6
        expected_zcornsv_size = self.nx * self.ny * self.nz * 8
        expected_actnumsv_size = self.nx * self.ny * self.nz

        if len(self.coordsv) != expected_coordsv_size:
            raise ValueError(
                f"coordsv should have length {expected_coordsv_size}, but got "
                f"length {len(self.coordsv)}"
            )
        if len(self.zcornsv) != expected_zcornsv_size:
            raise ValueError(
                f"zcornsv should have length {expected_zcornsv_size}, but got "
                f"length {len(self.zcornsv)}"
            )
        if len(self.actnumsv) != expected_actnumsv_size:
            raise ValueError(
                f"actnumsv should have length {expected_actnumsv_size}, but got "
                f"length {len(self.actnumsv)}"
            )

    def to_xtgeo_grid(self) -> Grid:
        """Convert this ResInsight grid data back to an XTGeo Grid."""
        from xtgeo.grid3d import _grid_import_ecl
        from xtgeo.grid3d._ecl_grid import GridRelative
        from xtgeo.grid3d.grid import Grid

        # Build XTGeo Grid from flat corner-point arrays and dimensions
        egrid = EGrid.default_settings_grid(
            coord=np.asarray(self.coordsv, dtype=np.float32),
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
        coord = np.asarray(egrid.coord, dtype=np.float32)
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

        zcorn, coord, actnum, nx, ny, nz = case.export_corner_point_grid()
        return GridDataResInsight(
            name=case.name,
            nx=nx,
            ny=ny,
            nz=nz,
            coordsv=np.asarray(coord, dtype=np.float32),
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

        If both case_name and case_id are not specified, create a new case
        """
        try:
            case = self.get_case(case_name=gname, find_last=find_last)
            if case:
                grid_type = type(case).__name__.split(".")[-1]
                if grid_type == "CornerPointCase":
                    logger.debug(
                        f"Found existing case named '{gname}'. Replacing its grid data."
                    )
                    # Replace existing case
                    case.replace_corner_point_grid(
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
                    f"No existing case named '{gname}' found. Creating a new case."
                )

            # Create a new case
            new_case = self.get_project().create_corner_point_grid(
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
