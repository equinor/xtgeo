"""Export functions for triangulated surfaces."""

from __future__ import annotations

from typing import TYPE_CHECKING

from xtgeo.common.log import null_logger
from xtgeo.io.tsurf._tsurf_io import TSurfCoordSys, TSurfData, TSurfHeader

if TYPE_CHECKING:
    from xtgeo.io._file import FileWrapper
    from xtgeo.surface._trisurf_primitives import TriangulatedSurfaceDict


logger = null_logger(__name__)


def export_tsurf(data: TriangulatedSurfaceDict, mfile: FileWrapper) -> None:
    """Write a triangulated surface to a TSurf file.

    Converts 0-based triangle indices to 1-based as required by the TSurf
    format, and restores the coordinate-system block from
    ``free_form_metadata["tsurf_coord_sys"]`` when present.

    Args:
        data: Format-neutral surface dict.
        mfile: Path to output file.
    """

    # Convert from 0-based to 1-based indexing for TSurf format
    tris = data["triangles"].copy()  # Avoid modifying original data
    tris += 1

    coord_sys = None
    if "free_form_metadata" in data and "tsurf_coord_sys" in data["free_form_metadata"]:
        cs = data["free_form_metadata"]["tsurf_coord_sys"]
        coord_sys = TSurfCoordSys(**cs)

    tsurf_data: TSurfData = TSurfData(
        header=TSurfHeader(name=data.get("name", "Unknown")),
        coord_sys=coord_sys,
        vertices=data["vertices"],
        triangles=tris,
    )

    tsurf_data.to_file(file=mfile.file)
