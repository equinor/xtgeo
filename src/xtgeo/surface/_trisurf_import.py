"""Import functions for triangulated surfaces.

Each format has a corresponding reader that parses the file and returns
a :class:`~xtgeo.surface._trisurf_primitives.TriangulatedSurfaceDict`
suitable for constructing a
:class:`~xtgeo.surface.triangulated_surface.TriangulatedSurface`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from xtgeo.common.log import null_logger
from xtgeo.common.xtgeo_dialog import XTGeoDialog
from xtgeo.io.tsurf._tsurf_io import TSurfData

if TYPE_CHECKING:
    from xtgeo.io._file import FileWrapper
    from xtgeo.surface._trisurf_primitives import TriangulatedSurfaceDict


xtg = XTGeoDialog()

logger = null_logger(__name__)


def import_tsurf(mfile: FileWrapper) -> TriangulatedSurfaceDict:
    """Import a TSurf file and return a format-neutral surface dict.

    Converts 1-based triangle indices from the TSurf format to 0-based.
    If the file contains format-specific data, it is stored under
    ``free_form_metadata["key"]`` for round-trip fidelity.

    Args:
        mfile: path to input file.

    Returns:
        A :class:`TriangulatedSurfaceDict` ready for
        :meth:`TriangulatedSurface.from_dict`.
    """

    tsurf_data = TSurfData.from_file(mfile.file).to_dict()

    # Uses 0-based indexing, so we need to convert from 1-based
    # indexing used in TSurf files
    tris = tsurf_data["triangles"].copy()  # Avoid modifying original data
    tris -= 1

    data: TriangulatedSurfaceDict = {
        "vertices": tsurf_data["vertices"],
        "triangles": tris,
        "name": tsurf_data["header"]["name"],
        "filesrc": "unknown",  # To be set by caller if needed
    }

    if "coord_sys" in tsurf_data:
        data["free_form_metadata"] = {"tsurf_coord_sys": tsurf_data["coord_sys"]}

    return data
