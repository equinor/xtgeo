"""Triangulated surface primitives and type definitions"""

from typing import Any, NotRequired, TypedDict, TypeVar

import numpy as np
import numpy.typing as npt

try:
    from typing import closed  # type: ignore[attr-defined]
except ImportError:
    _T = TypeVar("_T")

    def closed(cls: _T) -> _T:  # no-op fallback
        return cls


@closed
class TriangulatedSurfaceDict(TypedDict):
    """Format-neutral intermediary for import/export of triangulated surfaces.

    Acts as the contract between the IO layer (format-specific readers/writers)
    and the TriangulatedSurface domain class, avoiding circular imports between
    them.  Import functions produce this dict; export functions consume it.

    ``free_form_metadata`` is an escape hatch for format-specific data that must
    survive a round-trip (import → domain object → export).  For example, the
    TSurf coordinate-system block is stored under the key ``tsurf_coord_sys``.
    Each format should use a unique, format-prefixed key inside this dict so
    that entries from different formats never collide.
    """

    vertices: npt.NDArray[np.float64]
    triangles: npt.NDArray[np.int_]
    filesrc: NotRequired[str]
    # fformat is intentionally omitted — it is set at time of import/export
    # by the caller, not carried through the dict.
    name: NotRequired[str]
    free_form_metadata: NotRequired[dict[str, Any]]
