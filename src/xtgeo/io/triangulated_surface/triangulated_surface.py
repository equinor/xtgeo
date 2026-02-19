import copy
from dataclasses import dataclass
from enum import Enum
from typing import Union

import numpy as np
import numpy.typing as npt
from typing_extensions import Self

from xtgeo.common.log import null_logger
from xtgeo.common.types import FileLike
from xtgeo.common_geometry.bounding_box_3d import BoundingBox3D
from xtgeo.io.tsurf._tsurf_io import TSurfCoordSys, TSurfData, TSurfHeader

logger = null_logger(__name__)


class TriangulatedSurfaceFileFormat(Enum):
    """Supported file formats for I/O of triangulated surfaces."""

    TSURF = "tsurf"
    GXF = "gxf"  # Not yet implemented


class TSurfInfoKeyType(Enum):
    """
    Types of additional (meta-) data that can be stored with the triangulated surface.
    They vary from format to format and may include headers, metadata, etc.
    """

    HEADER = "header"
    COORDSYS = "coordsys"


# types of additional information that can be stored for TSurf files
TSurfInfoValueType = Union[TSurfHeader, TSurfCoordSys]


class GxfInfoKeyType(Enum):
    """Types of additional (meta-) data for GXF format files.

    To be populated when GXF support is implemented.
    """

    DUMMYTYPE_1 = "dummytype1"
    DUMMYTYPE_2 = "dummytype2"


# types of additional information that can be stored for GXF files
GxfInfoValueType = Union[object]


# Discriminated union: a single dict is either all-TSurf or all-GXF keyed
InfoType = Union[
    dict[TSurfInfoKeyType, TSurfInfoValueType],
    dict[GxfInfoKeyType, GxfInfoValueType],
]


@dataclass(frozen=True, init=False)
class TriangulatedSurface:
    """Internal data container for read/write of triangulated surfaces in 3D space.
    It is immutable to ensure data integrity during I/O operations.

    Args:
        _vertices: A numpy array of shape (n_vertices, 3) with (x, y, z)
            coordinates for each vertex.
        _triangles: A 0-based numpy array of shape (n_triangles, 3), each row
            containing three integer indices that each refer to a vertex in
            the vertices array.
        _info: A single dictionary of format-specific metadata, keyed by
            TSurfInfoKeyType or GxfInfoKeyType.  Defaults to an empty dict.
    """

    _vertices: npt.NDArray[np.float64]
    _triangles: npt.NDArray[np.int_]
    _info: InfoType

    def __init__(
        self,
        _vertices: npt.NDArray[np.float64],
        _triangles: npt.NDArray[np.int_],
        _info: InfoType | None = None,
    ) -> None:
        object.__setattr__(self, "_vertices", _vertices)
        object.__setattr__(self, "_triangles", _triangles)

        if _info is None:
            _info = {}

        if not isinstance(_info, dict):
            raise ValueError("Info must be a dictionary.")

        has_tsurf_info = False
        has_gxf_info = False

        for key, value in _info.items():
            if isinstance(key, TSurfInfoKeyType):
                has_tsurf_info = True
                if not isinstance(value, (TSurfHeader, TSurfCoordSys)):
                    raise ValueError(
                        f"Invalid tsurf info value for key {key}: {value}. "
                        "Must be of type TSurfHeader or TSurfCoordSys."
                    )
            elif isinstance(key, GxfInfoKeyType):
                has_gxf_info = True
                # GXF value validation will be added when concrete
                # GxfInfoValueType members are defined
            else:
                raise ValueError(
                    f"Invalid info key: {key}. "
                    "Must be of type TSurfInfoKeyType or GxfInfoKeyType."
                )

        if has_tsurf_info and has_gxf_info:
            raise ValueError(
                "Cannot mix TSurf and GXF info keys in a single "
                "TriangulatedSurface instance."
            )

        object.__setattr__(self, "_info", _info)

        self.validate_triangulation_data(self._vertices, self._triangles)

    @property
    def vertices(self) -> npt.NDArray[np.float64]:
        """
        Get a read-only view of the vertices of the surface.

        Returns:
        A numpy array of (x, y, z) coordinates for each vertex.
        """
        v = self._vertices.view()
        v.flags.writeable = False
        return v

    @property
    def triangles(self) -> npt.NDArray[np.int_]:
        """
        Get a read-only view of the triangle indices of the surface.

        Returns:
        A numpy array of 0-based triangle vertex indices.
        """
        t = self._triangles.view()
        t.flags.writeable = False
        return t

    @property
    def num_vertices(self) -> int:
        """
        Get the number of vertices in the surface.
        """
        return len(self._vertices)

    @property
    def num_triangles(self) -> int:
        """
        Get the number of triangles in the surface.
        """
        return len(self._triangles)

    @property
    def bounding_box(self) -> BoundingBox3D:
        """
        Get the axis-aligned bounding box of the triangulated surface.
        """
        return self._compute_bounding_box()

    @property
    def tsurf_info(self) -> dict[TSurfInfoKeyType, TSurfInfoValueType]:
        """Additional information associated with the TSurf triangulated surface.
        Returns an empty dict if the instance does not hold TSurf info.
        """
        if self._info and isinstance(next(iter(self._info)), TSurfInfoKeyType):
            return self._info  # type: ignore[return-value]
        return {}

    @property
    def gxf_info(self) -> dict[GxfInfoKeyType, GxfInfoValueType]:
        """Additional information associated with the GXF triangulated surface.
        Returns an empty dict if the instance does not hold GXF info.
        """
        if self._info and isinstance(next(iter(self._info)), GxfInfoKeyType):
            return self._info  # type: ignore[return-value]
        return {}

    @property
    def info(self) -> InfoType:
        """Get the format-specific metadata dictionary."""
        return self._info

    def _compute_bounding_box(self) -> BoundingBox3D:
        """Compute the axis-aligned bounding box of the triangulated surface.

        Returns:
            BoundingBox3D: The bounding box.

        Raises:
            ValueError: If the bounding box cannot be computed.
        """
        min_x = np.min(self._vertices[:, 0])
        max_x = np.max(self._vertices[:, 0])
        min_y = np.min(self._vertices[:, 1])
        max_y = np.max(self._vertices[:, 1])
        min_z = np.min(self._vertices[:, 2])
        max_z = np.max(self._vertices[:, 2])

        if (
            np.isnan(min_x)
            or np.isnan(max_x)
            or np.isnan(min_y)
            or np.isnan(max_y)
            or np.isnan(min_z)
            or np.isnan(max_z)
        ):
            raise ValueError(
                "Bounding box cannot be computed due to NaN values in vertices."
            )

        if (
            np.isclose(min_x, max_x)
            and np.isclose(min_y, max_y)
            and np.isclose(min_z, max_z)
        ):
            raise ValueError("Bounding box cannot be computed for degenerate surface.")

        return BoundingBox3D((min_x, max_x, min_y, max_y, min_z, max_z))

    @staticmethod
    def _compute_triangle_areas(
        vertices: npt.NDArray[np.float64],
        triangles: npt.NDArray[np.int_],
    ) -> npt.NDArray[np.float64]:
        tri_vertices = vertices[triangles]
        v0 = tri_vertices[:, 0, :]
        v1 = tri_vertices[:, 1, :]
        v2 = tri_vertices[:, 2, :]
        cross = np.cross(v1 - v0, v2 - v0)
        return 0.5 * np.linalg.norm(cross, axis=1)

    @staticmethod
    def validate_triangulation_data(
        vertices: npt.NDArray[np.float64], triangles: npt.NDArray[np.int_]
    ) -> None:
        """Basic validation of triangulation data.

        Args:
            vertices: Array of vertex coordinates
            triangles: Array of triangle vertex indices

        Raises:
            ValueError: If any validation check fails
        """

        if not isinstance(vertices, np.ndarray):
            raise ValueError("Vertices must be a numpy array.")
        if not isinstance(triangles, np.ndarray):
            raise ValueError("Triangles must be a numpy array.")

        if not issubclass(vertices.dtype.type, np.floating):
            raise ValueError("Vertices array must consist of floating point numbers.")
        if not issubclass(triangles.dtype.type, np.integer):
            raise ValueError("Triangles array must consist of integers.")

        if triangles.ndim != 2 or triangles.shape[1] != 3:
            raise ValueError(
                "Triangles array must be 2-dimensional with shape (num_triangles, 3)."
            )

        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError(
                "Vertices array must be 2-dimensional with shape (num_vertices, 3)."
            )

        # Need at least one triangle to form a surface
        if triangles.size == 0:
            raise ValueError("No triangles found in the triangulation data.")

        # Need at least three vertices to form a triangle
        if vertices.shape[0] < 3:
            raise ValueError("At least three vertices are required to form triangles.")

        # Verify that vertices do not contain NaN values
        if np.isnan(vertices).any():
            raise ValueError("Vertices array must not contain NaN values.")

        # Verify that the triangulation is 0-based (at least one index is zero)
        # So indices must be in the closed range [0, number of vertices - 1]
        if np.any(triangles < 0):
            raise ValueError(
                "Triangle vertex indices must be >= 0 in triangulation data."
            )

        if np.any(triangles >= len(vertices)):
            raise ValueError(
                "Triangle vertex indices must be < number of vertices in "
                "triangulation data."
            )

        if np.any(
            (triangles[:, 0] == triangles[:, 1])
            | (triangles[:, 0] == triangles[:, 2])
            | (triangles[:, 1] == triangles[:, 2])
        ):
            raise ValueError("Triangles must not contain duplicate vertex indices.")

        areas = TriangulatedSurface._compute_triangle_areas(vertices, triangles)
        if np.any(np.isclose(areas, 0.0)):
            raise ValueError("Triangles must not be degenerate (zero area).")

    def copy(self) -> Self:
        """Deep copy of the TriangulatedSurface instance."""

        return type(self)(
            _vertices=self._vertices.copy(),
            _triangles=self._triangles.copy(),
            _info=copy.deepcopy(self._info),
        )

    @classmethod
    def from_file(
        cls,
        filepath: FileLike,
        fformat: TriangulatedSurfaceFileFormat = TriangulatedSurfaceFileFormat.TSURF,
    ) -> Self:
        """Read triangulated surface from a file with format selection.

        Args:
            filepath: Path to input file
            fformat: File format (TriangulatedSurfaceFileFormat enum)
        """
        if fformat == TriangulatedSurfaceFileFormat.TSURF:
            return cls._from_tsurf_file(filepath)
        if fformat == TriangulatedSurfaceFileFormat.GXF:
            raise NotImplementedError("GXF format is not yet supported")

        raise NotImplementedError(f"File format {fformat} is not supported")

    @classmethod
    def _from_tsurf_file(
        cls,
        filepath: FileLike,
    ) -> Self:
        """Read a triangulated surface from a TSurf file.

        Args:
            filepath: Path to input file
        """

        tsurf_data = TSurfData.from_file(file=filepath)

        logger.debug(
            "Successfully read tsurf surface '%s' with %d vertices and %d triangles",
            tsurf_data.header.name,
            tsurf_data.vertices.shape[0],
            tsurf_data.triangles.shape[0],
        )

        # This class uses 0-based indexing, so we need to convert from 1-based
        # indexing used in TSurf files
        tris = tsurf_data.triangles.copy()  # Avoid modifying original data
        tris -= 1

        tsurf_info: dict[TSurfInfoKeyType, TSurfInfoValueType] = {}
        tsurf_info[TSurfInfoKeyType.HEADER] = tsurf_data.header
        if tsurf_data.coord_sys is not None:
            tsurf_info[TSurfInfoKeyType.COORDSYS] = tsurf_data.coord_sys

        return cls(
            _vertices=tsurf_data.vertices,
            _triangles=tris,
            _info=tsurf_info,
        )

    def to_file(
        self,
        filepath: FileLike,
        fformat: TriangulatedSurfaceFileFormat = TriangulatedSurfaceFileFormat.TSURF,
    ) -> None:
        """Write triangulated surface to a file with format selection.

        Args:
            filepath: Path to output file
            fformat: File format (TriangulatedSurfaceFileFormat enum)
        """
        if fformat == TriangulatedSurfaceFileFormat.TSURF:
            self._to_tsurf_file(filepath)

        elif fformat == TriangulatedSurfaceFileFormat.GXF:
            self._to_gxf_file(filepath)

        else:
            raise NotImplementedError(f"File format {fformat} is not supported")

    def _to_tsurf_file(self, filepath: FileLike) -> None:
        """Write the triangulated surface to a TSurf file.

        Args:
            filepath: Path to output file
        """
        # Convert from 0-based to 1-based indexing for TSurf format
        tris = self._triangles.copy()  # Avoid modifying original data
        tris += 1

        info = self.tsurf_info

        if TSurfInfoKeyType.HEADER not in info:
            raise ValueError("Header information is required for writing TSurf files.")
        _header_value = info[TSurfInfoKeyType.HEADER]
        if not isinstance(_header_value, TSurfHeader):
            raise ValueError("The provided header is not of type TSurfHeader.")
        header: TSurfHeader = _header_value

        coord_sys = None
        if TSurfInfoKeyType.COORDSYS in info:
            coord_sys_value = info[TSurfInfoKeyType.COORDSYS]
            if not isinstance(coord_sys_value, TSurfCoordSys):
                raise ValueError(
                    "The provided coordinate system is not of type TSurfCoordSys."
                )
            coord_sys = coord_sys_value

        tsurf_data = TSurfData(
            vertices=self._vertices,
            triangles=tris,
            header=header,
            coord_sys=coord_sys,
        )

        tsurf_data.to_file(file=filepath)

    def _to_gxf_file(self, filepath: FileLike) -> None:
        """Write the triangulated surface to a GXF file.

        Args:
            filepath: Path to output file
        """
        raise NotImplementedError("GXF format is not yet supported")
