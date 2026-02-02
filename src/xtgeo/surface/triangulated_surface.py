from __future__ import annotations

import copy
import warnings
from typing import TYPE_CHECKING, Any, Callable, Literal, Optional

import numpy as np
import numpy.typing as npt
from typing_extensions import Self

from xtgeo.common.exceptions import InvalidFileFormatError
from xtgeo.common.log import null_logger
from xtgeo.common.sys import generic_hash
from xtgeo.common.xtgeo_dialog import XTGDescription
from xtgeo.common_geometry.bounding_box_3d import BoundingBox3D
from xtgeo.io._file import FileFormat, FileWrapper
from xtgeo.metadata.metadata import MetaDataTriangulatedSurface
from xtgeo.surface import _trisurf_export
from xtgeo.surface._trisurf_import import import_tsurf

if TYPE_CHECKING:
    from io import BytesIO, StringIO
    from pathlib import Path

    from xtgeo.common.types import FileLike
    from xtgeo.surface._trisurf_primitives import TriangulatedSurfaceDict


logger = null_logger(__name__)


def triangulated_surface_from_file(
    mfile: FileLike,
    fformat: Optional[str] = "guess",
) -> TriangulatedSurface:
    """Convenience function to read a triangulated surface from file.

    Args:
        mfile: File-like or memory stream instance.
        fformat: File format. If 'None' or 'guess', the file 'signature' is
            used to guess format first, then file extension.

    Returns:
        TriangulatedSurface instance.

    Example::

        Imports a TSurf file as a TriangulatedSurface instance
        by using a convenience function::

        >>> import xtgeo
        >>> surf = xtgeo.triangulated_surface_from_file(surface_dir + "/filename.ts")
        or
        >>> surf = xtgeo.triangulated_surface_from_file(
        ...     surface_dir + "/filename.unknown_extension", fformat="tsurf"
        ... )

    """
    return TriangulatedSurface._read_file(mfile, fformat=fformat)


def triangulated_surface_from_rms() -> None:
    """This makes an instance of a TriangulatedSurface directly from RMS input."""
    raise NotImplementedError(
        "Import from RMS not implemented yet for TriangulatedSurface."
    )


class TriangulatedSurface:
    """
    Class for a triangulated surface in 3D space in the XTGeo framework.
    It holds the triangulation data and associated metadata, and the original file
    can be recreated from the data and metadata.
    The class is populated via tailored import functions for various file formats,
    and can be correspondingly exported.
    File formats can be recognized by file signature (typically the first few bytes
    of the file) or by file extension.
    The import/export functions are are designed to handle the most common cases and
    to be robust to variations in the files, but are not guaranteed to handle all
    possible variations of the file formats (various keywords, header types, etc).
    The surface is defined by a set of vertices and a set of triangles,
    where each triangle is defined by three vertex indices.
    Hence the topological information is limited as it does not include
    connectivity beyond the triangles themselves.
    Basic validation is performed, typically that the data constitute a valid
    set of vertices and triangles.
    Geometric quality is the responsibility of the source of the triangulation data.
    No assumption is made on the order of the vertices in the triangles,
    on the orientation of the triangles, or otherwise on the geometric
    quality of the triangulation (overlapping triangles, holes in the surface,
    self-intersections, etc).
    The triangles are 0-based, meaning that the indices to the vertices
    in the triangles array start from 0.

    Args:
        _vertices: A numpy array of shape (num_vertices, 3) with (x, y, z)
            coordinates for each vertex.
        _triangles: A 0-based numpy array of shape (num_triangles, 3), each row
            containing three integer indices that each refer to a vertex in
            the vertices array.
        name: A free form name for the surface, to be used in display etc.
        filesrc: The name of the file source (if any).
        fformat: The file format of the source file (if any).
    """

    _vertices: npt.NDArray[np.float64]
    _triangles: npt.NDArray[np.int_]
    _filesrc: Optional[str]
    _fformat: Optional[str]
    _name: Optional[str]

    def __init__(
        self,
        vertices: npt.NDArray[np.float64],
        triangles: npt.NDArray[np.int_],
        filesrc: Optional[str] = None,
        fformat: Optional[str] = "unknown",  # file format of source file
        name: Optional[str] = "unknown",
    ) -> None:
        """
        Instantiating a TriangulatedSurface object.
        Required metadata values are automatically extracted,
        while optional/freeform metadata can be set after instantiation.

        Args:
            vertices: A numpy array of shape (num_vertices, 3) with (x, y, z)
                coordinates for each vertex.
            triangles: A 0-based numpy array of shape (num_triangles, 3), each row
                containing three integer indices that each refer to a vertex in
                the vertices array.
            filesrc: The name of the file source (if any).
            fformat: The file format of the source file (if any).
            name: A free form name for the surface, to be used in display etc.
                Extracted a) from file or b) from file name root,
                or 'unknown' if constructed from scratch.

        Examples:
            The instance can be made by specification::
                >>> vertices = np.array(
                ...     [
                ...         [0.0, 0.0, 0.0],
                ...         [1.0, 0.0, 0.0],
                ...         [1.0, 1.0, 0.0],
                ...         [0.0, 1.0, 0.0],
                ...     ],
                ...     dtype=np.float64,
                ... )
                >>> triangles = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int_)
                >>> surface = TriangulatedSurface(
                ...     vertices=vertices,
                ...     triangles=triangles,
                ... )

        """

        logger.info("Start __init__ method for TriangulatedSurface object %s", id(self))

        if not isinstance(filesrc, (str, type(None))):
            raise ValueError("filesrc must be a string or None.")
        self._filesrc = filesrc

        if not isinstance(fformat, (str, type(None))):
            raise ValueError("file format must be a string or None.")
        self._fformat = fformat

        if not isinstance(name, (str, type(None))):
            raise ValueError("name must be a string or None.")
        self._name = name

        TriangulatedSurface.validate_triangulation_data(vertices, triangles)
        self._vertices = vertices
        self._triangles = triangles

        self._metadata = MetaDataTriangulatedSurface()
        self._metadata.update_from(self._get_required_metadata_values())

    def __repr__(self) -> str:
        """Magic method __repr__."""
        return (
            f"{self.__class__.__name__}"
            f"num_vertices={self.num_vertices!r}, "
            f"num_triangles={self.num_triangles!r}, "
            f"fformat={self._fformat!r}, "
            f"name='{self._name!r}', "
            f"filesrc='{self._filesrc!r}')"
            f"ID={id(self)}."
        )

    def __str__(self) -> str:
        """Magic method __str__ for user friendly print."""
        return self.describe(flush=False)

    def __eq__(self, value: object) -> bool:
        """
        Magic method __eq__ for equality comparison.
        Two instances are considered equal if they have the same
        file source, file format, name, vertices, triangles, and metadata.
        """
        if not isinstance(value, TriangulatedSurface):
            return NotImplemented

        if self is value:
            return True

        return (
            self._filesrc == value._filesrc
            and self._fformat == value._fformat
            and self._name == value._name
            and np.array_equal(self._vertices, value._vertices)
            and np.array_equal(self._triangles, value._triangles)
            and self.metadata == value.metadata
        )

    @property
    def metadata(self) -> MetaDataTriangulatedSurface:
        """Return metadata object instance of type MetaDataTriangulatedSurface."""
        return self._metadata

    @metadata.setter
    def metadata(self, obj: MetaDataTriangulatedSurface) -> None:
        """Set the metadata object instance of type MetaDataTriangulatedSurface."""
        if not isinstance(obj, MetaDataTriangulatedSurface):
            raise ValueError("Input obj not an instance of MetaDataTriangulatedSurface")

        if not MetaDataTriangulatedSurface.is_valid_for(
            self._get_required_metadata_values(), obj
        ):
            raise ValueError("Metadata does not match this TriangulatedSurface")
        self._metadata = obj

    def _get_required_metadata_values(self) -> dict[str, Any]:
        """Return a dict of required metadata values for this surface."""
        values = {
            "num_vertices": self.num_vertices,
            "num_triangles": self.num_triangles,
        }
        assert values.keys() == MetaDataTriangulatedSurface.REQUIRED.keys(), (
            f"_get_required_metadata_values keys {set(values.keys())} do not match "
            f"REQUIRED keys {set(MetaDataTriangulatedSurface.REQUIRED.keys())}"
        )
        return values

    @property
    def vertices(self) -> npt.NDArray[np.float64]:
        """
        Get read-only view of the vertices.

        Returns:
        A numpy array of (x, y, z) coordinates for each vertex.
        """
        # Get view to prevent in-place modifications without updating metadata
        view = self._vertices.view()
        view.flags.writeable = False
        return view

    @vertices.setter
    def vertices(self, new_vertices: npt.NDArray[np.float64]) -> None:
        """Set new vertices and update metadata."""
        self.validate_triangulation_data(new_vertices, self._triangles)
        self._vertices = new_vertices
        self._metadata.update_from(self._get_required_metadata_values())

    @property
    def triangles(self) -> npt.NDArray[np.int_]:
        """
        Get read-only view of the triangles.

        Returns:
        A numpy array of 0-based triangle vertex indices.
        """
        # Get view to prevent in-place modifications without updating metadata
        view = self._triangles.view()
        view.flags.writeable = False
        return view

    @triangles.setter
    def triangles(self, new_triangles: npt.NDArray[np.int_]) -> None:
        """Set new triangles and update metadata."""
        self.validate_triangulation_data(self._vertices, new_triangles)
        self._triangles = new_triangles
        self._metadata.update_from(self._get_required_metadata_values())

    @property
    def num_vertices(self) -> int:
        """
        Get the number of vertices in the surface.
        """
        return self._vertices.shape[0]

    @property
    def num_triangles(self) -> int:
        """
        Get the number of triangles in the surface.
        """
        return self._triangles.shape[0]

    @property
    def bounding_box(self) -> BoundingBox3D:
        """
        Get the axis-aligned bounding box of the triangulated surface.
        """
        return self._compute_bounding_box()

    @property
    def name(self) -> Optional[str]:
        """A free form name for the surface, to be used in display etc."""
        return self._name

    @name.setter
    def name(self, newname: str) -> None:
        """Set a free form name for the surface, to be used in display etc."""
        if isinstance(newname, str) and newname.strip():
            self._name = newname

    @property
    def filesrc(self) -> Optional[str]:
        """The name of the file source."""
        return self._filesrc

    @filesrc.setter
    def filesrc(self, name: str) -> None:
        """Set the name of the file source."""
        self._filesrc = name

    @property
    def fformat(self) -> Optional[str]:
        """The file format of the source file."""
        return self._fformat

    @fformat.setter
    def fformat(self, fmt: Optional[str]) -> None:
        """Set the file format of the source file."""
        self._fformat = fmt

    def to_dict(self) -> TriangulatedSurfaceDict:
        """Return a format-neutral dictionary representation of this surface.

        The returned dict can be passed to :meth:`from_dict` to reconstruct
        the surface, or handed to an export function.
        """
        tri_data: TriangulatedSurfaceDict = {
            "vertices": self._vertices,
            "triangles": self._triangles,
        }
        tri_data["filesrc"] = self._filesrc if self._filesrc else "unknown"
        tri_data["name"] = self._name if self._name else "unknown"

        if self.metadata.freeform and "tsurf_coord_sys" in self.metadata.freeform:
            tri_data["free_form_metadata"] = {
                "tsurf_coord_sys": self.metadata.freeform["tsurf_coord_sys"]
            }

        return tri_data

    @classmethod
    def from_dict(
        cls,
        data: TriangulatedSurfaceDict,
        fformat: Optional[str] = "unknown",
        **overrides: Any,
    ) -> Self:
        """Construct a TriangulatedSurface from a TriangulatedSurfaceDict.

        This is the symmetric counterpart of :meth:`to_dict`.  It is the
        preferred way for IO functions to build domain objects from the
        format-neutral intermediary dict.

        Args:
            data: A :class:`TriangulatedSurfaceDict` as produced by an import
                function or by :meth:`to_dict`.
            fformat: File format tag (not stored in the dict by design).
            **overrides: Any key that also exists in *data* can be overridden
                here.  Typical use: ``filesrc`` and ``name`` when the caller
                has better information than the import function.

        Returns:
            A new TriangulatedSurface instance.
        """
        instance = cls(
            vertices=overrides.pop("vertices", data["vertices"]),
            triangles=overrides.pop("triangles", data["triangles"]),
            name=overrides.pop("name", data.get("name", "unknown")),
            filesrc=overrides.pop("filesrc", data.get("filesrc")),
            fformat=overrides.pop("fformat", fformat),
        )

        if "free_form_metadata" in data:
            instance.metadata.freeform.update(data["free_form_metadata"])

        return instance

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
                # Should not happen, validation should have caught this
                "Bounding box cannot be computed due to NaN values in vertices."
            )

        if (
            np.isclose(min_x, max_x)
            and np.isclose(min_y, max_y)
            and np.isclose(min_z, max_z)
        ):
            raise ValueError("Bounding box cannot be computed for degenerate surface.")

        return BoundingBox3D((min_x, max_x, min_y, max_y, min_z, max_z))

    def compute_triangle_areas(self) -> npt.NDArray[np.float64]:
        """Compute the area of each triangle.

        Returns:
            Array of shape ``(num_triangles,)`` with the area of each triangle.
        """
        tri_vertices = self._vertices[self._triangles]
        v0 = tri_vertices[:, 0, :]
        v1 = tri_vertices[:, 1, :]
        v2 = tri_vertices[:, 2, :]
        cross = np.cross(v1 - v0, v2 - v0)
        return 0.5 * np.linalg.norm(cross, axis=1)

    def compute_triangle_normals(self) -> npt.NDArray[np.float64]:
        """Compute the unit normal of each triangle.

        Degenerate triangles (zero area) get a zero-length normal to
        avoid division by zero.

        Returns:
            Array of shape ``(num_triangles, 3)``.
        """
        tri_vertices = self._vertices[self._triangles]
        v0 = tri_vertices[:, 0, :]
        v1 = tri_vertices[:, 1, :]
        v2 = tri_vertices[:, 2, :]
        cross = np.cross(v1 - v0, v2 - v0)
        norms = np.linalg.norm(cross, axis=1)
        # Avoid division by zero for degenerate triangles
        norms[norms == 0] = 1.0
        return cross / norms[:, np.newaxis]

    def compute_triangle_centroids(self) -> npt.NDArray[np.float64]:
        """Compute the centroid of each triangle.

        Returns:
            Array of shape ``(num_triangles, 3)``.
        """
        tri_vertices = self._vertices[self._triangles]
        return np.mean(tri_vertices, axis=1)

    def compute_centroid(self) -> npt.NDArray[np.float64]:
        """Compute the centroid of the surface (mean of all vertices).

        Returns:
            Array of shape ``(3,)``.
        """
        return np.mean(self._vertices, axis=0)

    def compute_surface_area(self) -> float:
        """Compute the total surface area (sum of all triangle areas)."""
        areas = self.compute_triangle_areas()
        return np.sum(areas)

    def generate_hash(
        self, hashmethod: Literal["md5", "sha256", "blake2d"] | Callable = "md5"
    ) -> str:
        """Return a unique hash ID for current instance.

        See :meth:`~xtgeo.common.sys.generic_hash()` for documentation.

        """
        gid = ""
        gid += self._vertices.data.tobytes().hex()
        gid += self._triangles.data.tobytes().hex()

        return generic_hash(gid, hashmethod=hashmethod)

    def describe(self, flush: bool = True) -> str:
        """
        Return a string description of the triangulated surface.

        Args:
            flush: If True, print the description to stdout.
                   If False, only return the description string without printing.
        """

        dsc = XTGDescription()
        dsc.title("Description of TriangulatedSurface instance")
        dsc.txt("Object ID", id(self))
        dsc.txt("File source", self._filesrc)
        dsc.txt("Name", self._name)
        dsc.txt("Number of vertices", self.num_vertices)
        dsc.txt("Number of triangles", self.num_triangles)
        np.set_printoptions(threshold=1000)
        dsc.txt("Vertices", self._vertices.reshape(-1), self._vertices.dtype)
        dsc.txt("Triangles", self._triangles.reshape(-1), self._triangles.dtype)
        dsc.txt("Bounding box", self.bounding_box.describe())
        msize_vertices = float(self._vertices.size * 8) / (1024 * 1024 * 1024)
        msize_triangles = float(self._triangles.size * 8) / (1024 * 1024 * 1024)
        dsc.txt(
            "Minimum memory usage of array (GB):",
            f"Vertices: {msize_vertices:.6f} GB, Triangles: {msize_triangles:.6f} GB",
            f"Total: {msize_vertices + msize_triangles:.6f} GB",
        )

        if flush:
            dsc.flush()

        return dsc.astext()

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

    def copy(self) -> Self:
        """Deep copy of the TriangulatedSurface instance."""

        cp = type(self)(
            vertices=self._vertices.copy(),
            triangles=self._triangles.copy(),
            filesrc=self._filesrc,
            fformat=self._fformat,
            name=self._name,
        )

        cp.metadata = copy.deepcopy(self._metadata)
        return cp

    @classmethod
    def _read_file(cls, mfile: FileLike, fformat: Optional[str] = "guess") -> Self:
        """Import triangulated surface from file.

        Note that the ``fformat=None`` or ``guess`` option will guess format by
        looking at the file or stream signature or file extension.
        For the signature, the first bytes are scanned for 'patterns'. If that
        does not work (and input is not a memory stream), it will try to use
        file extension where e.g. "ts" will assume a TSurf file.

        Args:
            mfile: File-like instance (including memory stream).
            fformat: File format, e.g. 'tsurf'.
                If None or guess, the file 'signature' is
                used to guess format first, then file extension.

        Returns:
            Object instance.

        """

        wrapped_file = FileWrapper(mfile)
        wrapped_file.check_file(raiseerror=ValueError)
        fmt = wrapped_file.fileformat(fformat)

        if fmt == FileFormat.TSURF:
            return cls._from_tsurf_file(wrapped_file)

        extensions = FileFormat.extensions_string(
            [FileFormat.TSURF],
        )
        raise InvalidFileFormatError(
            f"File format {fmt} is invalid for type TriangulatedSurface. "
            f"Supported formats are {extensions}."
        )

    @classmethod
    def _from_tsurf_file(
        cls,
        filepath: FileWrapper,
    ) -> Self:
        """Read a triangulated surface from a TSurf file.

        Args:
            filepath: Path to input file
        """

        tsurf_data = import_tsurf(filepath)

        logger.debug(
            "Successfully read tsurf surface '%s' with %d vertices and %d triangles",
            tsurf_data["name"],
            tsurf_data["vertices"].shape[0],
            tsurf_data["triangles"].shape[0],
        )

        tri_surf = cls.from_dict(
            tsurf_data,
            filesrc=str(filepath.name),
            fformat=FileFormat.TSURF.value[0],
        )

        # The TSurf coordinate system contains information about the units and
        # the direction of the z-axis, which are critical for the correct
        # interpretation of the surface.
        # But in the TSurf format, the coordinate system is optional.
        # Hence, if it exists, provide a warning if the z-axis is positive upwards,
        # and modify the z-values accordingly, since in FMU z-axis
        # is positive downwards.
        # Note: from_dict already stored free_form_metadata into metadata.freeform.

        if "tsurf_coord_sys" in tri_surf.metadata.freeform:
            tsurf_coord_sys = copy.deepcopy(
                tri_surf.metadata.freeform["tsurf_coord_sys"]
            )
            if "axis_unit" in tsurf_coord_sys:
                coord_sys_units = tsurf_coord_sys["axis_unit"]
                # Just use the unit from the first axis
                if coord_sys_units[0] == "m":
                    tri_surf.metadata.optional["units"] = "metric"
                    # Currently the only choice in metadata.optional["units"]
                    # TODO: could be an enum to pick from
            if "zpositive" in tsurf_coord_sys:
                if tsurf_coord_sys["zpositive"] == "Depth":
                    # z-axis is positive downwards, nothing to do
                    pass
                elif tsurf_coord_sys["zpositive"] == "Elevation":
                    # z-axis is positive upwards, but in FMU z-axis is positive
                    # downwards.
                    # So z-values should be modified.
                    # This may or may not be a simple negation.
                    # It depends on the reference of the z-values,
                    # if they are relative to a datum or absolute.
                    # Since we do not have a reference, we will just
                    # negate the z-values (implying datum is z=0) and warn the user.
                    tri_surf._vertices[:, 2] *= -1

                    # Correspondingly modify the TSurf coord_sys metadata in case of
                    # export back to the TSurf format
                    tsurf_coord_sys["zpositive"] = "Depth"

                    warnings.warn(
                        f"UserWarning: the TSurf file {filepath} has "
                        "a 'COORDINATE_SYSTEM' where 'ZPOSITIVE' is 'Elevation', "
                        "which means that the z-values are positive upwards. "
                        "In FMU z-values are positive downwards, so the imported "
                        "z-values are negated."
                        "If the z-values are relative to a datum that is not at z=0,"
                        "simple negation may not be correct."
                        "Please verify that simple negation is correct "
                        "for your data."
                    )
                    logger.warning(
                        f"Inverted z-values as "
                        f"COORDINATE_SYSTEM.ZPOSITIVE = 'Elevation' "
                        f"in the imported TSurf file {filepath}. "
                        f"The user should verify that this is correct for the data."
                    )
                    # NOTE: If datum is not at z=0, simple negation is
                    # not correct. In that case, an export of this surface yields
                    # a TSurf file with wrong z-values.
                    # However, the COORDINATE_SYSTEM section in the TSurf format
                    # is optional, and there is no standard way to specify the reference
                    # of the z-values (so a non-trivial issue).
                    # Hence, it must be up to the user to provide correct z-values.

            # Store format-specific data, typically for export back to TSurf format
            tri_surf.metadata.freeform["tsurf_coord_sys"] = tsurf_coord_sys

        return tri_surf

    def to_file(
        self,
        filepath: FileLike,
        fformat: str = "guess",
    ) -> Path | BytesIO | StringIO | None:
        """
        Write triangulated surface to a file with format selection.
        If this instance was imported from a file and is exported back to
        the same format, the output file should be essentially the same as
        the input file, except for unhandled keywords, minor differences in formatting,
        ordering of keywords, etc.

        Args:
            filepath: Path to output file
            fformat: File format (str)
        """

        logger.info("Export TriangulatedSurface to file or memstream...")

        mfile = FileWrapper(filepath)
        mfile.check_folder(raiseerror=OSError)
        fmt = mfile.fileformat(fformat)

        if fmt == FileFormat.TSURF:
            _trisurf_export.export_tsurf(self.to_dict(), mfile)

        else:
            extensions = FileFormat.extensions_string(
                [
                    FileFormat.TSURF,
                ]
            )
            raise InvalidFileFormatError(
                f"File format {fformat} is invalid for type TriangulatedSurface. "
                f"Supported formats are {extensions}."
            )

        logger.info("Export TriangulatedSurface to file or memstream... done")

        if mfile.memstream:
            return None
        return mfile.file

    def to_rms(self) -> None:
        raise NotImplementedError(
            "Export to RMS not implemented yet for TriangulatedSurface."
        )

    def to_hdf(self) -> None:
        raise NotImplementedError(
            "Export to HDF not implemented yet for TriangulatedSurface."
        )
