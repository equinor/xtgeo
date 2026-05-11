"""
Module for reading and writing the TSurf file format.
Currently only supports a subset of the TSurf format, namely sections with
data for a triangulated surface.
"""

import warnings
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from typing import Any, NotRequired, TypedDict, TypeVar, cast

import numpy as np
import numpy.typing as npt
from typing_extensions import Self

# PEP 728 introduces the `@closed` decorator for TypedDicts to forbid extra keys.
# It is not yet available in the standard `typing` module on all supported Python
# versions, so fall back to a no-op decorator when the import fails.
try:
    from typing import closed  # type: ignore[attr-defined]
except ImportError:
    _T = TypeVar("_T")

    def closed(cls: _T) -> _T:  # no-op fallback
        return cls


from xtgeo.common.types import FileLike
from xtgeo.io._file import FileFormat, FileWrapper
from xtgeo.io._tokens import (
    TokenizedLine,
    iter_noncomment_lines,
    line_matches,
    strip_surrounding_delimiters,
)


@closed
class TSurfHeaderDict(TypedDict):
    name: str


@closed
class TSurfCoordinateSystemDict(TypedDict):
    name: str
    axis_name: tuple[str, str, str]
    axis_unit: tuple[str, str, str]
    zpositive: str


@closed
class TSurfDict(TypedDict):
    """
    Dictionary representation of TSurf triangulated surface data, used for
    intermediate data handling and conversion to/from TSurfData objects.
    Note that triangles are 1-based (the first vertex has index 1)
    as required by the TSurf format.

    Args:
    - header: Dictionary with surface name (key: 'name')
    - vertices: List of vertex coordinates, each vertex is a list of [x, y, z]
    - triangles: List of triangles, each triangle is a list of vertex indices (1-based)
    - coord_sys: Optional dictionary with coordinate system data, with keys:
        - 'name': name of the coordinate system
        - 'axis_name': tuple of axis names (e.g. ("X", "Y", "Z"))
        - 'axis_unit': tuple of axis units (e.g. ("m", "m", "m"))
        - 'zpositive': direction of the Z axis, either
            - "Depth" (z-axis increasing downwards) or
            - "Elevation" (z-axis increasing upwards)
    """

    header: TSurfHeaderDict
    coord_sys: NotRequired[TSurfCoordinateSystemDict]
    vertices: npt.NDArray[np.float64]
    triangles: npt.NDArray[np.int_]


class ValidatorCoordSys:
    """
    Validator for the coordinate system in the TSurf file format.
    For each keyword there are multiple possible values.
    The reader does not (yet) recognise and handle all values
    that are valid in the TSurf format.

    Note that there is a short-list of allowed values for each keyword.
    The reader will issue an error when an invalid value is encountered,
    and it will issue a warning when an "uncommon" value is used.

    'axis_names' and 'axis_units': the whole tuple is specified, instead of
    single values that can be used for each position in the tuples.
    This is to ensure that physically meaningful relations are used.
    For example, it makes no sense to let two elements in 'axis_names' be equal.
    Or to use two different 'axis_units' laterally.

    'axis_names', 'axis_units' and 'zpositive' are case-insensitive when validating.
    """

    common_axis_names = [("x", "y", "z")]
    """('x', 'y', 'z') is the most common set of axis names"""

    common_axis_units = [("m", "m", "m"), ("ft", "ft", "ft")]
    """meters is the most common unit"""

    valid_z_positive_values = ["Depth", "Elevation"]
    """
    ZPOSITIVE = 'Depth': Z is increasing downwards
    ZPOSITIVE = 'Elevation': Z is increasing upwards
    """

    @classmethod
    def validate(cls, data: TSurfCoordinateSystemDict, fileref_errmsg: str) -> None:
        """Validate coordinate system data that has been read from a file.

        Args:
            data: Dictionary with coordinate system data
            fileref_errmsg: Error context for meaningful error messages

        Raises:
            ValueError: If coordinate system data is invalid
        """

        required_fields = ["name", "axis_name", "axis_unit", "zpositive"]
        missing_fields = [f for f in required_fields if f not in data]
        if missing_fields:
            raise ValueError(
                f"\nIn file {fileref_errmsg}:\n"
                f"Coordinate system section missing fields: {missing_fields}"
            )

        # Validate coordinate system data
        cls._validate_axis_names(data["axis_name"], fileref_errmsg)
        cls._validate_axis_units(data["axis_unit"], fileref_errmsg)
        cls._validate_zpositive(data["zpositive"], fileref_errmsg)

    @classmethod
    def _validate_axis_names(
        cls, axis_names: tuple[str, str, str], fileref_errmsg: str
    ) -> None:
        """Validate axis names tuple against known combinations.

        Args:
            axis_names: Tuple of axis names to validate
            fileref_errmsg: Error context for meaningful messages

        Raises:
            ValueError: If coordinate system data is invalid
            Warning: If uncommon axis names combination is used
        """

        cls._validate_axis_elements(
            axis_names,
            cls.common_axis_names,
            "AXIS_NAME",
            fileref_errmsg,
            check_uniqueness=True,
        )

    @classmethod
    def _validate_axis_units(
        cls, axis_units: tuple[str, str, str], fileref_errmsg: str
    ) -> None:
        """Validate axis units tuple against known combinations.

        Args:
            axis_units: Tuple of axis units to validate
            fileref_errmsg: Error context for meaningful messages

        Raises:
            ValueError: If coordinate system data is invalid
            Warning: If uncommon axis units combination is used
        """

        cls._validate_axis_elements(
            axis_units,
            cls.common_axis_units,
            "AXIS_UNIT",
            fileref_errmsg,
            check_uniqueness=False,
        )

    @classmethod
    def _validate_zpositive(cls, zpositive: str, fileref_errmsg: str) -> None:
        """Validate zpositive value and raise error for invalid values.

        Args:
            zpositive: Z-positive value to validate
            fileref_errmsg: Error context for meaningful messages

        Raises:
            ValueError: If invalid value
        """

        if zpositive.lower() not in (z.lower() for z in cls.valid_z_positive_values):
            raise ValueError(
                f"In file {fileref_errmsg}: Invalid ZPOSITIVE value '{zpositive}'. "
                f"Allowed values are: {list(cls.valid_z_positive_values)}"
            )

    @classmethod
    def _validate_axis_elements(
        cls,
        axis_elements: tuple[str, str, str],
        common_combos: list[tuple[str, str, str]],
        element_type: str,
        fileref_errmsg: str,
        check_uniqueness: bool = True,
    ) -> None:
        """Validate axis elements tuple (typically axis names and units) against common
        combinations.

        Args:
            axis_elements: Tuple of axis elements to validate
            common_combos: List of most common combinations
            element_type: Type of axis elements (for error messages)
            fileref_errmsg: Error context for meaningful messages
            check_uniqueness: Whether to check for uniqueness of elements in a tuple

        Raises:
            ValueError: If data is invalid
            Warning: If uncommon combination is used
        """

        if len(axis_elements) != 3 or any(name == "" for name in axis_elements):
            raise ValueError(
                f"In file {fileref_errmsg}: "
                f"{element_type} must have exactly three values, "
                f"found the following {len(axis_elements)} values:\n {axis_elements}"
            )

        axis_elements_lower = tuple([elm.lower() for elm in axis_elements[:3]])

        if check_uniqueness and len(set(axis_elements_lower)) != 3:
            raise ValueError(
                f"In file {fileref_errmsg}: "
                f"{element_type} values (in lowercase) must be unique, "
                f"found: {axis_elements_lower}"
            )

        common_combos_lower = []
        for t in common_combos:
            common_combos_lower.append(tuple([name.lower() for name in t]))

        # Check if the axis elements tuple is in our known common combinations
        if axis_elements_lower not in common_combos_lower:
            warnings.warn(
                f"In file {fileref_errmsg}: Uncommon {element_type} combination:\n"
                f"{axis_elements}. "
                f"More common combinations are: \n"
                f"{common_combos}",
                UserWarning,
                stacklevel=3,
            )


@dataclass(frozen=True)
class TSurfHeader:
    name: str

    @classmethod
    def validate(cls, data: TSurfHeaderDict, fileref_errmsg: str) -> None:
        """
        Validate header data that has been read from a file.

        Args:
            data: Dictionary with header data
            fileref_errmsg: Error context for meaningful error messages

        Raises:
                ValueError: If header data is invalid
        """
        err_msg = (
            f"\nIn file {fileref_errmsg}:\n"
            "Missing or invalid name in the 'HEADER' section."
        )

        if not data.get("name") or data["name"].strip() == "":
            raise ValueError(err_msg)


@dataclass(frozen=True)
class TSurfCoordSys:
    name: str
    axis_name: tuple[str, str, str]
    axis_unit: tuple[str, str, str]
    zpositive: str

    @classmethod
    def validate(cls, data: TSurfCoordinateSystemDict, fileref_errmsg: str) -> None:
        """
        Validate coordinate system data that has been read from a file.

        Args:
            data: Dictionary with coordinate system data
            fileref_errmsg: Error context for meaningful error messages

        Raises:
            ValueError: If coordinate system data is invalid
        """
        ValidatorCoordSys.validate(data, fileref_errmsg)


@dataclass(frozen=True)
class TSurfData:
    """
    Internal data class for TSurf triangulated surface data.
    Immutable to ensure data integrity during I/O operations.

    Attributes:
    - header: Dictionary with surface name (key: 'name')
    - vertices: List of vertex coordinates, each vertex is a list of [x, y, z]
        coordinates
    - triangles: List of triangles, each triangle is a list of vertex indices (1-based)
    - coord_sys: Optional dictionary with coordinate system data, with keys:
        - 'name': name of the coordinate system
        - 'axis_name': tuple of axis names (e.g. ("X", "Y", "Z"))
        - 'axis_unit': tuple of axis units (e.g. ("m", "m", "m"))
        - 'zpositive': direction of the Z axis, either
            - "Depth" (z-axis increasing downwards) or
            - "Elevation" (z-axis increasing upwards)

    Raises:
        ValueError: If input data is invalid
    """

    header: TSurfHeader
    coord_sys: TSurfCoordSys | None
    vertices: npt.NDArray[np.float64]
    triangles: npt.NDArray[np.int_]

    def __post_init__(self) -> None:
        # Need at least 3 vertices to form a triangle
        if self.vertices.size == 0 or self.vertices.shape[0] < 3:
            raise ValueError("Less than 3 vertices found in TSurf triangulation data.")

        # Need at least one triangle to form a surface
        if self.triangles.size == 0:
            raise ValueError("No triangles found in TSurf triangulation data.")

        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise ValueError(
                f"vertices must have shape (N, 3), got {self.vertices.shape}"
            )
        if self.triangles.ndim != 2 or self.triangles.shape[1] != 3:
            raise ValueError(
                f"triangles must have shape (M, 3), got {self.triangles.shape}"
            )

        # Triangle indices must be 1-based and within [1, num_vertices]
        if np.any(self.triangles < 1):
            raise ValueError(
                "Triangle vertex indices must be >= 1 in triangulation data."
            )
        if np.any(self.triangles > self.vertices.shape[0]):
            raise ValueError(
                "Triangle vertex indices must be <= number of vertices in "
                "triangulation data."
            )

        # Lock the arrays so the dataclass is truly immutable (not just at the
        # attribute level). Frozen dataclasses normally only prevent rebinding
        # of attributes; the underlying numpy buffers are still writable.
        self.vertices.flags.writeable = False
        self.triangles.flags.writeable = False

    @property
    def num_vertices(self) -> int:
        """Return the number of vertices in the triangulated surface."""
        return self.vertices.shape[0]

    @property
    def num_triangles(self) -> int:
        """Return the number of triangles in the triangulated surface."""
        return self.triangles.shape[0]

    @staticmethod
    def _parse_header_section(
        lines: Iterator[TokenizedLine], fileref_errmsg: str
    ) -> str:
        """Parse the HEADER section and extract the surface name.

        Args:
            lines: Tokenized text lines to read from
            fileref_errmsg: Error context for meaningful error messages

        Returns:
            str: The surface name from the header

        Raises:
            ValueError: If header section is malformed or missing name
        """

        # Expected format of the HEADER section
        err_msg = (
            f"\nIn file {fileref_errmsg}:\n"
            "The 'HEADER' section is mandatory and has the following format:\n"
            "HEADER {\n"
            "name: <surface_name>\n"
            "}\n"
        )

        header_name = []

        # Is '}' present to end the HEADER section?
        end_is_present = False

        for line in lines:
            # Assume it can be either ['name:F5'] or ['name:', 'F5']
            # Assume the name itself may be split into several strings:
            # e.g. 'name:Massive listric fault' -> ['Massive', 'listric', 'fault']
            # Assume line[0] is lowercase
            if line[0].startswith("name:"):
                # Extract name after the colon and join with remaining parts
                tmp = [line[0][5:]] + line[1:]
                header_name = [item.strip() for item in tmp if item.strip() != ""]
            elif line_matches(line, "}"):
                end_is_present = True
                break
            else:
                raise ValueError(err_msg + f"Invalid 'HEADER' section line:\n'{line}'")

        if not end_is_present:
            raise ValueError(
                err_msg + "Missing '}' at the end of the 'HEADER' section."
            )

        return " ".join(header_name)

    @staticmethod
    def _parse_coordinate_system_section(
        lines: Iterator[TokenizedLine], fileref_errmsg: str
    ) -> TSurfCoordinateSystemDict:
        """Parse the coordinate system section.

        Args:
            lines: Tokenized text lines to read from
            fileref_errmsg: Error context for meaningful error messages

        Returns:
            TSurfCoordinateSystemDict: Coordinate system data with keys:
            name, axis_name, axis_unit, zpositive

        Raises:
            ValueError: If coordinate system section is malformed or incomplete
        """

        # Expected format
        err_msg = (
            f"\nIn file {fileref_errmsg}:\n"
            "The 'COORDINATE_SYSTEM' section is optional "
            "and has the following format:\n"
            "GOCAD_ORIGINAL_COORDINATE_SYSTEM\n"
            "NAME <name>\n"
            'AXIS_NAME "X" "Y" "Z"\n'
            'AXIS_UNIT "m" "m" "m"\n'
            "ZPOSITIVE Depth\n"
            "END_ORIGINAL_COORDINATE_SYSTEM\n\n"
            "where:\n"
            "AXIS_NAME: names of the X, Y, and Z axes\n"
            "AXIS_UNIT: units of the X, Y, and Z axes\n"
            "ZPOSITIVE: direction of the Z axis, 'Depth' or 'Elevation'\n"
            "(meaning positive direction 'down' or 'up', respectively)\n"
        )

        coord_sys_data: dict[str, Any] = {}

        # Is 'END_ORIGINAL_COORDINATE_SYSTEM' statement present?
        end_is_present = False

        # Parse coordinate system attributes
        for line in lines:
            if line[0] == "NAME":
                coord_sys_data["name"] = " ".join(line[1:])
            elif line[0] == "AXIS_NAME":
                # Extract axis names and remove quotes
                coord_sys_data["axis_name"] = tuple(
                    strip_surrounding_delimiters(y, '"') for y in line[1:4]
                )
            elif line[0] == "AXIS_UNIT":
                # Extract axis units and remove quotes
                coord_sys_data["axis_unit"] = tuple(
                    strip_surrounding_delimiters(y, '"') for y in line[1:4]
                )
            elif line[0] == "ZPOSITIVE":
                # Extract zpositive value
                coord_sys_data["zpositive"] = line[1]
            elif line_matches(line, "END_ORIGINAL_COORDINATE_SYSTEM"):
                end_is_present = True
                break
            else:
                err_msg += f"Invalid line in 'COORDINATE_SYSTEM' section:\n'{line}'"
                raise ValueError(err_msg)

        if not end_is_present:
            err_msg += "Missing 'END_ORIGINAL_COORDINATE_SYSTEM' statement"
            raise ValueError(err_msg)

        ValidatorCoordSys.validate(
            cast("TSurfCoordinateSystemDict", coord_sys_data), fileref_errmsg
        )

        return TSurfCoordinateSystemDict(
            name=coord_sys_data["name"],
            axis_name=coord_sys_data["axis_name"],
            axis_unit=coord_sys_data["axis_unit"],
            zpositive=coord_sys_data["zpositive"],
        )

    @staticmethod
    def _parse_tface_section(
        lines: Iterator[TokenizedLine], fileref_errmsg: str
    ) -> tuple[list[list[float]], list[list[int]]]:
        """Parse the TFACE section with data defining a triangulated surface
        (vertices and triangles).

        Args:
            lines: Tokenized text lines to read from
            fileref_errmsg: Error context for meaningful error messages

        Returns:
            tuple: (vertices, triangles) where 'vertices' is a list of
            [x,y,z] coordinates and 'triangles' is a list of vertex indices
            (starting with 1)

        Raises:
            ValueError: If TFACE section contains invalid lines
        """

        vertices: list[list[float]] = []
        triangles: list[list[int]] = []

        err_msg_vrtx = (
            f"\nIn file {fileref_errmsg}:\n"
            "Invalid 'VRTX' line in 'TFACE' section.\n"
            "Expected format: 'VRTX id x y z [attributes]'\n"
            "where:\n"
            "  - 'id' starts with '1' and increments by 1 for each vertex\n"
            "  - 'x', 'y', and 'z' are floating-point numbers\n"
            "  - the only currently supported attribute is 'CNXYZ'."
        )

        err_msg_trgl = (
            f"\nIn file {fileref_errmsg}:\n"
            "Invalid 'TRGL' line in 'TFACE' section.\n"
            "Expected format: 'TRGL vertex_id1 vertex_id2 vertex_id3'\n"
            "where:\n"
            "  - each vertex id must be an integer\n"
            "  - each vertex id must be in the range "
            "{{1, number of vertices}} (1-based indexing)\n"
        )

        # Keep track of expected vertex numbering, 1-based numbering
        vrtx_no = 0

        # Is 'END' statement present to end the TFACE section?
        end_is_present = False

        for line in lines:
            if line[0] == "VRTX":
                if len(line) != 6 or line[1] != str(vrtx_no + 1) or line[5] != "CNXYZ":
                    err_msg_vrtx += f"Failing line: '{line}'"
                    raise ValueError(err_msg_vrtx)

                try:
                    vertices.append([float(line[2]), float(line[3]), float(line[4])])
                except ValueError:
                    err_msg_vrtx += f"Failing line: '{line}'"
                    raise ValueError(err_msg_vrtx)

                vrtx_no += 1

            elif line[0] == "TRGL":
                if len(line) != 4:
                    err_msg_trgl += f"Failing line: '{line}'"
                    raise ValueError(err_msg_trgl)

                try:
                    triangles.append([int(line[1]), int(line[2]), int(line[3])])
                except ValueError:
                    err_msg_trgl += f"Failing line: '{line}'"
                    raise ValueError(err_msg_trgl)

            elif line[0] == "END":
                end_is_present = True
                break

            else:
                raise ValueError(
                    f"\nIn file {fileref_errmsg}:\n"
                    "Invalid line in 'TFACE' section with triangulated data.\n"
                    "Expect lines to start with 'VRTX', 'TRGL', or 'END'.\n"
                    f"Failing line: '{line}'",
                )

        if not end_is_present:
            raise ValueError(
                f"\nIn file {fileref_errmsg}:\n"
                "Missing 'END' statement at the end of the 'TFACE' section."
            )

        return vertices, triangles

    @classmethod
    def _create_tsurf_data(
        cls,
        header_name: str,
        coord_sys_data: TSurfCoordinateSystemDict | None,
        vertices: list[list[float]],
        triangles: list[list[int]],
    ) -> Self:
        """
        Create TSurfData object from parsed components.

        Args:
            header_name: Surface name from header
            coord_sys_data: Optional dictionary with coordinate system data
            vertices: List of vertex coordinates
            triangles: List of vertex indices

        Returns:
            TSurfData: Complete TSurf data object
        """

        header = TSurfHeader(name=header_name)

        coord_sys = None
        if coord_sys_data is not None:
            coord_sys = TSurfCoordSys(
                name=coord_sys_data["name"],
                axis_name=coord_sys_data["axis_name"],
                axis_unit=coord_sys_data["axis_unit"],
                zpositive=coord_sys_data["zpositive"],
            )

        vertices_array = np.array(vertices, dtype=np.float64)
        triangles_array = np.array(triangles, dtype=np.int64)

        return cls(
            header=header,
            coord_sys=coord_sys,
            vertices=vertices_array,
            triangles=triangles_array,
        )

    @classmethod
    def _parse_tsurf(cls, raw_lines: Iterable[str], fileref_errmsg: str) -> Self:
        """
        Parse a TSurf file from raw text lines and create a TSurfData object.

        Args:
            raw_lines: Text lines from the TSurf file.
            fileref_errmsg: Error context for meaningful error messages

        Note:
        - The TSurf format has many more sections and attributes than
            those currently captured in this parser
        - The first line is skipped, already checked to be
            the TSurf signature line: 'GOCAD TSurf 1'
        - While we are not aware of whether the ordering of the other
            sections (HEADER, coordinate system, TFACE) is important,
            the ordering is not strictly enforced in this parser
        - Only one section of each type is allowed
        """

        header_name: str = ""
        coord_sys_data: TSurfCoordinateSystemDict | None = None
        vertices: list[list[float]] = []
        triangles: list[list[int]] = []

        # Keep track of which sections that have been read and validated
        header_section_completed = False
        coord_sys_section_completed = False
        tface_section_completed = False

        token_lines = iter_noncomment_lines(raw_lines, ["#"])

        # Skip the already verified TSurf signature line
        next(token_lines)

        # Loop over sections, each section starts with a specific keyword.
        # For each section there is a parsing function
        # which reads lines until the end of the section.
        for line in token_lines:
            # HEADER section (mandatory)
            if line_matches(line, "HEADER {"):
                # Ensure only one section of this type
                if header_section_completed:
                    raise ValueError(
                        f"\nIn file {fileref_errmsg}:\n"
                        "Multiple 'HEADER' sections found, "
                        "but only one is allowed."
                    )
                header_section_completed = True

                header_name = TSurfData._parse_header_section(
                    token_lines, fileref_errmsg
                )
                header_dict = TSurfHeaderDict({"name": header_name})
                TSurfHeader.validate(header_dict, fileref_errmsg)
                continue

            # COORDINATE_SYSTEM section (optional)
            if line_matches(line, "GOCAD_ORIGINAL_COORDINATE_SYSTEM"):
                # Ensure only one section of this type
                if coord_sys_section_completed:
                    raise ValueError(
                        f"\nIn file {fileref_errmsg}:\n"
                        "Multiple 'COORDINATE_SYSTEM' sections found, "
                        "but only one is allowed."
                    )
                coord_sys_section_completed = True

                coord_sys_data = TSurfData._parse_coordinate_system_section(
                    token_lines, fileref_errmsg
                )
                continue

            # TFACE section with triangulated surface data (mandatory)
            if line_matches(line, "TFACE"):
                # Ensure only one section of this type
                if tface_section_completed:
                    raise ValueError(
                        f"\nIn file {fileref_errmsg}:\n"
                        "Multiple 'TFACE' sections found, "
                        "but only one is allowed."
                    )
                tface_section_completed = True

                vertices, triangles = TSurfData._parse_tface_section(
                    token_lines, fileref_errmsg
                )
                continue

            # -----------------------
            # Handle unknown keywords
            # -----------------------
            # The TSurf file format handles many more keywords and attributes
            # than those captured in this parser. Could potentially issue a warning
            # instead of an error and continue. But if the section
            # continues over several lines, it is difficult to know where it
            # ends and where parsing can resume.

            raise ValueError(
                f"\nIn file {fileref_errmsg}:\n"
                "The file contains an invalid line which is not recognized \n"
                "as a section identifier (first line of a section).\n"
                "This may be either an error, or a valid TSurf section identifier\n"
                "that is not (yet) handled by the file parser.\n"
                f"Failing line: '{line}'",
            )

        if not header_section_completed:
            raise ValueError(
                f"\nIn file {fileref_errmsg}:\nMissing mandatory 'HEADER' section.\n"
            )
        if not tface_section_completed:
            raise ValueError(
                f"\nIn file {fileref_errmsg}:\nMissing mandatory 'TFACE' section.\n"
            )

        return cls._create_tsurf_data(header_name, coord_sys_data, vertices, triangles)

    def to_dict(self) -> TSurfDict:
        """
        Returns a deep, recursive copy of the TSurfData as a dictionary.

        COORDINATE_SYSTEM is optional in the TSurf file specification,
        thus 'coord_sys' is not required and is only included in the output dictionary
        if it was present in the original TSurf data/file.
        """

        data = asdict(self)

        result = TSurfDict(
            header=data["header"],
            vertices=data["vertices"],
            triangles=data["triangles"],
        )
        if data.get("coord_sys") is not None:
            result["coord_sys"] = cast("TSurfCoordinateSystemDict", data["coord_sys"])
        return result

    @classmethod
    def from_dict(cls, data: TSurfDict) -> Self:
        """
        Create a TSurfData object from a TSurfDict dictionary.

        Args:
            data: TSurfDict dictionary containing triangulated surface data
        Returns:
            TSurfData: TSurfData object created from the input dictionary
        """

        header_dict = data["header"]
        TSurfHeader.validate(header_dict, fileref_errmsg="input dictionary")
        header = TSurfHeader(name=header_dict["name"])

        coord_sys = None
        if "coord_sys" in data and data["coord_sys"] is not None:
            coord_sys_dict = data["coord_sys"]
            TSurfCoordSys.validate(coord_sys_dict, fileref_errmsg="input dictionary")
            coord_sys = TSurfCoordSys(
                name=coord_sys_dict["name"],
                axis_name=coord_sys_dict["axis_name"],
                axis_unit=coord_sys_dict["axis_unit"],
                zpositive=coord_sys_dict["zpositive"],
            )

        return cls(
            header=header,
            coord_sys=coord_sys,
            vertices=np.array(data["vertices"], dtype=np.float64),
            triangles=np.array(data["triangles"], dtype=np.int64),
        )

    @classmethod
    def from_file(
        cls,
        file: FileLike,
        encoding: str = "utf-8",
    ) -> Self:
        """
        Read a file on the TSURF format and parse its triangulated surface data.
        Note that only a subset of the TSurf format is currently supported,
        more types of sections and keywords exist in the format specification.

        Args:
            file: Path to TSurf file (str or Path) or
                a file-like object (BytesIO or StringIO).
            encoding: Text encoding for the input file (default: 'utf-8')

        Returns:
            TSurfData: Parsed surface data containing header, coordinate system,
                vertices and triangles

        Raises:
            FileNotFoundError: If file path doesn't exist or isn't a regular file
            ValueError: If file format is invalid

        Note:
            - TSurf is a file format used in for example the GOCAD software.
            - RMS can export triangulated surfaces in its
            structural model in the TSurf format.
            - When unhandled keywords are present in the file, the processing is halted
            and an error message is issued. The user is expected to take action
            to fix the file.
            - Documentation for the TSurf format is limited if you don't
            have access to the  GOCAD Developer's Guide, but
            here is one: https://paulbourke.net/dataformats/gocad/gocad.pdf
        """

        wrapped_file = FileWrapper(file)

        if not wrapped_file.check_file():
            raise FileNotFoundError(
                f"\nIn file {wrapped_file.name}:\nThe file does not exist."
            )
        wrapped_file.fileformat(FileFormat.TSURF.value[0], strict=True)

        with wrapped_file.get_text_stream_read(encoding=encoding) as stream:
            return cls._parse_tsurf(stream, fileref_errmsg=str(wrapped_file.name))

    def to_file(
        self: Self,
        file: FileLike,
        encoding: str = "utf-8",
    ) -> None:
        """
        Write TSurfData to a file stream in the TSurf format.

        Args:
            data: TSurfData object containing triangulated surface data
            file: Path to output TSurf file (str or Path) or
                a file-like object (BytesIO or StringIO).
            encoding: Text encoding for the output file (default: 'utf-8')

        Raises:
            FileNotFoundError: If file path doesn't exist or isn't a regular file
        """

        wrapped_file = FileWrapper(file)

        with wrapped_file.get_text_stream_write(encoding=encoding) as stream:
            # TSurf signature line
            stream.write("GOCAD TSurf 1\n")

            # HEADER section
            stream.write("HEADER {\n")
            stream.write(f"name: {self.header.name}\n")
            stream.write("}\n")

            # Optional: coordinate system
            if self.coord_sys:
                stream.write("GOCAD_ORIGINAL_COORDINATE_SYSTEM\n")
                stream.write(f"NAME {self.coord_sys.name}\n")

                axis_name_str = " ".join(
                    [f'"{name}"' for name in self.coord_sys.axis_name]
                )
                stream.write(f"AXIS_NAME {axis_name_str}\n")

                axis_unit_str = " ".join(
                    [f'"{unit}"' for unit in self.coord_sys.axis_unit]
                )
                stream.write(f"AXIS_UNIT {axis_unit_str}\n")

                stream.write(f"ZPOSITIVE {self.coord_sys.zpositive}\n")
                stream.write("END_ORIGINAL_COORDINATE_SYSTEM\n")

            # TFACE section with vertices and triangles
            stream.write("TFACE\n")
            for i, vertex in enumerate(self.vertices, start=1):
                stream.write(f"VRTX {i} {vertex[0]} {vertex[1]} {vertex[2]} CNXYZ\n")
            for triangle in self.triangles:
                stream.write(f"TRGL {triangle[0]} {triangle[1]} {triangle[2]}\n")
            stream.write("END\n")
