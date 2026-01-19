import warnings
from dataclasses import dataclass
from typing import Any, Generator, TextIO

import numpy as np
import numpy.typing as npt
from typing_extensions import Self

from xtgeo.common.types import FileLike
from xtgeo.io._file import FileFormat, FileWrapper


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
    def validate(cls, data: dict[str, Any], fileref_errmsg: str) -> None:
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
        cls._validate_axis_names(tuple(data["axis_name"]), fileref_errmsg)
        cls._validate_axis_units(tuple(data["axis_unit"]), fileref_errmsg)
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
    def validate(cls, data: dict[str, Any], fileref_errmsg: str) -> None:
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
    axis_name: list[str]
    axis_unit: list[str]
    zpositive: str

    @classmethod
    def validate(cls, data: dict[str, Any], fileref_errmsg: str) -> None:
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
        header: TSurfHeader with surface name
        coord_sys: Optional TSurfCoordSys with coordinate system data
        vertices: numpy array of vertex coordinates (n x 3)
        triangles: numpy array of triangle vertex indices (m x 3)

    Raises:
        ValueError: If input data is invalid
    """

    header: TSurfHeader
    coord_sys: TSurfCoordSys | None
    vertices: npt.NDArray[np.float64]
    triangles: npt.NDArray[np.int_]

    @property
    def get_vertices(self) -> npt.NDArray[np.float64]:
        """Return the vertices of the triangulated surface."""
        return self.vertices

    @property
    def get_cells(self) -> npt.NDArray[np.int_]:
        """Return the triangles of the triangulated surface."""
        return self.triangles

    @staticmethod
    def _read_line(
        stream: TextIO,
    ) -> Generator[list[str], None, None]:
        """
        Iterate over lines from a TextIO, yielding lists of strings.
        Filters out empty lines and comment lines.
        """
        for line in stream:
            split_line = line.strip().split()

            if not split_line:
                continue

            # Skip comments
            if split_line[0].startswith("#"):
                continue

            yield split_line

    @staticmethod
    def _is_header_section_first_line(line: list[str]) -> bool:
        """Check if the line is "HEADER {", to indicate the start of a HEADER section.

        Args:
            line: Line tokens from the file

        Returns:
            bool: True if the line indicates the start of a HEADER section,
                False otherwise
        """

        return len(line) == 2 and line[0] == "HEADER" and line[1] == "{"

    @staticmethod
    def _is_coordinate_system_section_first_line(line: list[str]) -> bool:
        """Check if the line is "GOCAD_ORIGINAL_COORDINATE_SYSTEM",
        which indicates the start of a COORDINATE_SYSTEM section.

        Args:
            line: Line tokens from the file

        Returns:
            bool: True if the line indicates the start of a COORDINATE_SYSTEM section,
                False otherwise
        """

        return len(line) == 1 and line[0] == "GOCAD_ORIGINAL_COORDINATE_SYSTEM"

    @staticmethod
    def _is_tface_section_first_line(line: list[str]) -> bool:
        """Check if the line is "TFACE", which indicates the start of a TFACE section.

        Args:
            line: Line tokens from the file

        Returns:
            bool: True if the line indicates the start of a TFACE section,
                False otherwise
        """

        return len(line) == 1 and line[0] == "TFACE"

    @staticmethod
    def _parse_header_section(stream: TextIO, fileref_errmsg: str) -> str:
        """Parse the HEADER section and extract the surface name.

        Args:
            stream: Text stream to read from
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

        for line in TSurfData._read_line(stream):
            # Assume it can be either ['name:F5'] or ['name:', 'F5']
            # Assume the name itself may be split into several strings:
            # e.g. 'name:Massive listric fault' -> ['Massive', 'listric', 'fault']
            # Assume line[0] is lowercase
            if line[0].startswith("name:"):
                # Extract name after the colon and join with remaining parts
                tmp = [line[0][5:]] + line[1:]
                header_name = [item.strip() for item in tmp if item.strip() != ""]
            elif line[0] == "}":
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
        stream: TextIO, fileref_errmsg: str
    ) -> dict[str, Any]:
        """Parse the coordinate system section.

        Args:
            stream: Text stream to read from
            fileref_errmsg: Error context for meaningful error messages

        Returns:
            dict: Coordinate system data with keys:
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
        for line in TSurfData._read_line(stream):
            if line[0] == "NAME":
                coord_sys_data["name"] = " ".join(line[1:])
            elif line[0] == "AXIS_NAME":
                # Extract axis names and remove quotes
                coord_sys_data["axis_name"] = [y.strip('"') for y in line[1:4]]
            elif line[0] == "AXIS_UNIT":
                # Extract axis units and remove quotes
                coord_sys_data["axis_unit"] = [y.strip('"') for y in line[1:4]]
            elif line[0] == "ZPOSITIVE":
                # Extract zpositive value
                coord_sys_data["zpositive"] = line[1]
            elif line[0] == "END_ORIGINAL_COORDINATE_SYSTEM":
                end_is_present = True
                break
            else:
                err_msg += f"Invalid line in 'COORDINATE_SYSTEM' section:\n'{line}'"
                raise ValueError(err_msg)

        if not end_is_present:
            err_msg += "Missing 'END_ORIGINAL_COORDINATE_SYSTEM' statement"
            raise ValueError(err_msg)

        return coord_sys_data

    @staticmethod
    def _parse_tface_section(
        stream: TextIO, fileref_errmsg: str
    ) -> tuple[list[list[float]], list[list[int]]]:
        """Parse the TFACE section with data defining a triangulated surface
        (vertices and triangles).

        Args:
            stream: Text stream to read from
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

        for line in TSurfData._read_line(stream):
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

    @staticmethod
    def _validate_triangulation_data(
        vertices: np.ndarray, triangles: np.ndarray
    ) -> None:
        """Basic validation of the triangulation data in the TSurfData object.

        Args:
            vertices: Array of vertex coordinates
            triangles: Array of triangle vertex indices
        """

        # Need at least 3 vertices to form a triangle
        if vertices.shape[0] < 3:
            raise ValueError("Less than 3 vertices found in TSurf triangulation data.")

        # Need at least one triangle to form a surface
        if triangles.size == 0:
            raise ValueError("No triangles found in TSurf triangulation data.")

        # Check for valid vertex indices in triangles
        # Must be in the closed range [1, number of vertices]
        if np.any(triangles < 1):
            raise ValueError(
                "Triangle vertex indices must be >= 1 in triangulation data."
            )

        if np.any(triangles > len(vertices)):
            raise ValueError(
                "Triangle vertex indices must be <= number of vertices in "
                "triangulation data."
            )

    @classmethod
    def _create_tsurf_data(
        cls,
        header_name: str,
        coord_sys_data: dict[str, Any],
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
        if coord_sys_data:
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
    def _parse_tsurf(cls, stream: TextIO, fileref_errmsg: str) -> Self:
        """
        Parse a TSurf file from a file stream.

        Args:
            stream: A stream of the TSurf file.
            fileref_errmsg: Error context for meaningful error messages

        Note:
        - The TSurf format has many more sections and attributes than
            those currently captured in this parser
        - The first line must be the TSurf signature line: 'GOCAD TSurf 1'
        - While we are not aware of whether the ordering of the other
            sections (HEADER, coordinate system, TFACE) is important,
            the ordering is not strictly enforced in this parser
        - Only one section of each type is allowed
        """

        header_name: str = ""
        coord_sys_data: dict[str, Any] = {}
        vertices: list[list[float]] = []
        triangles: list[list[int]] = []

        # Keep track of which sections that have been read and validated
        header_section_completed = False
        coord_sys_section_completed = False
        tface_section_completed = False

        # Skip the already verified TSurf signature line
        next(TSurfData._read_line(stream))

        # Loop over sections, each section starts with a specific keyword.
        # For each section there is a parsing function
        # which reads lines until the end of the section.
        for line in TSurfData._read_line(stream):
            if TSurfData._is_header_section_first_line(line):
                # Ensure only one section of this type
                if header_section_completed:
                    raise ValueError(
                        f"\nIn file {fileref_errmsg}:\n"
                        "Multiple 'HEADER' sections found, "
                        "but only one is allowed."
                    )
                header_section_completed = True

                header_name = TSurfData._parse_header_section(stream, fileref_errmsg)
                header_dict = {"name": header_name}
                TSurfHeader.validate(header_dict, fileref_errmsg)
                continue

            if TSurfData._is_coordinate_system_section_first_line(line):
                # Ensure only one section of this type
                if coord_sys_section_completed:
                    raise ValueError(
                        f"\nIn file {fileref_errmsg}:\n"
                        "Multiple 'COORDINATE_SYSTEM' sections found, "
                        "but only one is allowed."
                    )
                coord_sys_section_completed = True

                coord_sys_data = TSurfData._parse_coordinate_system_section(
                    stream, fileref_errmsg
                )
                TSurfCoordSys.validate(coord_sys_data, fileref_errmsg)
                continue

            if TSurfData._is_tface_section_first_line(line):
                # Ensure only one section of this type
                if tface_section_completed:
                    raise ValueError(
                        f"\nIn file {fileref_errmsg}:\n"
                        "Multiple 'TFACE' sections found, "
                        "but only one is allowed."
                    )
                tface_section_completed = True

                vertices, triangles = TSurfData._parse_tface_section(
                    stream, fileref_errmsg
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

        tsurf_data = cls._create_tsurf_data(
            header_name, coord_sys_data, vertices, triangles
        )
        cls._validate_triangulation_data(tsurf_data.vertices, tsurf_data.triangles)
        return tsurf_data

    @classmethod
    def from_file(
        cls,
        file: FileLike,
    ) -> Self:
        """
        Read a file on the TSURF format and parse its triangulated surface data.
        Note that only a subset of the TSurf format is currently supported,
        more types of sections and keywords exist in the format specification.

        Args:
            file: Path to TSurf file (str or Path) or
                a file-like object (BytesIO or StringIO).

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

        with wrapped_file.get_text_stream() as stream:
            return cls._parse_tsurf(stream, fileref_errmsg=str(wrapped_file.name))

    def to_file(
        self: Self,
        file: FileLike,
    ) -> None:
        """
        Write TSurfData to a file stream in the TSurf format.

        Args:
            data: TSurfData object containing triangulated surface data
            file: Path to output TSurf file (str or Path) or
                a file-like object (BytesIO or StringIO).

        Raises:
            FileNotFoundError: If file path doesn't exist or isn't a regular file
        """

        raise NotImplementedError("TSurfData.to_file() is not yet implemented.")
