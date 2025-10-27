import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from io import BytesIO, StringIO, TextIOWrapper
from pathlib import Path
from typing import Any, Generator, Optional, Union

import numpy as np

from xtgeo.common.types import FileLike


class ValidatorCoordSys:
    """
    Validator for the coordinate system in the TSurf file format.
    For each keyword there are multiple possible values.
    The reader does not (yet) recognise and handle all values
    that are valid in the TSurf format.

    Note that for some of the keywords there is a short-list of allowed values.
    The reader will issue an error when an invalid value is encountered.
    For other keywords the reader will issue a warning
    when an "uncommon" value is used.
    """

    # axis_names and axis_units: the whole tuple is specified, instead of
    # single values that can be used for each position in the tuples.
    # This is to ensure that physically meaningful relations are used.
    # For example, it makes no sense to let two elements in axis_names be equal.
    # Or to use two different axis_units laterally.

    axis_names = [("X", "Y", "Z")]
    """XYZ are the most common axis names"""

    axis_units = [("m", "m", "m"), ("ft", "ft", "ft")]
    """meters is the most common unit"""

    z_positives = ["Depth", "Elevation"]
    """
    ZPOSITIVE = 'Depth': Z is increasing downwards
    ZPOSITIVE = 'Elevation': Z is increasing upwards
    """

    @classmethod
    def validate_coord_sys_data(
        cls, coord_sys_data: dict[str, Any], filepath_errmsg: str
    ) -> None:
        """Validate the coordinate system data.

        Args:
            coord_sys_data: Dictionary with coordinate system data
            filepath_errmsg: Error context for meaningful error messages

        Raises:
            ValueError: If coordinate system data is invalid
        """

        required_fields = ["name", "axis_name", "axis_unit", "zpositive"]
        missing_fields = [f for f in required_fields if f not in coord_sys_data]
        if missing_fields:
            raise ValueError(
                f"\nIn file {filepath_errmsg}:\n"
                f"Coordinate system section missing fields: {missing_fields}"
            )

        # Validate coordinate system data
        cls._validate_axis_names(tuple(coord_sys_data["axis_name"]), filepath_errmsg)

        cls._validate_axis_units(tuple(coord_sys_data["axis_unit"]), filepath_errmsg)

        cls._validate_zpositive(coord_sys_data["zpositive"], filepath_errmsg)

    @classmethod
    def _validate_axis_names(
        cls, axis_names: tuple[str, str, str], filepath_errmsg: str
    ) -> None:
        """Validate axis names tuple against known combinations.

        Args:
            axis_names: Tuple of axis names to validate
            filepath_errmsg: Error context for meaningful messages
        """

        if (
            len(axis_names) != 3
            or axis_names[0] == ""
            or axis_names[1] == ""
            or axis_names[2] == ""
        ):
            raise ValueError(
                f"In file {filepath_errmsg}: "
                "AXIS_NAME must have exactly three values, "
                f"found the following {len(axis_names)}:\n {axis_names}"
            )

        # Check if the axis names tuple is in our known common combinations
        if axis_names not in cls.axis_names:
            warnings.warn(
                f"In file {filepath_errmsg}: Uncommon AXIS_NAME combination: "
                f"{axis_names}. More common combinations are: "
                f"{list(cls.axis_names)}",
                UserWarning,
                stacklevel=3,
            )

    @classmethod
    def _validate_axis_units(
        cls, axis_units: tuple[str, str, str], filepath_errmsg: str
    ) -> None:
        """Validate axis units tuple against known combinations.

        Args:
            axis_units: Tuple of axis units to validate
            filepath_errmsg: Error context for meaningful messages
        """

        if len(axis_units) != 3 or any(unit == "" for unit in axis_units):
            raise ValueError(
                f"In file {filepath_errmsg}: AXIS_UNIT must have exactly three values, "
                f"found {len(axis_units)}: {axis_units}"
            )

        # Check if the axis units are in our known common values
        if axis_units not in cls.axis_units:
            warnings.warn(
                f"In file {filepath_errmsg}: Uncommon AXIS_UNIT combination: "
                f"{axis_units}. More common combinations are: "
                f"{list(cls.axis_units)}",
                UserWarning,
                stacklevel=3,
            )

    @classmethod
    def _validate_zpositive(cls, zpositive: str, filepath_errmsg: str) -> None:
        """Validate zpositive value and raise error for invalid values.

        Args:
            zpositive: Z-positive value to validate
            filepath_errmsg: Error context for meaningful messages

        Raises:
            ValueError: If zpositive value is not in allowed values
        """

        if zpositive.lower() not in (z.lower() for z in cls.z_positives):
            raise ValueError(
                f"In file {filepath_errmsg}: Invalid ZPOSITIVE value '{zpositive}'. "
                f"Allowed values are: {list(cls.z_positives)}"
            )


@dataclass
class TSurfHeader:
    name: str


@dataclass
class TSurfCoordSys:
    name: str
    axis_name: list[str]
    axis_unit: list[str]
    zpositive: str


@dataclass
class TSurfData:
    header: TSurfHeader
    coord_sys: Optional[TSurfCoordSys]
    vertices: np.ndarray
    triangles: np.ndarray


def _read_line(
    stream: Union[TextIOWrapper, StringIO],
) -> Generator[list[str], Any, None]:
    """
    Iterate over lines from a TextIOWrapper or StringIO,
    yielding lists of strings.

    Filters out empty lines and comment lines.
    """
    for line in stream:
        split_line = line.strip().split()

        if not split_line or all(s == "" for s in split_line):
            continue

        # Skip comments
        if split_line[0].startswith("#"):
            continue

        yield split_line


def _is_tsurf_signature(line: list[str]) -> bool:
    """Check if the line is the TSurf signature line, which should be the
    first line in a TSurf file.

    Args:
        line: Line tokens from the file

    Returns:
        bool: True if the line is the TSurf signature line, False otherwise
    """

    # First line should be exactly "GOCAD TSurf 1"
    return " ".join(line) == "GOCAD TSurf 1"


def _is_header_section_first_line(line: list[str]) -> bool:
    """Check if the line indicates the start of a HEADER section.

    Args:
        line: Line tokens from the file

    Returns:
        bool: True if the line indicates the start of a HEADER section,
              False otherwise
    """

    # First line should be exactly "HEADER {"
    return len(line) == 2 and line[0] == "HEADER" and line[1] == "{"


def _is_coordinate_system_section_first_line(line: list[str]) -> bool:
    """Check if the line indicates the start of a COORDINATE_SYSTEM section.

    Args:
        line: Line tokens from the file

    Returns:
        bool: True if the line indicates the start of a COORDINATE_SYSTEM section,
              False otherwise
    """

    return len(line) == 1 and line[0] == "GOCAD_ORIGINAL_COORDINATE_SYSTEM"


def _is_tface_section_first_line(line: list[str]) -> bool:
    """Check if the line indicates the start of a TFACE section.

    Args:
        line: Line tokens from the file

    Returns:
        bool: True if the line indicates the start of a TFACE section,
              False otherwise
    """

    return len(line) == 1 and line[0] == "TFACE"


def _parse_header_section(
    stream: Union[TextIOWrapper, StringIO], filepath_errmsg: str
) -> str:
    """Parse the HEADER section and extract the surface name.

    Args:
        stream: Text stream to read from
        filepath_errmsg: Error context for meaningful error messages

    Returns:
        str: The surface name from the header

    Raises:
        ValueError: If header section is malformed or missing name
    """

    # Expected format of the HEADER section
    err_msg = (
        f"\nIn file {filepath_errmsg}:\n"
        "The 'HEADER' section is mandatory and has the following format:\n"
        "HEADER {\n"
        "name: <surface_name>\n"
        "}\n"
    )

    header_name = []

    end_is_present = False

    for line in _read_line(stream):
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
        raise ValueError(err_msg + "Missing '}' at the end of the 'HEADER' section.")

    return " ".join(header_name)


def _validate_header_section(header_name: str, filepath_errmsg: str) -> None:
    """Validate the HEADER section, i.e. the name.

    Args:
        header_name: The name extracted from the header section
        filepath_errmsg: Error context for meaningful error messages

    Raises:
        ValueError: If header section name is invalid
    """

    err_msg = (
        f"\nIn file {filepath_errmsg}:\n"
        "Missing or invalid name in the 'HEADER' section."
    )

    if not header_name or header_name == "":
        raise ValueError(err_msg)


def _parse_coordinate_system_section(
    stream: Union[TextIOWrapper, StringIO], filepath_errmsg: str
) -> dict[str, Any]:
    """Parse the coordinate system section.

    Args:
        stream: Text stream to read from
        filepath_errmsg: Error context for meaningful error messages

    Returns:
        dict: Coordinate system data with keys:
        name, axis_name, axis_unit, zpositive

    Raises:
        ValueError: If coordinate system section is malformed or incomplete
    """

    # Expected format
    err_msg = (
        f"\nIn file {filepath_errmsg}:\n"
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
    for line in _read_line(stream):
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


def _parse_tface_section(
    stream: Union[TextIOWrapper, StringIO], filepath_errmsg: str
) -> tuple[list[list[float]], list[list[int]]]:
    """Parse the TFACE section with data defining a triangulated surface
    (vertices and triangles).

    Args:
        stream: Text stream to read from
        filepath_errmsg: Error context for meaningful error messages

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
        f"\nIn file {filepath_errmsg}:\n"
        "Invalid 'VRTX' line in 'TFACE' section.\n"
        "Expected format: 'VRTX id x y z [attributes]'\n"
        "where:\n"
        "  - 'id' starts with '1' and increments by 1 for each vertex\n"
        "  - 'x', 'y', and 'z' are floating-point numbers\n"
        "  - the only currently supported attribute is 'CNXYZ'."
    )

    err_msg_trgl = (
        f"\nIn file {filepath_errmsg}:\n"
        "Invalid 'TRGL' line in 'TFACE' section.\n"
        "Expected format: 'TRGL vertex_id1 vertex_id2 vertex_id3'\n"
        "where:\n"
        "  - each vertex id must be an integer\n"
        "  - each vertex id must be in the range "
        "{{1, number of vertices}} (1-based indexing)\n"
    )

    # Keep track of expected vertex numbering, 1-based numbering
    vrtx_no = 0

    # Is 'END' statement present?
    end_is_present = False

    for line in _read_line(stream):
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
                f"\nIn file {filepath_errmsg}:\n"
                "Invalid line in 'TFACE' section with triangulated data.\n"
                "Expect lines to start with 'VRTX', 'TRGL', or 'END'.\n"
                f"Failing line: '{line}'",
            )

    if not end_is_present:
        raise ValueError(
            f"\nIn file {filepath_errmsg}:\n"
            "Missing 'END' statement at the end of the 'TFACE' section."
        )

    return vertices, triangles


def _create_tsurf_data(
    header_name: str,
    coord_sys_data: dict[str, Any],
    vertices: list[list[float]],
    triangles: list[list[int]],
) -> TSurfData:
    """Create TSurfData object from parsed components.

    Args:
        header_name: Surface name from header
        coord_sys_data: Optional dictionary with coordinate system data
        vertices: List of vertex coordinates
        triangles: List of vertex indices

    Returns:
        TSurfData: Complete TSurf data object
    """

    # Create header object
    header = TSurfHeader(name=header_name)

    # Create coordinate system object if data was provided
    coord_sys = None
    if coord_sys_data:
        coord_sys = TSurfCoordSys(
            name=coord_sys_data["name"],
            axis_name=coord_sys_data["axis_name"],
            axis_unit=coord_sys_data["axis_unit"],
            zpositive=coord_sys_data["zpositive"],
        )

    # Convert to numpy arrays with appropriate data types
    vertices_array = np.array(vertices, dtype=np.float64)
    triangles_array = np.array(triangles, dtype=np.int64)

    return TSurfData(
        header=header,
        coord_sys=coord_sys,
        vertices=vertices_array,
        triangles=triangles_array,
    )


def _validate_triangulation_data(vertices: np.ndarray, triangles: np.ndarray) -> None:
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
        raise ValueError("Triangle vertex indices must be >= 1 in triangulation data.")

    if np.any(triangles > len(vertices)):
        raise ValueError(
            "Triangle vertex indices must be <= number of vertices in "
            "triangulation data."
        )


def _parse_tsurf(
    stream: Union[TextIOWrapper, StringIO], filepath_errmsg: str
) -> TSurfData:
    """
    Parse a TSurf file from a file stream.

    Args:
        stream: A stream of the TSurf file.
        filepath_errmsg: Error context for meaningful error messages
    """

    # NOTES:
    # - The TSurf format has many more sections and attributes than
    #     those currently captured in this parser
    # - The first line must be the TSurf signature line: 'GOCAD TSurf 1'
    # - While we are not aware of whether the ordering of the other
    #     sections is important, the ordering is not strictly enforced
    #     in this parser
    # - Only one section of each type is allowed
    # - The parser raises an error for unknown sections/keywords
    # - The parser could be extended to handle more sections/keywords

    # Initialize all variables
    header_name: str = ""
    coord_sys_data: dict[str, Any] = {}
    vertices: list[list[float]] = []
    triangles: list[list[int]] = []

    # Keep track of which sections that have been read and validated.
    header_section_read = False
    coord_sys_section_read = False
    tface_section_read = False

    # Check if file has data
    first_char = stream.read(1)
    if not first_char:
        raise ValueError(
            f"\nIn file {filepath_errmsg}:\n"
            "This file is not a valid TSurf file, it is empty."
        )
    stream.seek(0)

    # Ensure first line is the TSurf signature line
    first_line = _read_line(stream).__next__()
    if not _is_tsurf_signature(first_line):
        raise ValueError(
            f"\nIn file {filepath_errmsg}:\n"
            "This file is not a valid TSurf file.\n"
            "The TSurf signature line 'GOCAD TSurf 1' must be "
            "the first line in the file."
        )

    # Loop over sections, each section starts with a specific keyword.
    # For each section there is a specific parsing function
    # which reads lines until the end of the section.
    # Line pointer is now on the second line in the file.
    for line in _read_line(stream):
        if _is_tsurf_signature(line):
            raise ValueError(
                f"\nIn file {filepath_errmsg}:\n"
                "Multiple TSurf signature lines 'GOCAD TSurf 1' found, "
                "but only one is allowed."
            )

        if _is_header_section_first_line(line):
            # Ensure only one section of this type
            if header_section_read:
                raise ValueError(
                    f"\nIn file {filepath_errmsg}:\n"
                    "Multiple 'HEADER' sections found, "
                    "but only one is allowed."
                )
            header_section_read = True

            header_name = _parse_header_section(stream, filepath_errmsg)
            _validate_header_section(header_name, filepath_errmsg)
            continue

        if _is_coordinate_system_section_first_line(line):
            # Ensure only one section of this type
            if coord_sys_section_read:
                raise ValueError(
                    f"\nIn file {filepath_errmsg}:\n"
                    "Multiple 'COORDINATE_SYSTEM' sections found, "
                    "but only one is allowed."
                )
            coord_sys_section_read = True

            coord_sys_data = _parse_coordinate_system_section(stream, filepath_errmsg)
            ValidatorCoordSys.validate_coord_sys_data(coord_sys_data, filepath_errmsg)
            continue

        if _is_tface_section_first_line(line):
            # Ensure only one section of this type
            if tface_section_read:
                raise ValueError(
                    f"\nIn file {filepath_errmsg}:\n"
                    "Multiple 'TFACE' sections found, "
                    "but only one is allowed."
                )
            tface_section_read = True

            vertices, triangles = _parse_tface_section(stream, filepath_errmsg)
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
            f"\nIn file {filepath_errmsg}:\n"
            "The file contains an invalid line which is not recognized \n"
            "as a section identifier (first line of a section).\n"
            "This may be either an error, or a valid TSurf section identifier\n"
            "that is not (yet) handled by the file parser.\n"
            f"Failing line: '{line}'",
        )

    # Ensure mandatory sections were read
    if not header_section_read:
        raise ValueError(
            f"\nIn file {filepath_errmsg}:\nMissing mandatory 'HEADER' section.\n"
        )
    if not tface_section_read:
        raise ValueError(
            f"\nIn file {filepath_errmsg}:\nMissing mandatory 'TFACE' section.\n"
        )

    tsurf_data = _create_tsurf_data(header_name, coord_sys_data, vertices, triangles)
    _validate_triangulation_data(tsurf_data.vertices, tsurf_data.triangles)
    return tsurf_data


def _validate_file_path(file: Path, filepath_errmsg: str) -> None:
    """Validate that file path exists and has correct extension.

    Args:
        file: Path object to validate
        filepath_errmsg: Error context for meaningful error messages

    Raises:
        FileNotFoundError: If file doesn't exist or isn't a regular file
        ValueError: If file doesn't have .ts extension
    """
    if not file.exists():
        raise FileNotFoundError(f"\nFile {filepath_errmsg}:\nThe file does not exist.")

    if not file.is_file():
        raise FileNotFoundError(
            f"\nFile {filepath_errmsg}:\n"
            + "The file is not a regular file. "
            + "It may be a directory or a special file type."
        )

    if file.suffix != ".ts":
        raise ValueError(
            f"\nFile {filepath_errmsg}:\n"
            "The file is not a TSurf file. Expected '.ts' extension."
        )


@contextmanager
def _get_text_stream(
    file: FileLike, encoding: str
) -> Generator[Union[TextIOWrapper, StringIO], None, None]:
    """Context manager that yields a text stream.

    Handles different input types:
    - str: converts to Path and opens file with specified encoding
    - Path objects: opens file with specified encoding
    - BytesIO: wraps with TextIOWrapper with specified encoding
    - StringIO: yields directly (already text-based)
    """
    if isinstance(file, (str, Path)):
        file_path = Path(file) if isinstance(file, str) else file
        with open(file_path, encoding=encoding) as stream:
            yield stream
    elif isinstance(file, BytesIO):
        with TextIOWrapper(file, encoding=encoding) as text_wrapper:
            yield text_wrapper
    elif isinstance(file, StringIO):
        yield file
    else:
        raise TypeError(f"Invalid type for 'file': {type(file)}")


def read_tsurf(
    file: FileLike,
    encoding: str = "utf-8",
) -> TSurfData:
    """
    Read a TSurf file and parse its triangulated surface data.
    Note that only a subset of the TSurf format is currently supported,
    more types of sections exist in the format specification.

    Args:
        file: Path to TSurf file (str or Path) or
              a file-like object (BytesIO or StringIO).
        encoding: Text encoding for reading the file

    Returns:
        TSurfData: Parsed surface data containing header, coordinate system,
            vertices, and triangles

    Raises:
        FileNotFoundError: If file path doesn't exist or isn't a regular file
        ValueError: If file extension isn't .ts or file format is invalid
    """

    # TSurf is a file format used in for example
    # the GOCAD software. RMS can export triangulated surfaces in its
    # structural model in the TSurf format.
    # When unhandled keywords are present in the file, the processing is halted
    # and an error message is issued. The user is expected to take action
    # to fix the file.
    # Documentation for the TSurf format is limited if you don't
    # have access to the  GOCAD Developer's Guide, but
    # here is one:
    # - https://paulbourke.net/dataformats/gocad/gocad.pdf

    # Determine error message context based on file type
    filepath_errmsg: str
    if isinstance(file, str):
        filepath_errmsg = file
    elif isinstance(file, Path):
        filepath_errmsg = str(file)
    elif isinstance(file, StringIO):
        filepath_errmsg = "'input stream (StringIO)'"
    elif isinstance(file, BytesIO):
        filepath_errmsg = "'input stream (BytesIO)'"
    else:
        filepath_errmsg = "'input stream (unknown type)'"

    if isinstance(file, (str, Path)):
        file_path = Path(file) if isinstance(file, str) else file
        _validate_file_path(file_path, filepath_errmsg)

    # Parse the file using context manager for proper resource handling
    with _get_text_stream(file, encoding) as stream:
        return _parse_tsurf(stream, filepath_errmsg)
