import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from io import BytesIO, StringIO, TextIOWrapper
from pathlib import Path
from typing import Any, Generator, Optional, Union

import numpy as np

from xtgeo.common.types import FileLike


class KeywordValidatorCoordSys:
    """
    Validator for keywords in the TSurf file format.
    For each keyword there are multiple possible values.
    The reader does not (yet) recognise and handle all values
    that are valid in the TSurf format.
    When needed, the list can be extended.

    Note that for some of the keywords there is a short-list of allowed values
    (e.g. 'ZPOSITIVE'), other values are not allowed. The reader will issue an
    error when an invalid value is encountered.
    For other keywords (e.g. 'AXIS_NAME'), the reader will only issue a warning
    when an "uncommon" value is used.
    """

    # axis_names and axis_units: the whole tuple is specified, instead of single
    # values that can be used for each position in the tuples.
    # This is to ensure that physically meaningful relations are used.
    # For example, it makes no sense to let two elements in axis_names be equal.
    # Or to use two wildly different axis_units laterally.
    # This is quite strict and could be relaxed if needed.

    axis_names = {"xyz": ("X", "Y", "Z")}
    """XYZ are the most common axis names"""

    axis_units = {"mmm": ("m", "m", "m"), "fff": ("ft", "ft", "ft")}
    """meters is the most common unit"""

    z_positives = {"depth": "Depth", "elevation": "Elevation"}
    """Z is increasing downwards and upwards, respectively"""

    @classmethod
    def validate_axis_names(cls, axis_names: list[str], filepath_errmsg: str) -> None:
        """Validate axis names and issue warnings for uncommon values.

        Args:
            axis_names: List of axis names to validate
            filepath_errmsg: Error context for meaningful messages
        """

        axis_names_tuple = tuple(axis_names)

        if (
            len(axis_names) != 3
            or axis_names[0] == ""
            or axis_names[1] == ""
            or axis_names[2] == ""
        ):
            raise ValueError(
                f"In file {filepath_errmsg}: AXIS_NAME must have exactly three values, "
                f"found {len(axis_names)}: {axis_names}"
            )

        # Check if the axis names are in our known common values
        if axis_names_tuple not in cls.axis_names.values():
            # TODO: looks like it's testing the individual elements,
            # not the complete tuple as it should

            # Check if any individual names are completely unknown
            all_known_names: set[str] = set()
            for names in cls.axis_names.values():
                all_known_names.update(names)

            # Provide user with hints of more common names
            # List of common names can be extended
            unknown_names = [name for name in axis_names if name not in all_known_names]
            if unknown_names:
                warnings.warn(
                    f"In file {filepath_errmsg}: Uncommon AXIS_NAME values detected: "
                    f"{unknown_names}. More common names are: "
                    f"{list(all_known_names)}",
                    UserWarning,
                    stacklevel=3,
                )
            else:
                warnings.warn(
                    f"In file {filepath_errmsg}: Uncommon AXIS_NAME combination: "
                    f"{axis_names_tuple}. More common combinations are: "
                    f"{list(cls.axis_names.values())}",
                    UserWarning,
                    stacklevel=3,
                )

    @classmethod
    def validate_axis_units(cls, axis_units: list[str], filepath_errmsg: str) -> None:
        """Validate axis units and issue warnings for uncommon values.

        Args:
            axis_units: List of axis units to validate
            filepath_errmsg: Error context for meaningful messages
        """

        axis_units_tuple = tuple(axis_units)

        if (
            len(axis_units_tuple) != 3
            or axis_units_tuple[0] == ""
            or axis_units_tuple[1] == ""
            or axis_units_tuple[2] == ""
        ):
            raise ValueError(
                f"In file {filepath_errmsg}: AXIS_UNIT must have exactly three values, "
                f"found {len(axis_units)}: {axis_units}"
            )

        # Check if the axis units are in our known common values
        if axis_units_tuple not in cls.axis_units.values():
            warnings.warn(
                f"In file {filepath_errmsg}: Uncommon AXIS_UNIT combination: "
                f"{axis_units_tuple}. More common combinations are: "
                f"{list(cls.axis_units.values())}",
                UserWarning,
                stacklevel=3,
            )

    @classmethod
    def validate_zpositive(cls, zpositive: str, filepath_errmsg: str) -> None:
        """Validate zpositive value and raise error for invalid values.

        Args:
            zpositive: Z-positive value to validate
            filepath_errmsg: Error context for meaningful messages

        Raises:
            ValueError: If zpositive value is not in allowed values
        """

        if zpositive not in cls.z_positives.values():
            raise ValueError(
                f"In file {filepath_errmsg}: Invalid ZPOSITIVE value '{zpositive}'. "
                f"Allowed values are: {list(cls.z_positives.values())}"
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

        if not split_line:
            continue

        if split_line[0].startswith("#"):
            continue

        yield split_line


def _validate_tsurf_signature(line: list[str], filepath_errmsg: str) -> None:
    """Validate that the first line contains the correct TSurf signature.

    Args:
        line: First line tokens from the file
        filepath_errmsg: Error context for meaningful error messages

    Raises:
        ValueError: If signature is not 'GOCAD TSurf 1'
    """

    # First line should be exactly "GOCAD TSurf 1"
    if " ".join(line) != "GOCAD TSurf 1":
        raise ValueError(
            f"\nIn input {filepath_errmsg}:\n"
            "The first line indicates that this is not a valid TSurf "
            "object. Expected 'GOCAD TSurf 1'."
        )


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

    header_name = None

    for line in _read_line(stream):
        # Assume it can be either ['name:F5'] or ['name:', 'F5']
        # Assume the name itself may be split into several strings:
        # e.g. ['Massive', 'listric', 'fault']
        # Assume line[0] is lowercase
        if line[0].startswith("name:"):
            # TODO: make tests for the different options
            # Extract name after the colon and join with remaining parts
            name_parts = [line[0][5:]] + line[1:]
            header_name = " ".join(part for part in name_parts if part)
        elif line[0] == "}":
            break
        else:
            raise ValueError(
                f"\nIn file {filepath_errmsg}:\n"
                "The 'HEADER' section must exist and "
                "is expected to have exactly one attribute: 'name'."
            )

    if header_name is None:
        raise ValueError(
            f"\nIn file {filepath_errmsg}:\n"
            "The 'HEADER' section must contain a 'name' attribute."
        )

    return header_name


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

    coord_sys_data: dict[str, Any] = {}

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
            break
        else:
            raise ValueError(
                f"\nIn file {filepath_errmsg}:\n"
                "Invalid 'COORDINATE_SYSTEM' section, expected"
                " exactly four attributes:\n"
                "'NAME', 'AXIS_NAME', 'AXIS_UNIT' and 'ZPOSITIVE'"
            )

    return coord_sys_data


def _parse_triangulation_data_section(
    stream: Union[TextIOWrapper, StringIO], filepath_errmsg: str
) -> tuple[list[list[float]], list[list[int]]]:
    """Parse the TFACE section with data defining a triangulated surface.

    Args:
        stream: Text stream to read from
        filepath_errmsg: Error context for meaningful error messages

    Returns:
        tuple: (vertices, triangles) where vertices is
        a list of [x,y,z] coordinates and triangles is a list of vertex indices

    Raises:
        ValueError: If TFACE section contains invalid lines
    """

    vertices: list[list[float]] = []
    triangles: list[list[int]] = []

    err_msg_vrtx = f"\nIn file {filepath_errmsg}:\n"
    err_msg_vrtx += "Invalid 'VRTX' line in 'TFACE' section.\n"
    err_msg_vrtx += "Expected format: 'VRTX id x y z [attributes]'\n"
    err_msg_vrtx += "where:\n"
    err_msg_vrtx += "  - 'id' starts with '1' and increments by 1 "
    err_msg_vrtx += "for each consecutive vertex\n"
    err_msg_vrtx += "  - 'x', 'y', and 'z' are floating-point numbers\n"
    err_msg_vrtx += "  - the only currently supported attribute is 'CNXYZ'."

    err_msg_trgl = f"\nIn file {filepath_errmsg}:\n"
    err_msg_trgl += "Invalid 'TRGL' line in 'TFACE' section.\n"
    err_msg_trgl += "Expected format: 'TRGL vertex_id1 vertex_id2 vertex_id3'\n"
    err_msg_trgl += "where:\n"
    err_msg_trgl += "  - each vertex id must be an integer\n"
    err_msg_trgl += "  - each vertex id must be in the range "
    err_msg_trgl += "{1, number of vertices} (1-based indexing)\n"

    # Keep track of expected vertex numbering, 1-based numbering
    vrtx_no = 1

    for line in _read_line(stream):
        if line[0] == "VRTX":
            if len(line) != 6 or line[1] != str(vrtx_no) or line[5] != "CNXYZ":
                err_msg_vrtx += f"Failing line: '{line}'"
                raise ValueError(err_msg_vrtx)

            try:
                vertices.append([float(line[2]), float(line[3]), float(line[4])])
            except ValueError:
                err_msg_vrtx += f"Failing line: '{line}'"
                raise ValueError(err_msg_vrtx)

            vrtx_no += 1

        elif line[0] == "TRGL":
            if (
                len(line) != 4
                or int(line[1]) < 1
                or int(line[2]) < 1
                or int(line[3]) < 1
                or int(line[1]) > len(vertices)
                or int(line[2]) > len(vertices)
                or int(line[3]) > len(vertices)
            ):
                err_msg_trgl += f"Failing line: '{line}'"
                raise ValueError(err_msg_trgl)

            try:
                triangles.append([int(line[1]), int(line[2]), int(line[3])])
            except ValueError:
                err_msg_trgl += f"Failing line: '{line}'"
                raise ValueError(err_msg_trgl)

        elif line[0] == "END":
            # End of TFACE section
            break

        else:
            raise ValueError(
                f"\nIn file {filepath_errmsg}:\n"
                "Invalid line in 'TFACE' section with triangulated data.\n"
                "Expect lines to start with 'VRTX', 'TRGL', or 'END'.\n"
                f"Failing line: '{line}'",
            )

    return vertices, triangles


def _validate_parsed_data(
    header_name: str,
    coord_sys_data: dict[str, Any],
    vertices: list[list[float]],
    triangles: list[list[int]],
    filepath_errmsg: str,
) -> None:
    """Validate that parsed data meets the requirements.

    Args:
        header_name: Surface name from header
        coord_sys_data: Optional dictionary with coordinate system data
        vertices: List of vertex coordinates
        triangles: List of vertex indices
        filepath_errmsg: Error context for meaningful error messages

    Raises:
        ValueError: If data doesn't meet the requirements
    """

    if header_name == "":
        raise ValueError(
            f"\nIn file {filepath_errmsg}:\n"
            "Missing required HEADER section with 'name' attribute."
        )

    if coord_sys_data:
        required_fields = ["name", "axis_name", "axis_unit", "zpositive"]
        missing_fields = [f for f in required_fields if f not in coord_sys_data]
        if missing_fields:
            raise ValueError(
                f"\nIn file {filepath_errmsg}:\n"
                f"Coordinate system section missing fields: {missing_fields}"
            )

        # Validate coordinate system data
        if "axis_name" in coord_sys_data:
            KeywordValidatorCoordSys.validate_axis_names(
                coord_sys_data["axis_name"], filepath_errmsg
            )

        if "axis_unit" in coord_sys_data:
            KeywordValidatorCoordSys.validate_axis_units(
                coord_sys_data["axis_unit"], filepath_errmsg
            )

        if "zpositive" in coord_sys_data:
            KeywordValidatorCoordSys.validate_zpositive(
                coord_sys_data["zpositive"], filepath_errmsg
            )

    if len(vertices) < 3:
        raise ValueError(
            f"\nIn file {filepath_errmsg}:\n"
            f"Insufficient vertices: found {len(vertices)}, need at least 3."
        )

    if len(triangles) < 1:
        raise ValueError(
            f"\nIn file {filepath_errmsg}:\n"
            f"Insufficient triangles: found {len(triangles)}, need at least 1."
        )


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


def _parse_tsurf(
    stream: Union[TextIOWrapper, StringIO], filepath_errmsg: str
) -> TSurfData:
    """
    Parse a TSurf file from a file stream.

    Args:
        stream: A stream of the TSurf file.
        filepath_errmsg: Error context for meaningful error messages
    """

    # Initialize all variables
    header_name: str = str()
    coord_sys_data: dict[str, Any] = {}
    vertices: list[list[float]] = []
    triangles: list[list[int]] = []

    # Validate the first line (GOCAD TSurf format identifier)
    first_line = True
    for line in _read_line(stream):
        if first_line:
            _validate_tsurf_signature(line, filepath_errmsg)
            first_line = False
            continue

        if line[0] == "HEADER" and len(line) > 1 and line[1] == "{":
            header_name = _parse_header_section(stream, filepath_errmsg)
            continue

        if line[0] == "GOCAD_ORIGINAL_COORDINATE_SYSTEM":
            coord_sys_data = _parse_coordinate_system_section(stream, filepath_errmsg)
            continue

        if line[0] == "TFACE":
            vertices, triangles = _parse_triangulation_data_section(
                stream, filepath_errmsg
            )
            continue

        # -----------------------
        # Handle unknown keywords
        # -----------------------
        # The TSurf file format handles many more keywords and attributes
        # than those captured in this parser. Could issue a warning
        # instead of an error. But if this is a section that
        # continues over several lines, it is difficult to know where the
        # section ends and where parsing can continue.
        # So for now, raise an error.

        raise ValueError(
            f"\nIn file {filepath_errmsg}:\n"
            "The file contains an invalid line.\n"
            "This may be either an error, or a valid TSurf keyword or attribute\n"
            "that is not (yet) handled by the file parser.\n"
            f"Failing line: '{line}'",
        )

    _validate_parsed_data(
        header_name, coord_sys_data, vertices, triangles, filepath_errmsg
    )

    return _create_tsurf_data(header_name, coord_sys_data, vertices, triangles)


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
            f"\nFile {filepath_errmsg}:\nThe file is not a regular file."
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
        raise ValueError(f"Invalid type for 'file': {type(file)}")


def read_tsurf(
    file: FileLike,
    encoding: str = "utf-8",
) -> TSurfData:
    """
    Read a TSurf file and parse triangulated surface data.

    Args:
        file: Path to TSurf file (str or Path) or
            a file-like object (BytesIO or StringIO).
        encoding: Text encoding for reading the file (default: utf-8)

    Returns:
        TSurfData: Parsed surface data containing header, coordinate system,
            vertices, and triangles

    Raises:
        FileNotFoundError: If file path doesn't exist or isn't a regular file
        ValueError: If file extension isn't .ts or file format is invalid
    """

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
        raise ValueError(f"Invalid type for 'file': {type(file)}")

    if isinstance(file, (str, Path)):
        file_path = Path(file) if isinstance(file, str) else file
        _validate_file_path(file_path, filepath_errmsg)

    # Parse the file using context manager for proper resource handling
    with _get_text_stream(file, encoding) as stream:
        return _parse_tsurf(stream, filepath_errmsg)
