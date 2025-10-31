from io import BytesIO, StringIO
from pathlib import Path

import pytest

from xtgeo.io.tsurf._tsurf_reader import (
    ValidatorCoordSys,
    read_tsurf,
)

# ============================================================================
# PYTEST FIXTURES FOR TSURF FILE CONTENT
# ============================================================================


@pytest.fixture
def signature_line() -> str:
    """Return the TSurf signature line."""
    return "GOCAD TSurf 1\n"


@pytest.fixture
def header_section() -> str:
    """Return a valid header section."""
    return "HEADER {\nname: test_surface\n}\n"


@pytest.fixture
def coordinate_system_section() -> str:
    """Return a valid coordinate system section."""
    return (
        "GOCAD_ORIGINAL_COORDINATE_SYSTEM\n"
        "NAME Default\n"
        'AXIS_NAME "X" "Y" "Z"\n'
        'AXIS_UNIT "m" "m" "m"\n'
        "ZPOSITIVE Depth\n"
        "END_ORIGINAL_COORDINATE_SYSTEM\n"
    )


@pytest.fixture
def tface_section() -> str:
    """Return a basic TFACE section with minimal geometry."""
    return (
        "TFACE\n"
        "VRTX 1 0.0 0.0 0.0 CNXYZ\n"
        "VRTX 2 1.0 0.0 0.0 CNXYZ\n"
        "VRTX 3 0.0 1.0 0.0 CNXYZ\n"
        "TRGL 1 2 3\n"
        "END\n"
    )


@pytest.fixture
def complete_tsurf_file(
    signature_line,
    header_section,
    coordinate_system_section,
    tface_section,
) -> str:
    """Return a complete valid TSurf file with all sections."""
    return signature_line + header_section + coordinate_system_section + tface_section


@pytest.fixture
def minimal_tsurf_file(signature_line, header_section, tface_section) -> str:
    """Return a minimal valid TSurf file (no coordinate system)."""
    return signature_line + header_section + tface_section


@pytest.fixture
def missing_signature_file(
    header_section,
    coordinate_system_section,
    tface_section,
) -> str:
    """Return TSurf file content missing signature line."""
    return header_section + coordinate_system_section + tface_section


@pytest.fixture
def missing_header_file(
    signature_line, coordinate_system_section, tface_section
) -> str:
    """Return TSurf file content missing header section."""
    return signature_line + coordinate_system_section + tface_section


@pytest.fixture
def missing_tface_file(
    signature_line, header_section, coordinate_system_section
) -> str:
    """Return TSurf file content missing TFACE section."""
    return signature_line + header_section + coordinate_system_section


@pytest.fixture
def only_signature_file(signature_line) -> str:
    """Return TSurf file content with only signature line."""
    return signature_line


def tsurf_stream(content: str) -> StringIO:
    """Helper function to create StringIO from TSurf content."""
    return StringIO(content)


def test_file_string_input(tmp_path: str, complete_tsurf_file) -> None:
    """Test reading from string input."""
    filepath = str(tmp_path) + "/test.ts"
    with open(filepath, "w") as f:
        f.write(complete_tsurf_file)

    result_path = read_tsurf(filepath)
    assert result_path is not None


def test_file_path_input(tmp_path: Path, complete_tsurf_file) -> None:
    """Test reading from Path input."""
    # Test with Path
    filepath = tmp_path / "test.ts"
    with open(filepath, "w") as f:
        f.write(complete_tsurf_file)

    result_path = read_tsurf(filepath)
    assert result_path is not None


def test_file_stringio_input(complete_tsurf_file) -> None:
    """Test reading from StringIO input."""
    result_stringio = read_tsurf(tsurf_stream(complete_tsurf_file))
    assert result_stringio is not None


def test_file_bytesio_input(complete_tsurf_file) -> None:
    """Test reading from BytesIO input."""
    result_bytesio = read_tsurf(BytesIO(complete_tsurf_file.encode("utf-8")))
    assert result_bytesio is not None


def test_file_non_regular_file_input(tmp_path: Path) -> None:
    """Test reading from a non-regular file (e.g., folder)."""

    non_regular_file = tmp_path / "some_folder"
    non_regular_file.mkdir()

    with pytest.raises(FileNotFoundError, match="is not a regular file"):
        read_tsurf(non_regular_file)


def test_file_other_than_filelike_input() -> None:
    """Test reading from an unsupported input type."""
    with pytest.raises(TypeError, match="Invalid type for 'file'"):
        read_tsurf(12345)  # Invalid input type


def test_file_invalid_suffix(tmp_path: Path) -> None:
    """Test with invalid file suffix."""
    filepath = tmp_path / "invalid.txt"
    filepath.touch()
    with pytest.raises(ValueError, match="is not a TSurf file"):
        read_tsurf(filepath)


def test_file_non_existent(tmp_path: Path) -> None:
    """Test with non-existent file."""
    filepath = tmp_path / "non_existent.ts"
    with pytest.raises(FileNotFoundError, match="The file does not exist"):
        read_tsurf(filepath)


def test_file_empty():
    """Test that empty file raises appropriate error."""
    with pytest.raises(ValueError, match="not a valid TSurf"):
        read_tsurf(StringIO(""))


def test_comments_and_empty_lines(tmp_path: Path) -> None:
    """Test handling of comments and empty lines in files."""
    content_lines = [
        "GOCAD TSurf 1",
        "# This is a comment",
        "",  # Empty line
        "HEADER {",
        "name: test_surface",
        "}",
        "# Another comment",
        "",  # Empty line
        "TFACE",
        "VRTX 1 0.0 0.0 0.0 CNXYZ",
        "VRTX 2 1.0 0.0 0.0 CNXYZ",
        "# Another comment",
        "VRTX 3 0.0 1.0 0.0 CNXYZ",
        "# Another comment",
        "TRGL 1 2 3",
        "# Another comment",
        "END",
    ]

    content = "\n".join(content_lines) + "\n"
    filepath = tmp_path / "with_comments.ts"
    with open(filepath, "w") as f:
        f.write(content)

    result = read_tsurf(filepath)
    assert result is not None
    assert result.header.name == "test_surface"


def test_sections_all(complete_tsurf_file):
    """Test a valid TSurf file with all sections."""
    result = read_tsurf(tsurf_stream(complete_tsurf_file))

    assert result is not None
    assert result.header.name == "test_surface"
    assert result.coord_sys is not None
    assert result.coord_sys.name == "Default"
    assert len(result.vertices) == 3
    assert len(result.triangles) == 1


def test_sections_minimal(minimal_tsurf_file):
    """Test valid TSurf file with only mandatory sections."""
    result = read_tsurf(tsurf_stream(minimal_tsurf_file))

    assert result is not None
    assert result.header.name == "test_surface"
    assert result.coord_sys is None
    assert len(result.vertices) == 3
    assert len(result.triangles) == 1


def test_section_missing_signature(missing_signature_file):
    """Test that missing signature line raises appropriate error."""
    with pytest.raises(ValueError, match="not a valid TSurf"):
        read_tsurf(tsurf_stream(missing_signature_file))


def test_section_missing_header(missing_header_file):
    """Test that missing header section raises appropriate error."""
    with pytest.raises(ValueError, match="Missing mandatory 'HEADER' section"):
        read_tsurf(tsurf_stream(missing_header_file))


def test_section_missing_coordinate_system(minimal_tsurf_file):
    """Test that missing optional coordinate system section is handled"""
    result = read_tsurf(tsurf_stream(minimal_tsurf_file))

    assert result is not None
    assert result.coord_sys is None  # Coordinate system is optional


def test_section_missing_tface(missing_tface_file):
    """Test that missing TFACE section raises appropriate error."""
    with pytest.raises(ValueError, match="Missing mandatory 'TFACE' section"):
        read_tsurf(tsurf_stream(missing_tface_file))


def test_section_invalid_keyword(signature_line, header_section, tface_section):
    """Test that invalid section keyword raises appropriate error."""
    tsurf_content = (
        signature_line + header_section + "INVALID_SECTION\n" + tface_section
    )

    with pytest.raises(
        ValueError, match="The file contains an invalid line which is not recognized"
    ):
        read_tsurf(tsurf_stream(tsurf_content))


def test_section_header_after_coordinate_system(
    signature_line, coordinate_system_section, header_section, tface_section
):
    """Test header section appearing after coordinate system section."""
    content = (
        signature_line + coordinate_system_section + header_section + tface_section
    )

    result = read_tsurf(tsurf_stream(content))
    assert result is not None
    assert result.header is not None
    assert result.header.name == "test_surface"
    assert result.coord_sys is not None


def test_section_header_after_tface(signature_line, tface_section, header_section):
    """Test header section appearing after TFACE section."""
    content = signature_line + tface_section + header_section

    result = read_tsurf(tsurf_stream(content))
    assert result is not None
    assert result.header is not None
    assert result.header.name == "test_surface"
    assert result.vertices is not None
    assert result.triangles is not None


def test_section_tface_appearing_twice(signature_line, header_section, tface_section):
    """Test TFACE section appearing twice."""
    content = signature_line + header_section + tface_section + tface_section

    with pytest.raises(ValueError, match="Multiple 'TFACE' sections found"):
        read_tsurf(tsurf_stream(content))


def test_signature_only(only_signature_file):
    """Test file with only signature line."""
    with pytest.raises(ValueError, match="Missing mandatory 'HEADER' section"):
        read_tsurf(tsurf_stream(only_signature_file))


def test_signature_multiple_header_between(
    signature_line, header_section, tface_section
):
    """
    Test for multiple signature lines.
    This test places the second signature line after a HEADER section.
    """
    tsurf_content = signature_line + header_section + signature_line + tface_section

    with pytest.raises(ValueError, match="Multiple TSurf signature lines"):
        read_tsurf(tsurf_stream(tsurf_content))


def test_signature_invalid_line(header_section, tface_section):
    """Test invalid signature line variations."""
    invalid_signatures = [
        "GOCAD TSurf 2\n",  # Wrong version
        "Invalid signature\n",  # Just wrong
    ]

    for invalid_sig in invalid_signatures:
        content = invalid_sig + header_section + tface_section

        with pytest.raises(ValueError, match="not a valid TSurf"):
            read_tsurf(tsurf_stream(content))


def test_header_with_different_names(signature_line, tface_section):
    """Test header section with various 'classic' surface names."""
    test_names = [
        "Surface_A",
        " Fault F1",
        "Complex Name With   Spaces",
        "*** $! --&((=))",
    ]

    for name in test_names:
        header_section = f"HEADER {{\nname: {name}\n}}\n"
        content = signature_line + header_section + tface_section

        result = read_tsurf(tsurf_stream(content))
        assert result.header.name[0] == name.strip()[0]


def test_header_name_format_empty(signature_line, tface_section):
    """Test header name format: empty name"""
    content_empty_name = signature_line + "HEADER {\nname: \n}\n" + tface_section
    with pytest.raises(
        ValueError, match="Missing or invalid name in the 'HEADER' section"
    ):
        read_tsurf(tsurf_stream(content_empty_name))


def test_header_name_format_no_space_after_colon(signature_line, tface_section):
    """Test header name format: no space after colon"""
    content_no_space = signature_line + "HEADER {\nname:F5\n}\n" + tface_section
    result = read_tsurf(tsurf_stream(content_no_space))
    assert result.header.name == "F5"


def test_header_with_invalid_line(signature_line, tface_section):
    """Test header section with invalid line."""
    malformed_header = "HEADER {\ninvalid_line_here\n}\n"

    content = signature_line + malformed_header + tface_section

    with pytest.raises(ValueError, match="Invalid 'HEADER' section line:"):
        read_tsurf(tsurf_stream(content))


def test_header_only_header_line(signature_line, tface_section):
    """Test header section that opens but has no content and no closing."""
    content = signature_line + "HEADER {\n" + tface_section

    with pytest.raises(ValueError, match="Invalid 'HEADER' section line"):
        read_tsurf(tsurf_stream(content))


def test_header_with_no_closing_brace(signature_line, tface_section):
    """Test header section missing closing brace."""
    content = (
        signature_line
        + "HEADER {\n"
        + "name: test_surface\n"
        # Missing closing brace
        + tface_section
    )

    with pytest.raises(ValueError, match="Invalid 'HEADER' section line"):
        read_tsurf(tsurf_stream(content))


def test_header_appearing_twice(signature_line, header_section, tface_section):
    """Test header section appearing twice."""
    another_header = "HEADER {\nname: another_surface\n}\n"
    content = signature_line + header_section + another_header + tface_section

    with pytest.raises(ValueError, match="Multiple 'HEADER' sections found"):
        read_tsurf(tsurf_stream(content))


def test_header_incomplete_eof(signature_line):
    """Test header section that opens but has no content."""
    content = (
        signature_line + "HEADER {\n"
        # File ends abruptly here - no closing tag, no content
    )

    with pytest.raises(
        ValueError, match="Missing '}' at the end of the 'HEADER' section"
    ):
        read_tsurf(tsurf_stream(content))


def test_coordinate_system_incomplete_eof(signature_line, header_section):
    """Test coordinate system section that opens but has no content."""
    content = (
        signature_line + header_section + "GOCAD_ORIGINAL_COORDINATE_SYSTEM\n"
        # File ends abruptly here - no closing tag, no content
    )

    with pytest.raises(
        ValueError, match="Missing 'END_ORIGINAL_COORDINATE_SYSTEM' statement"
    ):
        read_tsurf(tsurf_stream(content))


def test_coordinate_system_with_invalid_lines(
    signature_line, header_section, tface_section
):
    """Test coordinate system section with invalid/unknown lines."""
    content = (
        signature_line
        + header_section
        + "GOCAD_ORIGINAL_COORDINATE_SYSTEM\n"
        + "NAME Default\n"
        + "INVALID_LINE_HERE\n"
        + "END_ORIGINAL_COORDINATE_SYSTEM\n"
        + tface_section
    )

    with pytest.raises(ValueError, match="Invalid line in 'COORDINATE_SYSTEM' section"):
        read_tsurf(tsurf_stream(content))


def test_coordinate_system_missing_fields(
    signature_line, header_section, tface_section
):
    """Test coordinate system section missing required fields."""
    incomplete_coord_sys = (
        "GOCAD_ORIGINAL_COORDINATE_SYSTEM\n"
        "NAME Default\n"
        # Missing AXIS_NAME, AXIS_UNIT, ZPOSITIVE
        "END_ORIGINAL_COORDINATE_SYSTEM\n"
    )

    content = signature_line + header_section + incomplete_coord_sys + tface_section

    with pytest.raises(ValueError, match="Coordinate system section missing fields"):
        read_tsurf(tsurf_stream(content))


def test_coordinate_system_with_extra_fields(
    signature_line, header_section, tface_section
):
    """Test coordinate system section with extra unknown fields."""
    malformed_coord_sys = (
        "GOCAD_ORIGINAL_COORDINATE_SYSTEM\n"
        "NAME Default\n"
        'AXIS_NAME "X" "Y" "Z"\n'
        "EXTRA_FIELD unexpected_value\n"  # Extra unknown field
        'AXIS_UNIT "m" "m" "m"\n'
        "ZPOSITIVE Depth\n"
        "END_ORIGINAL_COORDINATE_SYSTEM\n"
    )

    content = signature_line + header_section + malformed_coord_sys + tface_section

    with pytest.raises(ValueError, match="Invalid line in 'COORDINATE_SYSTEM' section"):
        read_tsurf(tsurf_stream(content))


def test_coordinate_system_appearing_twice(
    signature_line, header_section, coordinate_system_section, tface_section
):
    """Test coordinate system section appearing twice."""
    content = (
        signature_line
        + header_section
        + coordinate_system_section
        + tface_section
        + coordinate_system_section
    )

    with pytest.raises(ValueError, match="Multiple 'COORDINATE_SYSTEM' sections found"):
        read_tsurf(tsurf_stream(content))


def test_coordinate_system_with_invalid_axis_names(
    signature_line, header_section, tface_section
):
    """Test coordinate system section with invalid axis names."""
    malformed_coord_sys = (
        "GOCAD_ORIGINAL_COORDINATE_SYSTEM\n"
        "NAME Default\n"
        'AXIS_NAME "X" "Y"\n'  # Only two axis names
        'AXIS_UNIT "m" "m" "m"\n'
        "ZPOSITIVE Depth\n"
        "END_ORIGINAL_COORDINATE_SYSTEM\n"
    )

    content = signature_line + header_section + malformed_coord_sys + tface_section

    with pytest.raises(ValueError, match="AXIS_NAME must have exactly three values"):
        read_tsurf(tsurf_stream(content))


def test_coordinate_system_axis_names() -> None:
    """Test axis names validation."""

    # Valid axis names
    ValidatorCoordSys._validate_axis_names(("X", "Y", "Z"), "test_file")

    # Invalid number
    with pytest.raises(ValueError, match="exactly three values"):
        ValidatorCoordSys._validate_axis_names(("X", "Y"), "test_file")

    # Empty value
    with pytest.raises(ValueError, match="exactly three values"):
        ValidatorCoordSys._validate_axis_names(("X", "", "Z"), "test_file")

    # Uncommon names should warn
    with pytest.warns(UserWarning, match="Uncommon AXIS_NAME"):
        ValidatorCoordSys._validate_axis_names(("X", "B", "Z"), "test_file")


def test_coordinate_system_axis_units() -> None:
    """Test axis units validation."""

    # Valid units
    ValidatorCoordSys._validate_axis_units(("m", "m", "m"), "test_file")

    # Invalid number
    with pytest.raises(ValueError, match="exactly three values"):
        ValidatorCoordSys._validate_axis_units(("m", "m"), "test_file")

    # Uncommon units should warn
    with pytest.warns(UserWarning, match="Uncommon AXIS_UNIT"):
        ValidatorCoordSys._validate_axis_units(("m", "cm", "m"), "test_file")


def test_coordinate_system_zpositive() -> None:
    """Test zpositive validation."""

    # Valid values
    ValidatorCoordSys._validate_zpositive("Depth", "test_file")
    ValidatorCoordSys._validate_zpositive("Elevation", "test_file")

    # Invalid value
    with pytest.raises(ValueError, match="Invalid ZPOSITIVE value"):
        ValidatorCoordSys._validate_zpositive("Invalid", "test_file")


def test_tface_with_slightly_more_complex_geometry(signature_line, header_section):
    """Test TFACE section with slightly more complex geometry."""
    complex_tface = (
        "TFACE\n"
        "VRTX 1 0.0 0.0 0.0 CNXYZ\n"
        "VRTX 2 1.0 0.0 0.0 CNXYZ\n"
        "VRTX 3 0.0 1.0 0.0 CNXYZ\n"
        "VRTX 4 1.0 1.0 0.0 CNXYZ\n"
        "TRGL 1 2 3\n"
        "TRGL 2 4 3\n"
        "END\n"
    )

    content = signature_line + header_section + complex_tface

    result = read_tsurf(tsurf_stream(content))
    assert len(result.vertices) == 4
    assert len(result.triangles) == 2


def test_tface_missing_vertex_coordinate(signature_line, header_section):
    """Test TFACE section with missing vertex coordinate."""
    malformed_tface = (
        "TFACE\n"
        "VRTX 1 0.0 0.0 0.0\n"
        "VRTX 2 1.0 0.0\n"  # Missing Z coordinate
        "VRTX 3 0.0 1.0 0.0\n"
        "TRGL 1 2 3\n"
        "END\n"
    )

    content = signature_line + header_section + malformed_tface

    with pytest.raises(ValueError, match="Invalid 'VRTX' line"):
        read_tsurf(tsurf_stream(content))


def test_tface_invalid_vertex_coordinates(signature_line, header_section):
    """Test TFACE section with invalid vertex coordinates."""
    malformed_tface = (
        "TFACE\n"
        "VRTX 1 0.0 0.0 not_a_number CNXYZ\n"  # Invalid Z coordinate
        "VRTX 2 1.0 0.0 0.0 CNXYZ\n"
        "VRTX 3 0.0 1.0 0.0 CNXYZ\n"
        "TRGL 1 2 3\n"
        "END\n"
    )

    content = signature_line + header_section + malformed_tface

    with pytest.raises(ValueError, match="Invalid 'VRTX' line"):
        read_tsurf(tsurf_stream(content))


def test_tface_invalid_vertex_attribute(signature_line, header_section):
    """Test TFACE section with invalid vertex attribute."""
    malformed_tface = (
        "TFACE\n"
        "VRTX 1 0.0 0.0 0.0 INVALID_ATTR\n"  # Invalid attribute
        "VRTX 2 1.0 0.0 0.0 CNXYZ\n"
        "VRTX 3 0.0 1.0 0.0 CNXYZ\n"
        "TRGL 1 2 3\n"
        "END\n"
    )

    content = signature_line + header_section + malformed_tface

    with pytest.raises(ValueError, match="Invalid 'VRTX' line"):
        read_tsurf(tsurf_stream(content))


def test_tface_invalid_vertex_numbering(signature_line, header_section):
    """Test TFACE section with non-sequential vertex numbering."""
    malformed_tface = (
        "TFACE\n"
        "VRTX 1 0.0 0.0 0.0 CNXYZ\n"
        "VRTX 3 1.0 0.0 0.0 CNXYZ\n"  # Skips VRTX 2
        "VRTX 4 0.0 1.0 0.0 CNXYZ\n"
        "TRGL 1 3 4\n"
        "END\n"
    )

    content = signature_line + header_section + malformed_tface

    with pytest.raises(ValueError, match="Invalid 'VRTX' line"):
        read_tsurf(tsurf_stream(content))


def test_tface_triangles_with_non_existent_vertices(
    signature_line, header_section, coordinate_system_section
):
    """Test TFACE section with triangles referencing non-existent vertices."""
    content = (
        signature_line
        + header_section
        + coordinate_system_section
        + "TFACE\n"
        + "VRTX 1 0.0 0.0 0.0 CNXYZ\n"
        + "VRTX 2 1.0 0.0 0.0 CNXYZ\n"
        + "VRTX 3 0.0 1.0 0.0 CNXYZ\n"
        + "TRGL 1 2 4\n"  # Triangle references a non-existent vertex (4)
        + "END\n"
    )

    with pytest.raises(
        ValueError, match="Triangle vertex indices must be <= number of vertices"
    ):
        read_tsurf(tsurf_stream(content))


def test_tface_triangles_with_wrong_number_of_vertices(
    signature_line, header_section, coordinate_system_section
):
    """Test TFACE section with wrong number of vertex indices."""
    content = (
        signature_line
        + header_section
        + coordinate_system_section
        + "TFACE\n"
        + "VRTX 1 0.0 0.0 0.0 CNXYZ\n"
        + "VRTX 2 1.0 0.0 0.0 CNXYZ\n"
        + "VRTX 3 0.0 1.0 0.0 CNXYZ\n"
        + "TRGL 1 2\n"  # Triangle with too few vertices
        + "END\n"
    )

    with pytest.raises(ValueError, match="Invalid 'TRGL' line in 'TFACE' section"):
        read_tsurf(tsurf_stream(content))


def test_tface_with_non_integer_triangle_indices(
    signature_line, header_section, coordinate_system_section
):
    """Test TFACE section with non-integer triangle vertex indices."""
    content = (
        signature_line
        + header_section
        + coordinate_system_section
        + "TFACE\n"
        + "VRTX 1 0.0 0.0 0.0 CNXYZ\n"
        + "VRTX 2 1.0 0.0 0.0 CNXYZ\n"
        + "VRTX 3 0.0 1.0 0.0 CNXYZ\n"
        + "TRGL 1 two 3\n"  # Non-integer index
        + "END\n"
    )

    with pytest.raises(ValueError, match="Invalid 'TRGL' line in 'TFACE' section"):
        read_tsurf(tsurf_stream(content))


def test_tface_with_no_vertices_no_triangles(
    signature_line, header_section, coordinate_system_section
):
    """Test TFACE section with no vertices and no triangles."""
    content = (
        signature_line
        + header_section
        + coordinate_system_section
        + "TFACE\n"
        + "END\n"  # No VRTX or TRGL lines
    )

    with pytest.raises(
        ValueError, match="Less than 3 vertices found in TSurf triangulation data"
    ):
        read_tsurf(tsurf_stream(content))


def test_tface_with_triangle_index_less_than_one(
    signature_line, header_section, coordinate_system_section
):
    """Test TFACE section with triangle vertex index less than one."""
    content = (
        signature_line
        + header_section
        + coordinate_system_section
        + "TFACE\n"
        + "VRTX 1 0.0 0.0 0.0 CNXYZ\n"
        + "VRTX 2 1.0 0.0 0.0 CNXYZ\n"
        + "VRTX 3 0.0 1.0 0.0 CNXYZ\n"
        + "TRGL -1 2 3\n"  # Triangle references vertex index -1
        + "END\n"
    )

    with pytest.raises(
        ValueError, match="Triangle vertex indices must be >= 1 in triangulation data."
    ):
        read_tsurf(tsurf_stream(content))


def test_tface_vertices_without_triangles(
    signature_line, header_section, coordinate_system_section
):
    """Test TFACE section with vertices but no triangles."""
    content = (
        signature_line
        + header_section
        + coordinate_system_section
        + "TFACE\n"
        + "VRTX 1 0.0 0.0 0.0 CNXYZ\n"
        + "VRTX 2 1.0 0.0 0.0 CNXYZ\n"
        + "VRTX 3 0.0 1.0 0.0 CNXYZ\n"
        + "END\n"
    )

    with pytest.raises(
        ValueError, match="No triangles found in TSurf triangulation data"
    ):
        read_tsurf(tsurf_stream(content))


def test_tface_triangles_without_vertices(signature_line, header_section):
    """Test TFACE section with triangles but no vertices."""
    content = (
        signature_line
        + header_section
        + "TFACE\n"
        + "TRGL 1 2 3\n"  # No VRTX lines
        + "END\n"
    )

    with pytest.raises(
        ValueError, match="Less than 3 vertices found in TSurf triangulation data"
    ):
        read_tsurf(tsurf_stream(content))


def test_tface_with_no_end_statement_eof(signature_line, header_section):
    """
    Test TFACE section missing END statement,
    with file ending immediately after TFACE data.
    """

    content = (
        signature_line
        + header_section
        + "TFACE\n"
        + "VRTX 1 0.0 0.0 0.0 CNXYZ\n"
        + "VRTX 2 1.0 0.0 0.0 CNXYZ\n"
        + "VRTX 3 0.0 1.0 0.0 CNXYZ\n"
        + "TRGL 1 2 3\n"
        # Missing END statement
    )

    with pytest.raises(
        ValueError, match="Missing 'END' statement at the end of the 'TFACE' section"
    ):
        read_tsurf(tsurf_stream(content))


def test_tface_with_no_end_statement_not_eof(signature_line, header_section):
    """Test TFACE section missing END statement, with more lines after."""
    content = (
        signature_line
        + "TFACE\n"
        + "VRTX 1 0.0 0.0 0.0 CNXYZ\n"
        + "VRTX 2 1.0 0.0 0.0 CNXYZ\n"
        + "VRTX 3 0.0 1.0 0.0 CNXYZ\n"
        + "TRGL 1 2 3\n"
        # Missing END statement
        + header_section
    )

    with pytest.raises(
        ValueError, match="Expect lines to start with 'VRTX', 'TRGL', or 'END'"
    ):
        read_tsurf(tsurf_stream(content))
