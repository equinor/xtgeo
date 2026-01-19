from dataclasses import FrozenInstanceError
from io import BytesIO, StringIO
from pathlib import Path

import numpy as np
import pytest

from xtgeo.io.tsurf._tsurf_reader import (
    TSurfCoordSys,
    TSurfData,
    TSurfHeader,
    ValidatorCoordSys,
)


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
    signature_line: str,
    header_section: str,
    coordinate_system_section: str,
    tface_section: str,
) -> str:
    """Return a complete valid TSurf file with all sections."""
    return signature_line + header_section + coordinate_system_section + tface_section


@pytest.fixture
def minimal_tsurf_file(
    signature_line: str, header_section: str, tface_section: str
) -> str:
    """Return a minimal valid TSurf file (no coordinate system)."""
    return signature_line + header_section + tface_section


@pytest.fixture
def missing_signature_file(
    header_section: str,
    coordinate_system_section: str,
    tface_section: str,
) -> str:
    """Return TSurf file content missing signature line."""
    return header_section + coordinate_system_section + tface_section


@pytest.fixture
def missing_header_file(
    signature_line: str, coordinate_system_section: str, tface_section: str
) -> str:
    """Return TSurf file content missing header section."""
    return signature_line + coordinate_system_section + tface_section


@pytest.fixture
def missing_tface_file(
    signature_line: str, header_section: str, coordinate_system_section: str
) -> str:
    """Return TSurf file content missing TFACE section."""
    return signature_line + header_section + coordinate_system_section


@pytest.fixture
def only_signature_file(signature_line: str) -> str:
    """Return TSurf file content with only signature line."""
    return signature_line


def tsurf_stream(content: str) -> StringIO:
    """Helper function to create StringIO from TSurf content."""
    return StringIO(content)


def test_file_string_input(tmp_path: str, complete_tsurf_file: str) -> None:
    """Test reading from string input."""
    filepath = str(tmp_path) + "/test.ts"
    with open(filepath, "w") as f:
        f.write(complete_tsurf_file)

    result = TSurfData.from_file(filepath)
    assert result is not None


def test_file_unusual_suffix(minimal_tsurf_file: str, tmp_path: Path) -> None:
    """
    Test with unusual file suffix.
    Normally TSurf files have .ts extension, but this is not enforced by the reader.
    """
    filepath = tmp_path / "unusual_suffix.txt"
    with open(filepath, "w") as f:
        f.write(minimal_tsurf_file)
    result_unusual_suffix = TSurfData.from_file(filepath)
    assert result_unusual_suffix is not None


def test_comments_and_empty_lines(tmp_path: Path) -> None:
    """Test handling of comments and empty lines in files."""
    content_lines = [
        "GOCAD TSurf 1",
        "# This is a comment",
        "",  # Empty line
        "HEADER {",
        "name: test_surface",
        "",  # Empty line
        "# Another comment",
        "}",
        "# Another comment",
        "",  # Empty line
        "TFACE",
        "VRTX 1 0.0 0.0 0.0 CNXYZ",
        "",  # Empty line
        "VRTX 2 1.0 0.0 0.0 CNXYZ",
        "# Another comment",
        "VRTX 3 0.0 1.0 0.0 CNXYZ",
        "# Another comment",
        "TRGL 1 2 3",
        "",  # Empty line
        "# Another comment",
        "END",
    ]

    content = "\n".join(content_lines) + "\n"
    filepath = tmp_path / "with_comments.ts"
    with open(filepath, "w") as f:
        f.write(content)

    result = TSurfData.from_file(filepath)
    assert result is not None
    assert result.header.name == "test_surface"


def test_sections_all(complete_tsurf_file: str) -> None:
    """Test a valid TSurf file with all sections."""
    result = TSurfData.from_file(tsurf_stream(complete_tsurf_file))

    assert result is not None
    assert result.header.name == "test_surface"
    assert result.coord_sys is not None
    assert result.coord_sys.name == "Default"
    assert len(result.vertices) == 3
    assert len(result.triangles) == 1


def test_sections_minimal(minimal_tsurf_file: str) -> None:
    """Test valid TSurf file with only mandatory sections."""
    result = TSurfData.from_file(tsurf_stream(minimal_tsurf_file))

    assert result is not None
    assert result.header.name == "test_surface"
    assert result.coord_sys is None
    assert len(result.vertices) == 3
    assert len(result.triangles) == 1


def test_section_missing_signature(missing_signature_file: str) -> None:
    """Test that missing signature line raises appropriate error."""
    with pytest.raises(
        ValueError, match="does not match format detected from file contents"
    ):
        TSurfData.from_file(tsurf_stream(missing_signature_file))


def test_section_missing_header(missing_header_file: str) -> None:
    """Test that missing header section raises appropriate error."""
    with pytest.raises(ValueError, match="Missing mandatory 'HEADER' section"):
        TSurfData.from_file(tsurf_stream(missing_header_file))


def test_section_missing_coordinate_system(minimal_tsurf_file: str) -> None:
    """Test that missing optional coordinate system section is handled"""
    result = TSurfData.from_file(tsurf_stream(minimal_tsurf_file))

    assert result is not None
    assert result.coord_sys is None  # Coordinate system is optional


def test_section_missing_tface(missing_tface_file: str) -> None:
    """Test that missing TFACE section raises appropriate error."""
    with pytest.raises(ValueError, match="Missing mandatory 'TFACE' section"):
        TSurfData.from_file(tsurf_stream(missing_tface_file))


def test_section_invalid_keyword(
    signature_line: str, header_section: str, tface_section: str
) -> None:
    """Test that invalid section keyword raises appropriate error."""
    tsurf_content = (
        signature_line + header_section + "INVALID_SECTION\n" + tface_section
    )

    with pytest.raises(
        ValueError, match="The file contains an invalid line which is not recognized"
    ):
        TSurfData.from_file(tsurf_stream(tsurf_content))


def test_section_header_after_coordinate_system(
    signature_line: str,
    coordinate_system_section: str,
    header_section: str,
    tface_section: str,
) -> None:
    """Test header section appearing after coordinate system section."""
    content = (
        signature_line + coordinate_system_section + header_section + tface_section
    )

    result = TSurfData.from_file(tsurf_stream(content))
    assert result is not None
    assert result.header is not None
    assert result.header.name == "test_surface"
    assert result.coord_sys is not None


def test_section_header_after_tface(
    signature_line: str, tface_section: str, header_section: str
) -> None:
    """Test header section appearing after TFACE section."""
    content = signature_line + tface_section + header_section

    result = TSurfData.from_file(tsurf_stream(content))
    assert result is not None
    assert result.header is not None
    assert result.header.name == "test_surface"
    assert result.vertices is not None
    assert result.triangles is not None


def test_section_tface_appearing_twice(
    signature_line: str, header_section: str, tface_section: str
) -> None:
    """Test TFACE section appearing twice."""
    content = signature_line + header_section + tface_section + tface_section

    with pytest.raises(ValueError, match="Multiple 'TFACE' sections found"):
        TSurfData.from_file(tsurf_stream(content))


def test_signature_only(only_signature_file: str) -> None:
    """Test file with only signature line."""
    with pytest.raises(ValueError, match="Missing mandatory 'HEADER' section"):
        TSurfData.from_file(tsurf_stream(only_signature_file))


def test_signature_multiple_header_between(
    signature_line: str, header_section: str, tface_section: str
) -> None:
    """
    Test for multiple signature lines.
    This test places the second signature line after a HEADER section.
    """
    tsurf_content = signature_line + header_section + signature_line + tface_section

    with pytest.raises(
        ValueError, match="The file contains an invalid line which is not recognized"
    ):
        TSurfData.from_file(tsurf_stream(tsurf_content))


def test_signature_invalid_line(header_section: str, tface_section: str) -> None:
    """Test invalid signature line variations."""
    invalid_signatures = [
        "GOCAD TSurf 2\n",  # Wrong version
        "Invalid signature\n",  # Just wrong
    ]

    for invalid_sig in invalid_signatures:
        content = invalid_sig + header_section + tface_section

        with pytest.raises(
            ValueError, match="does not match format detected from file contents"
        ):
            TSurfData.from_file(tsurf_stream(content))


def test_header_with_different_names(signature_line: str, tface_section: str) -> None:
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

        result = TSurfData.from_file(tsurf_stream(content))
        assert result.header.name[0] == name.strip()[0]


def test_header_name_format_empty(signature_line: str, tface_section: str) -> None:
    """Test header name format: empty name"""
    content_empty_name = signature_line + "HEADER {\nname: \n}\n" + tface_section
    with pytest.raises(
        ValueError, match="Missing or invalid name in the 'HEADER' section"
    ):
        TSurfData.from_file(tsurf_stream(content_empty_name))


def test_header_name_format_no_space_after_colon(
    signature_line: str, tface_section: str
) -> None:
    """Test header name format: no space after colon"""
    content_no_space = signature_line + "HEADER {\nname:F5\n}\n" + tface_section
    result = TSurfData.from_file(tsurf_stream(content_no_space))
    assert result.header.name == "F5"


def test_header_with_invalid_line(signature_line: str, tface_section: str) -> None:
    """Test header section with invalid line."""
    malformed_header = "HEADER {\ninvalid_line_here\n}\n"

    content = signature_line + malformed_header + tface_section

    with pytest.raises(ValueError, match="Invalid 'HEADER' section line:"):
        TSurfData.from_file(tsurf_stream(content))


def test_header_only_header_line(signature_line: str, tface_section: str) -> None:
    """Test header section that opens but has no content and no closing."""
    content = signature_line + "HEADER {\n" + tface_section

    with pytest.raises(ValueError, match="Invalid 'HEADER' section line"):
        TSurfData.from_file(tsurf_stream(content))


def test_header_with_no_closing_brace(signature_line: str, tface_section: str) -> None:
    """Test header section missing closing brace."""
    content = (
        signature_line
        + "HEADER {\n"
        + "name: test_surface\n"
        # Missing closing brace
        + tface_section
    )

    with pytest.raises(ValueError, match="Invalid 'HEADER' section line"):
        TSurfData.from_file(tsurf_stream(content))


def test_header_appearing_twice(
    signature_line: str, header_section: str, tface_section: str
) -> None:
    """Test header section appearing twice."""
    another_header = "HEADER {\nname: another_surface\n}\n"
    content = signature_line + header_section + another_header + tface_section

    with pytest.raises(ValueError, match="Multiple 'HEADER' sections found"):
        TSurfData.from_file(tsurf_stream(content))


def test_header_incomplete_eof(signature_line: str) -> None:
    """Test header section that opens but has no content."""
    content = (
        signature_line + "HEADER {\n"
        # File ends abruptly here - no closing tag, no content
    )

    with pytest.raises(
        ValueError, match="Missing '}' at the end of the 'HEADER' section"
    ):
        TSurfData.from_file(tsurf_stream(content))


def test_header_immutability_cannot_delete_attribute() -> None:
    """Test that attributes cannot be deleted from frozen dataclass."""
    header = TSurfHeader(name="TestSurface")

    with pytest.raises(FrozenInstanceError):
        del header.name


def test_header_validator_none_name_value() -> None:
    """Test that None name value raises ValueError."""
    data = {"name": None}
    with pytest.raises(ValueError, match="Missing or invalid name"):
        TSurfHeader.validate(data, "test_file.ts")


def test_coordinate_system_incomplete_eof(
    signature_line: str, header_section: str
) -> None:
    """Test coordinate system section that opens but has no content."""
    content = (
        signature_line + header_section + "GOCAD_ORIGINAL_COORDINATE_SYSTEM\n"
        # File ends abruptly here - no closing tag, no content
    )

    with pytest.raises(
        ValueError, match="Missing 'END_ORIGINAL_COORDINATE_SYSTEM' statement"
    ):
        TSurfData.from_file(tsurf_stream(content))


def test_coordinate_system_with_invalid_lines(
    signature_line: str, header_section: str, tface_section: str
) -> None:
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
        TSurfData.from_file(tsurf_stream(content))


def test_coordinate_system_missing_fields(
    signature_line: str, header_section: str, tface_section: str
) -> None:
    """Test coordinate system section missing required fields."""
    incomplete_coord_sys = (
        "GOCAD_ORIGINAL_COORDINATE_SYSTEM\n"
        "NAME Default\n"
        # Missing AXIS_NAME, AXIS_UNIT, ZPOSITIVE
        "END_ORIGINAL_COORDINATE_SYSTEM\n"
    )

    content = signature_line + header_section + incomplete_coord_sys + tface_section

    with pytest.raises(ValueError, match="Coordinate system section missing fields"):
        TSurfData.from_file(tsurf_stream(content))


def test_coordinate_system_with_extra_fields(
    signature_line: str, header_section: str, tface_section: str
) -> None:
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
        TSurfData.from_file(tsurf_stream(content))


def test_coordinate_system_appearing_twice(
    signature_line: str,
    header_section: str,
    coordinate_system_section: str,
    tface_section: str,
) -> None:
    """Test coordinate system section appearing twice."""
    content = (
        signature_line
        + header_section
        + coordinate_system_section
        + tface_section
        + coordinate_system_section
    )

    with pytest.raises(ValueError, match="Multiple 'COORDINATE_SYSTEM' sections found"):
        TSurfData.from_file(tsurf_stream(content))


def test_coordinate_system_with_invalid_axis_names(
    signature_line: str, header_section: str, tface_section: str
) -> None:
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
        TSurfData.from_file(tsurf_stream(content))


def test_coordinate_system_axis_names() -> None:
    """Test axis names validation."""

    # Valid axis names
    ValidatorCoordSys._validate_axis_names(("X", "Y", "Z"), "test_file")

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

    # Invalid number of units
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


def test_coordinate_system_axis_elements_empty_string() -> None:
    """Test that empty strings in axis elements raise ValueError."""

    # Test empty string in axis names
    with pytest.raises(ValueError, match="must have exactly three values"):
        ValidatorCoordSys._validate_axis_elements(
            ("X", "", "Z"),  # Empty string in axis names
            ValidatorCoordSys.common_axis_names,
            "AXIS_NAME",
            "test_file",
            check_uniqueness=True,
        )

    # Test empty string in axis units
    with pytest.raises(ValueError, match="must have exactly three values"):
        ValidatorCoordSys._validate_axis_elements(
            ("m", "", "m"),  # empty string in axis units
            ValidatorCoordSys.common_axis_units,
            "AXIS_UNIT",
            "test_file",
            check_uniqueness=False,
        )


def test_coordinate_system_validator_missing_field_delegated() -> None:
    """Test that missing field errors are properly delegated."""
    data = {
        "name": "Test",
        "axis_name": ["X", "Y", "Z"],
        # Missing axis_unit and zpositive
    }
    with pytest.raises(ValueError, match="missing fields"):
        TSurfCoordSys.validate(data, "test_file.ts")


def test_coordinate_system_immutability_cannot_delete_attribute() -> None:
    """Test that attributes cannot be deleted from frozen dataclass."""
    coord_sys = TSurfCoordSys(
        name="TestCoordSys",
        axis_name=("X", "Y", "Z"),
        axis_unit=("m", "m", "m"),
        zpositive="Depth",
    )

    with pytest.raises(FrozenInstanceError):
        del coord_sys.name


def test_axis_elements_uniqueness_validation() -> None:
    """Test that non-unique axis elements raise ValueError when uniqueness required."""

    # Test duplicate axis names (uniqueness required)
    with pytest.raises(ValueError, match="values \\(in lowercase\\) must be unique"):
        ValidatorCoordSys._validate_axis_elements(
            ("X", "Y", "X"),
            [(e.lower() for e in ValidatorCoordSys.common_axis_names[0])],
            "AXIS_NAME",
            "test_file",
            check_uniqueness=True,
        )

    # Test duplicate axis names with different cases (should still fail)
    with pytest.raises(ValueError, match="values \\(in lowercase\\) must be unique"):
        ValidatorCoordSys._validate_axis_elements(
            ("x", "Y", "X"),  # Lowercase 'x' and uppercase 'X' are considered the same
            [(e.lower() for e in ValidatorCoordSys.common_axis_names[0])],
            "AXIS_NAME",
            "test_file",
            check_uniqueness=True,
        )

    # Test duplicate axis units with uniqueness disabled (should NOT raise error)
    # This should work fine since check_uniqueness=False for units
    ValidatorCoordSys._validate_axis_elements(
        ("m", "m", "m"),
        [(e.lower() for e in ValidatorCoordSys.common_axis_units[0])],
        "AXIS_UNIT",
        "test_file",
        check_uniqueness=False,
    )


def test_tface_with_slightly_more_complex_geometry(
    signature_line: str,
    header_section: str,
) -> None:
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

    result = TSurfData.from_file(tsurf_stream(content))
    assert len(result.vertices) == 4
    assert len(result.triangles) == 2


def test_tface_missing_vertex_coordinate(
    signature_line: str, header_section: str
) -> None:
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
        TSurfData.from_file(tsurf_stream(content))


def test_tface_invalid_vertex_coordinates(
    signature_line: str, header_section: str
) -> None:
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
        TSurfData.from_file(tsurf_stream(content))


def test_tface_invalid_vertex_attribute(
    signature_line: str, header_section: str
) -> None:
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
        TSurfData.from_file(tsurf_stream(content))


def test_tface_invalid_vertex_numbering(
    signature_line: str, header_section: str
) -> None:
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
        TSurfData.from_file(tsurf_stream(content))


def test_tface_triangles_with_non_existent_vertices(
    signature_line: str, header_section: str, coordinate_system_section: str
) -> None:
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
        TSurfData.from_file(tsurf_stream(content))


def test_tface_triangles_with_wrong_number_of_vertices(
    signature_line: str, header_section: str, coordinate_system_section: str
) -> None:
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
        TSurfData.from_file(tsurf_stream(content))


def test_tface_with_non_integer_triangle_indices(
    signature_line: str, header_section: str, coordinate_system_section: str
) -> None:
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
        TSurfData.from_file(tsurf_stream(content))


def test_tface_with_no_vertices_no_triangles(
    signature_line: str, header_section: str, coordinate_system_section: str
) -> None:
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
        TSurfData.from_file(tsurf_stream(content))


def test_tface_with_triangle_index_less_than_one(
    signature_line: str, header_section: str, coordinate_system_section: str
) -> None:
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
        TSurfData.from_file(tsurf_stream(content))


def test_tface_vertices_without_triangles(
    signature_line: str, header_section: str, coordinate_system_section: str
) -> None:
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
        TSurfData.from_file(tsurf_stream(content))


def test_tface_triangles_without_vertices(
    signature_line: str, header_section: str
) -> None:
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
        TSurfData.from_file(tsurf_stream(content))


def test_tface_with_no_end_statement_eof(
    signature_line: str, header_section: str
) -> None:
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
        TSurfData.from_file(tsurf_stream(content))


def test_tface_with_no_end_statement_not_eof(
    signature_line: str, header_section: str
) -> None:
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
        TSurfData.from_file(tsurf_stream(content))


def test_tsurfdata_immutability_cannot_modify_header_attribute() -> None:
    """Test that header attribute cannot be modified after creation."""
    header = TSurfHeader(name="TestSurface")
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64
    )
    triangles = np.array([[1, 2, 3]], dtype=np.int64)

    data = TSurfData(
        header=header,
        coord_sys=None,
        vertices=vertices,
        triangles=triangles,
    )

    new_header = TSurfHeader(name="ModifiedSurface")
    with pytest.raises(FrozenInstanceError):
        data.header = new_header


def test_tsurfdata_immutability_vertices_array_content_can_be_modified() -> None:
    """
    Test that while the vertices attribute is frozen, the numpy array content can
    still be modified.

    Note: This is a known limitation of frozen dataclasses with mutable objects.
    The attribute reference is frozen, but the array content is not.
    """

    header = TSurfHeader(name="TestSurface")
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64
    )
    triangles = np.array([[1, 2, 3]], dtype=np.int64)

    data = TSurfData(
        header=header,
        coord_sys=None,
        vertices=vertices,
        triangles=triangles,
    )

    # This should succeed - array content can be modified
    original_value = data.vertices[0, 0]
    data.vertices[0, 0] = 999.0
    assert data.vertices[0, 0] == 999.0
    assert data.vertices[0, 0] != original_value


def test_tsurfdata_get_vertices_return_correct_array() -> None:
    """Test that get_vertices returns the correct numpy array."""
    header = TSurfHeader(name="TestSurface")
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    triangles = np.array([[1, 2, 3]], dtype=np.int64)

    data = TSurfData(
        header=header,
        coord_sys=None,
        vertices=vertices,
        triangles=triangles,
    )

    result = data.get_vertices
    np.testing.assert_array_equal(result, vertices)
    assert result.dtype == np.float64


def test_tsurfdata_get_cells_return_correct_array() -> None:
    """Test that get_cells returns the correct numpy array."""
    header = TSurfHeader(name="TestSurface")
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    triangles = np.array([[1, 2, 3]], dtype=np.int64)

    data = TSurfData(
        header=header,
        coord_sys=None,
        vertices=vertices,
        triangles=triangles,
    )

    result = data.get_cells
    np.testing.assert_array_equal(result, triangles)
    assert result.dtype == np.int64


def test_tsurfdata_get_vertices_return_same_reference() -> None:
    """Test that get_vertices returns the same array reference."""
    header = TSurfHeader(name="TestSurface")
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    triangles = np.array([[1, 2, 3]], dtype=np.int64)

    data = TSurfData(
        header=header,
        coord_sys=None,
        vertices=vertices,
        triangles=triangles,
    )

    assert data.get_vertices is data.vertices


def test_tsurf_data_roundtrip_write_to_file(
    complete_tsurf_file: str, tmp_path: Path
) -> None:
    """Test writing TSurfData."""
    result = TSurfData.from_file(tsurf_stream(complete_tsurf_file))
    assert result is not None

    output_filepath = tmp_path / "output.ts"
    result.to_file(output_filepath)

    result_written = TSurfData.from_file(output_filepath)
    assert result_written is not None

    assert result.header == result_written.header
    assert result.coord_sys == result_written.coord_sys
    assert np.array_equal(result.vertices, result_written.vertices)
    assert np.array_equal(result.triangles, result_written.triangles)


def test_tsurf_data_roundtrip_string_io(complete_tsurf_file: str) -> None:
    """Test writing and reading TSurfData using StringIO."""
    result = TSurfData.from_file(tsurf_stream(complete_tsurf_file))
    assert result is not None

    output_stream = StringIO()
    result.to_file(output_stream)

    output_stream.seek(0)
    result_written = TSurfData.from_file(output_stream)
    assert result_written is not None

    assert result.header == result_written.header
    assert result.coord_sys == result_written.coord_sys
    assert np.array_equal(result.vertices, result_written.vertices)
    assert np.array_equal(result.triangles, result_written.triangles)


def test_tsurf_data_roundtrip_bytes_io(complete_tsurf_file: str) -> None:
    """Test writing and reading TSurfData using BytesIO."""
    result = TSurfData.from_file(tsurf_stream(complete_tsurf_file))
    assert result is not None

    output_stream = BytesIO()
    result.to_file(output_stream)

    output_stream.seek(0)
    result_written = TSurfData.from_file(output_stream)
    assert result_written is not None

    assert result.header == result_written.header
    assert result.coord_sys == result_written.coord_sys
    assert np.array_equal(result.vertices, result_written.vertices)
    assert np.array_equal(result.triangles, result_written.triangles)

    # TODO: see dataio writer tests
    # TODO: test with invalid filepaths/streams, non-existing folders, etc.
    # TODO: test with a couple of different encodings
    # TODO: write erroneous files and check errors
    # TODO: dataio: test_tsurf_reader_invalid_lines()
