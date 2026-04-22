from __future__ import annotations

import shutil
from dataclasses import FrozenInstanceError
from io import BytesIO, StringIO
from pathlib import Path
from typing import IO, TYPE_CHECKING

import numpy as np
import pytest

import xtgeo
from xtgeo.io._file import FileWrapper
from xtgeo.io.gxf._gxf_io import GXFData
from xtgeo.surface import _regsurf_export
from xtgeo.surface._regsurf_import import import_gxf

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.fixture(scope="session")
def gxf_path(testdata_path: str) -> Path:
    """Return the path to a GXF test file."""
    p = Path(testdata_path) / "surfaces/etc/fdata_test.gxf"
    if not p.exists():
        pytest.skip(f"Test data file not found: {p}")  # pragma: no cover
    return p


@pytest.fixture(scope="session")
def gxf_data_from_file(gxf_path: Path) -> GXFData:
    """Return parsed GXF data from a GXF test file."""

    return GXFData.from_file(gxf_path)


@pytest.fixture(scope="session")
def regular_surface_from_gxf_file(gxf_path: Path) -> xtgeo.RegularSurface:
    """Return a RegularSurface imported from a GXF test file."""

    return xtgeo.surface_from_file(gxf_path)


def gxf_stream(content: str) -> StringIO:
    return StringIO(content)


def valid_gxfdata_args(**overrides: object) -> dict[str, object]:
    """Return a dictionary of valid GXFData constructor arguments.

    Args:
        **overrides: Constructor arguments to override in the default values.
    """
    args: dict[str, object] = {
        "points": 2,
        "rows": 2,
        "ptseparation": 1.0,
        "rwseparation": 1.0,
        "xorigin": 0.0,
        "yorigin": 0.0,
        "rotation": 0.0,
        "dummy": -999.0,
        "grid": np.ma.array([[1.0, 2.0], [3.0, 4.0]]),
    }
    args.update(overrides)
    return args


def make_gxf_data(**overrides: object) -> GXFData:
    """
    Return a GXFData instance with valid default arguments,
    overridden by any provided arguments.

    Args:
        **overrides: GXFData constructor arguments to override in the defaults.
    """
    return GXFData(**valid_gxfdata_args(**overrides))


def make_regular_surface(**overrides: object) -> xtgeo.RegularSurface:
    """Return a RegularSurface instance with valid default arguments,
    overridden by any provided arguments.

    Args:
        **overrides: RegularSurface constructor arguments to override in the defaults.
    """
    args: dict[str, object] = {
        "ncol": 3,
        "nrow": 2,
        "xinc": 10.0,
        "yinc": 20.0,
        "xori": 100.0,
        "yori": 200.0,
        "rotation": 15.0,
        "values": np.ma.array(
            [[11.0, 44.0], [22.0, 55.0], [33.0, 66.0]],
            mask=False,
        ),
    }
    args.update(overrides)
    return xtgeo.RegularSurface(**args)


@pytest.fixture
def gxf_producer_long_preamble() -> str:
    """Return GXF content with comments and free text before the first key."""

    return """! Example producer banner line 01
! Example producer banner line 02
! Example producer banner line 03
! Example producer banner line 04
! Example producer banner line 05
! Example producer banner line 06
! Example producer banner line 07
! Example producer banner line 08
! Example producer banner line 09
! Example producer banner line 10
! Example producer banner line 11
! Example producer banner line 12
! Example producer banner line 13
! Example producer banner line 14
! Example producer banner line 15
! Example producer banner line 16
! Example producer banner line 17
! Example producer banner line 18
! Example producer banner line 19
! Example producer banner line 20
Free text producer note before the first GXF key.
Another free text line that should not affect parsing.

#POINTS
2
#ROWS
2
#PTSEPARATION
25.0
#RWSEPARATION
-50.0
#XORIGIN
123.0
#YORIGIN
456.0
#ROTATION
370.0
#GRID
1 2
3 4
"""


@pytest.fixture
def gxf_double_hash_keys() -> str:
    """
    Return GXF content with double-hash keys.
    This fixture is used to test that extension keys are ignored without warnings.
    """

    return """! Producer with double-hash extension metadata.

#POINTS
2
#ROWS
2
#PTSEPARATION
10.0
#RWSEPARATION
20.0
#XORIGIN
1000.0
#YORIGIN
2000.0
#ROTATION
12.5
#DUMMY
-999.0
##PRODUCER
7.4
##EXPORT_NOTE
1
#GRID
10 -999
30 40
"""


@pytest.fixture
def gxf_content_minimal() -> str:
    """Return minimal GXF content containing only mandatory keys."""

    return """! Producer that emits only mandatory GXF keys.

#POINTS
3
#ROWS
2
#GRID
1 2 3
4 5 6
"""


@pytest.fixture
def valid_gxf_content() -> str:
    """Return valid GXF content with comments, extension keys, and dummy values."""

    return """! This is a comment line

Some free text to ignore

#POINTS
"3"
#ROWS
"2"
#PTSEPARATION
"30.0"
#RWSEPARATION
"40.0"
#XORIGIN
"1000.0"
#YORIGIN
"2000.0"
#ROTATION
"12.5"
#DUMMY
"9999999.0"
! This is another comment line
##XMAX
"9999.0"
##YMAX
"8888.0"
#GRID
1.0 2.0 3.0
! This is a comment inside the grid section
4.0 9999999.0 6.0
"""


@pytest.fixture
def gxf_content_with_unknown_key() -> str:
    """Return GXF content with an unknown single-hash key before #GRID."""

    return """
#POINTS
"3"
#ROWS
"2"
#PTSEPARATION
"1"
#RWSEPARATION
"1"
#XORIGIN
"0"
#YORIGIN
"0"
#ROTATION
"0"
#DUMMY
"999"
#UNKNOWN_KEY
"17"
#GRID
1 2 3 4 5 6
"""


@pytest.fixture
def gxf_content_missing_rows() -> str:
    """Return GXF content that omits the mandatory #ROWS key."""

    return """
#POINTS
"2"
! #ROWS intentionally omitted
#PTSEPARATION
"1"
#RWSEPARATION
"1"
#XORIGIN
"0"
#YORIGIN
"0"
#ROTATION
"0"
#DUMMY
"999"
#GRID
1 2 3 4
"""


@pytest.fixture
def non_gxf_content() -> str:
    """Return non-GXF text used to exercise format validation failures."""

    return """
This is not a GXF key
"""


@pytest.fixture
def gxf_content_small_grid() -> str:
    """Return a valid 2x2 GXF document for extension-based format guessing."""

    return """
#POINTS
"2"
#ROWS
"2"
#PTSEPARATION
"1"
#RWSEPARATION
"1"
#XORIGIN
"10"
#YORIGIN
"20"
#ROTATION
"0"
#DUMMY
"9999"
#GRID
1 2
3 9999
"""


@pytest.fixture
def gxf_content_no_dummy_with_default_value() -> str:
    """Return GXF content where 9999999.0 is valid because #DUMMY is absent."""

    return """
#POINTS
2
#ROWS
2
#PTSEPARATION
1
#RWSEPARATION
1
#XORIGIN
0
#YORIGIN
0
#ROTATION
0
#GRID
1 2 9999999.0 4
"""


@pytest.fixture
def gxf_content_with_grid_value_count() -> Callable[[int], str]:
    """Return a factory for GXF content with a chosen number of grid values."""

    def make_content(count: int) -> str:
        """Return GXF content whose #GRID section contains count values."""

        values = " ".join(str(i) for i in range(count))
        return f"""
#POINTS
"3"
#ROWS
"2"
#PTSEPARATION
"1"
#RWSEPARATION
"1"
#XORIGIN
"0"
#YORIGIN
"0"
#ROTATION
"0"
#DUMMY
"999"
#GRID
{values}
"""

    return make_content


@pytest.fixture
def gxf_content_missing_mandatory_key() -> Callable[[str], str]:
    """Return a factory for GXF content with one mandatory key omitted."""

    contents = {
        "POINTS": """
#ROWS
"2"
#GRID
1 2 3 4
""",
        "ROWS": """
#POINTS
"2"
#GRID
1 2 3 4
""",
        "GRID": """
#POINTS
"3"
#ROWS
"2"
#PTSEPARATION
"1"
#RWSEPARATION
"1"
#XORIGIN
"0"
#YORIGIN
"0"
#ROTATION
"0"
#DUMMY
"999"
""",
    }

    def get_content(missing_key: str) -> str:
        """Return content where missing_key is the omitted mandatory key."""

        return contents[missing_key]

    return get_content


@pytest.fixture
def gxf_content_with_gtype_positive() -> str:
    """Return GXF content with positive #GTYPE and compressed grid payload."""

    return """
#POINTS
"2"
#ROWS
"2"
#GTYPE
"3"
#GRID
(L2(/.()H)0@*&,
"""


@pytest.fixture
def gxf_content_with_gtype_zero() -> str:
    """Return GXF content with #GTYPE zero and uncompressed grid values."""

    return """
#POINTS
"2"
#ROWS
"2"
#GTYPE
"0"
#GRID
1 2 3 4
"""


@pytest.fixture
def gxf_content_with_extension_value() -> Callable[[str], str]:
    """Return a factory for GXF content with a chosen ##XMAX value."""

    def make_content(extension_value: str) -> str:
        """Return content whose ##XMAX value is extension_value."""

        return f"""
#POINTS
"2"
#ROWS
"2"
##XMAX
{extension_value}
#GRID
1 2 3 4
"""

    return make_content


@pytest.fixture
def gxf_content_with_ignored_key_before_grid() -> Callable[[str], str]:
    """Return a factory for content with an ignored key before #GRID."""

    def make_content(ignored_key: str) -> str:
        """Return content where ignored_key has no value before #GRID."""

        return f"""
#POINTS
"2"
#ROWS
"2"
{ignored_key}
#GRID
1 2 3 4
"""

    return make_content


@pytest.fixture
def gxf_content_with_ignored_key_at_eof() -> Callable[[str], str]:
    """Return a factory for content ending with an ignored key."""

    def make_content(ignored_key: str) -> str:
        """Return content where ignored_key appears without a value at EOF."""

        return f"""
#POINTS
"2"
#ROWS
"2"
{ignored_key}
"""

    return make_content


@pytest.fixture
def gxf_content_with_duplicate_scalar_key() -> Callable[[str], str]:
    """Return a factory for content with one scalar key duplicated."""

    def make_content(duplicate_key: str) -> str:
        """Return content where duplicate_key appears twice before #GRID."""

        scalar_values = {
            "POINTS": "3",
            "ROWS": "2",
            "GTYPE": "0",
            "PTSEPARATION": "1",
            "RWSEPARATION": "1",
            "XORIGIN": "0",
            "YORIGIN": "0",
            "ROTATION": "0",
            "DUMMY": "999",
        }
        lines = []
        for key, value in scalar_values.items():
            lines.extend([f"#{key}", f'"{value}"'])
            if key == duplicate_key:
                lines.extend([f"#{key}", f'"{value}"'])
        return "\n".join(["", *lines, "#GRID", "1 2 3 4 5 6", ""])

    return make_content


@pytest.fixture
def gxf_content_with_key_inside_grid() -> Callable[[str], str]:
    """Return a factory for content with a chosen body inside #GRID."""

    def make_content(grid_body: str) -> str:
        """Return content where grid_body is inserted after the #GRID key."""

        return f"""
#POINTS
"2"
#ROWS
"2"
#PTSEPARATION
"1"
#RWSEPARATION
"1"
#XORIGIN
"0"
#YORIGIN
"0"
#ROTATION
"0"
#DUMMY
"999"
#GRID
{grid_body}
"""

    return make_content


@pytest.fixture
def gxf_content_with_comment_inside_grid() -> str:
    """Return GXF content with a comment line inside the #GRID section."""

    return """
#POINTS
"2"
#ROWS
"2"
#PTSEPARATION
"1"
#RWSEPARATION
"1"
#XORIGIN
"0"
#YORIGIN
"0"
#ROTATION
"0"
#DUMMY
"999"
#GRID
1 2
! This is a comment inside the grid section
3 4
"""


@pytest.fixture
def gxf_content_latin1_comments_and_free_text() -> str:
    """Return GXF content containing Latin-1-only comment and free text."""

    return """! Latin-1 producer comment with Gr\u00f8d
Free text with more non-ASCII: \u00e6\u00f8\u00e5

#POINTS
2
#ROWS
2
#GRID
1 2 3 4
"""


@pytest.fixture
def gxf_content_with_rotation() -> Callable[[float], str]:
    """Return a factory for GXF content with a chosen #ROTATION value."""

    def make_content(file_rotation: float) -> str:
        """Return content whose #ROTATION value is file_rotation."""

        return f"""
#POINTS
2
#ROWS
2
#PTSEPARATION
1
#RWSEPARATION
1
#ROTATION
{file_rotation}
#GRID
1 2 3 4
"""

    return make_content


@pytest.fixture
def gxf_content_with_rwseparation() -> Callable[[float], str]:
    """Return a factory for GXF content with a chosen #RWSEPARATION value."""

    def make_content(rwseparation: float) -> str:
        """Return content whose #RWSEPARATION value is rwseparation."""

        return f"""
#POINTS
2
#ROWS
2
#PTSEPARATION
1
#RWSEPARATION
{rwseparation}
#GRID
1 2 3 4
"""

    return make_content


@pytest.fixture
def gxf_content_with_dummy_value() -> Callable[[str, str], str]:
    """Return a factory for GXF content with chosen #DUMMY and grid values."""

    def make_content(dummy_literal: str, grid_values: str) -> str:
        """Return content using dummy_literal in #DUMMY and grid_values in #GRID."""

        return f"""
#POINTS
2
#ROWS
2
#PTSEPARATION
1
#RWSEPARATION
1
#XORIGIN
0
#YORIGIN
0
#ROTATION
0
#DUMMY
{dummy_literal}
#GRID
{grid_values}
"""

    return make_content


@pytest.fixture
def gxf_content_with_grid_token() -> Callable[[object], str]:
    """Return a factory for GXF content with a chosen token inside #GRID."""

    def make_content(grid_token: object) -> str:
        """Return content where grid_token is placed among grid values."""

        return f"""
#POINTS
4
#ROWS
1
#GRID
1 {grid_token} 2 3
"""

    return make_content


@pytest.fixture
def gxf_content_bad_grid_value() -> str:
    """Return GXF content with a non-numeric token inside #GRID."""

    return """
#POINTS
2
#ROWS
2
#GRID
1 abc 3 4
"""


@pytest.fixture
def gxf_content_invalid_points_value() -> str:
    """Return GXF content with a non-numeric #POINTS scalar value."""

    return """
#POINTS
abc
#ROWS
2
#GRID
1 2 3 4
"""


@pytest.fixture
def gxf_content_with_non_base10_scalar() -> Callable[[str, str], str]:
    """Return a factory for content with a chosen invalid scalar value."""

    def make_content(key: str, value: str) -> str:
        """Return content where #key is assigned value before #GRID."""

        return f"""
#POINTS
2
#ROWS
2
#{key}
{value}
#GRID
1 2 3 4
"""

    return make_content


@pytest.fixture
def gxf_content_with_overflowing_rotation() -> str:
    """Return GXF content with a non-finite #ROTATION value."""

    return """\
    #POINTS
    2
    #ROWS
    2
    #ROTATION
    1e999
    #GRID
    1 2 3 4
    """


@pytest.fixture
def gxf_content_with_key_at_eof() -> str:
    """Return GXF content ending immediately after the #ROWS key."""

    return """
#POINTS
2
#ROWS
"""


@pytest.fixture
def gxf_content_with_adjacent_keys() -> str:
    """Return GXF content where #POINTS is followed by another key."""

    return """
#POINTS
#ROWS
2
#GRID
1 2
"""


@pytest.fixture
def gxf_content_empty() -> str:
    """Return empty GXF content for missing-mandatory-key validation."""

    return ""


@pytest.fixture
def gxf_content_only_comments() -> str:
    """Return GXF content containing only comment lines."""

    return """! comment 1
! comment 2
! comment 3
"""


@pytest.fixture
def gxf_content_with_float_points() -> str:
    """Return GXF content with a float value for integer key #POINTS."""

    return """
#POINTS
3.5
#ROWS
2
#GRID
1 2 3 4 5 6
"""


@pytest.fixture
def gxf_content_with_scientific_notation() -> str:
    """Return valid GXF content whose scalar values use scientific notation."""

    return """
#POINTS
"2"
#ROWS
"2"
#PTSEPARATION
"1.25e1"
#RWSEPARATION
"2.5e1"
#XORIGIN
"-4.27391726575e5"
#YORIGIN
"7.250373922731e6"
#ROTATION
"12.3456789"
#DUMMY
"-9.999e3"
#GRID
1 -9.999e3 3 4
"""


@pytest.fixture
def gxf_content_with_scalar_trailing_token() -> Callable[[str], str]:
    """Return a factory for content where a scalar key has extra tokens."""

    def make_content(key: str) -> str:
        """Return content where #key has a value plus trailing annotation."""

        return f"""
#POINTS
2
#ROWS
2
#{key}
1 trailing annotation
#GRID
1 2 3 4
"""

    return make_content


@pytest.fixture
def gxf_content_with_sense_value() -> Callable[[str], str]:
    """Return a factory for content with a chosen #SENSE value."""

    def make_content(value: str) -> str:
        """Return content whose #SENSE value is value."""

        return f"""
#POINTS
2
#ROWS
2
#SENSE
{value}
#GRID
1 2 3 4
"""

    return make_content


@pytest.fixture
def gxf_content_with_sense_tail() -> Callable[[str], str]:
    """Return a factory for content following #SENSE with a chosen tail."""

    def make_content(tail: str) -> str:
        """Return content where tail follows #SENSE instead of a valid value."""

        return f"""
#POINTS
2
#ROWS
2
#SENSE
{tail}
"""

    return make_content


@pytest.fixture
def gxf_content_with_invalid_key_case() -> Callable[[str], str]:
    """Return a factory for content containing a chosen non-uppercase key."""

    def make_content(invalid_key: str) -> str:
        """Return content where invalid_key appears before grid-like values."""

        return """
#POINTS
2
#ROWS
2
#PTSEPARATION
3.0
#RWSEPARATION
4.0
#XORIGIN
-10.0
#YORIGIN
-20.0
#ROTATION
33.5
#DUMMY
-999
{invalid_key}
1 2 3 -999
""".format(invalid_key=invalid_key)

    return make_content


class TestGXFParsing:
    """Tests for reading and parsing GXF content."""

    def test_valid_with_extension_keys(self, valid_gxf_content: str) -> None:
        """Valid GXF content with extension keys should parse all core values."""
        result = GXFData.from_file(gxf_stream(valid_gxf_content))

        assert result.points == 3
        assert result.rows == 2
        assert result.ptseparation == pytest.approx(30.0)
        assert result.rwseparation == pytest.approx(40.0)
        assert result.xorigin == pytest.approx(1000.0)
        assert result.yorigin == pytest.approx(2000.0)
        assert result.rotation == pytest.approx(12.5)
        assert result.dummy == pytest.approx(9999999.0)

        expected_data = np.array([[1.0, 2.0, 3.0], [4.0, 9999999.0, 6.0]])
        np.testing.assert_allclose(result.grid.data, expected_data)

        expected_mask = np.array([[False, False, False], [False, True, False]])
        np.testing.assert_array_equal(result.grid.mask, expected_mask)

    def test_from_file_path_input(self, tmp_path: Path, valid_gxf_content: str) -> None:
        """GXFData.from_file should accept filesystem paths."""
        path = tmp_path / "surface.gxf"
        path.write_text(valid_gxf_content)

        result = GXFData.from_file(path)

        assert result.points == 3

    def test_nonexistent_file_raises(self) -> None:
        """Reading a missing GXF path should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="does not exist"):
            GXFData.from_file("this_file_does_not_exist.gxf")

    @pytest.mark.parametrize("count", [5, 7])
    def test_grid_value_count_must_match_ncol_times_nrow(
        self, count: int, gxf_content_with_grid_value_count: Callable[[int], str]
    ) -> None:
        """The grid value count must match the declared dimensions."""
        content = gxf_content_with_grid_value_count(count)

        with pytest.raises(ValueError, match="Number of values in #GRID section"):
            GXFData.from_file(gxf_stream(content))

    @pytest.mark.parametrize(
        "missing_key",
        ["POINTS", "ROWS", "GRID"],
    )
    def test_missing_mandatory_key_raises(
        self,
        missing_key: str,
        gxf_content_missing_mandatory_key: Callable[[str], str],
    ) -> None:
        """Missing mandatory GXF keys must raise a clear parser error."""
        content = gxf_content_missing_mandatory_key(missing_key)

        with pytest.raises(ValueError, match=f"Missing mandatory.*{missing_key}"):
            GXFData.from_file(gxf_stream(content))

    def test_optional_keys_default_values_assigned(
        self, gxf_content_minimal: str
    ) -> None:
        """When optional keys are missing, defaults are applied."""

        result = GXFData.from_file(gxf_stream(gxf_content_minimal))

        assert result.points == 3
        assert result.rows == 2
        assert result.ptseparation == pytest.approx(1.0)
        assert result.rwseparation == pytest.approx(1.0)
        assert result.xorigin == pytest.approx(0.0)
        assert result.yorigin == pytest.approx(0.0)
        assert result.rotation == pytest.approx(0.0)

        # All 6 values should be present and valid, no masking due to
        # missing #DUMMY
        assert result.grid.count() == 6

    def test_gtype_positive_raises_for_compressed_grid(
        self, gxf_content_with_gtype_positive: str
    ) -> None:
        """Compressed GXF grids should be rejected."""
        with pytest.raises(ValueError, match="Compressed GXF #GRID values"):
            GXFData.from_file(gxf_stream(gxf_content_with_gtype_positive))

    def test_gtype_zero_is_uncompressed(self, gxf_content_with_gtype_zero: str) -> None:
        """A #GTYPE value of zero should parse as an uncompressed grid."""
        result = GXFData.from_file(gxf_stream(gxf_content_with_gtype_zero))

        assert result.points == 2
        assert result.rows == 2
        np.testing.assert_allclose(result.grid, [[1.0, 2.0], [3.0, 4.0]])

    def test_unknown_single_hash_key_warns_logs_and_skips(
        self, gxf_content_with_unknown_key: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Unknown single-hash keys should warn, log, and be skipped."""
        caplog.set_level("WARNING", logger="xtgeo.io.gxf._gxf_io")

        with pytest.warns(UserWarning, match="UNKNOWN_KEY"):
            result = GXFData.from_file(gxf_stream(gxf_content_with_unknown_key))

        assert result.points == 3
        assert result.rows == 2
        assert "Ignoring unknown GXF key '#UNKNOWN_KEY'" in caplog.text

    def test_extension_keys_are_ignored_without_warning_or_log(
        self,
        valid_gxf_content: str,
        caplog: pytest.LogCaptureFixture,
        recwarn: pytest.WarningsRecorder,
    ) -> None:
        """Double-hash extension keys should be ignored quietly."""
        caplog.set_level("WARNING", logger="xtgeo.io.gxf._gxf_io")

        result = GXFData.from_file(gxf_stream(valid_gxf_content))

        assert result.points == 3
        assert not recwarn
        assert "XMAX" not in caplog.text
        assert "YMAX" not in caplog.text

    @pytest.mark.parametrize(
        ("extension_value", "expected_message"),
        [
            ("1 2", "Expected a single finite decimal number"),
            ('"not-a-number"', "Invalid value"),
            ("1e309", "Expected a single finite decimal number"),
        ],
    )
    def test_extension_key_value_must_be_single_finite_decimal_number(
        self,
        extension_value: str,
        expected_message: str,
        gxf_content_with_extension_value: Callable[[str], str],
    ) -> None:
        """Extension key values should be single finite decimal numbers."""
        content = gxf_content_with_extension_value(extension_value)
        with pytest.raises(ValueError, match=expected_message):
            GXFData.from_file(gxf_stream(content))

    @pytest.mark.parametrize(
        ("ignored_key", "expected_message"),
        [
            ("#UNKNOWN", "Missing value for ignored key '#UNKNOWN'"),
            ("##XMAX", "Missing value for key '##XMAX'"),
        ],
    )
    def test_ignored_key_followed_by_grid_raises(
        self,
        ignored_key: str,
        expected_message: str,
        gxf_content_with_ignored_key_before_grid: Callable[[str], str],
    ) -> None:
        """Ignored keys without values before #GRID should raise."""
        content = gxf_content_with_ignored_key_before_grid(ignored_key)
        with pytest.raises(ValueError, match=expected_message):
            GXFData.from_file(gxf_stream(content))

    @pytest.mark.parametrize(
        ("ignored_key", "expected_message"),
        [
            ("#UNKNOWN", "Missing value for ignored key '#UNKNOWN'"),
            ("##XMAX", "Missing value for key '##XMAX'"),
        ],
    )
    def test_ignored_key_at_eof_raises(
        self,
        ignored_key: str,
        expected_message: str,
        gxf_content_with_ignored_key_at_eof: Callable[[str], str],
    ) -> None:
        """Ignored keys without values at EOF should raise."""
        content = gxf_content_with_ignored_key_at_eof(ignored_key)
        with pytest.raises(ValueError, match=expected_message):
            GXFData.from_file(gxf_stream(content))

    @pytest.mark.parametrize(
        "duplicate_key",
        [
            "POINTS",
            "ROWS",
            "GTYPE",
            "PTSEPARATION",
            "RWSEPARATION",
            "XORIGIN",
            "YORIGIN",
            "ROTATION",
            "DUMMY",
        ],
    )
    def test_duplicate_scalar_key_raises(
        self,
        duplicate_key: str,
        gxf_content_with_duplicate_scalar_key: Callable[[str], str],
    ) -> None:
        """Duplicate scalar keys should raise a parser error."""
        content = gxf_content_with_duplicate_scalar_key(duplicate_key)

        with pytest.raises(ValueError, match=f"Duplicate key '#{duplicate_key}'"):
            GXFData.from_file(gxf_stream(content))

    def test_sense_key_with_zero_value_is_accepted(
        self, gxf_content_with_sense_value: Callable[[str], str]
    ) -> None:
        """A quoted zero #SENSE value should be accepted."""
        result = GXFData.from_file(gxf_stream(gxf_content_with_sense_value('"0"')))
        assert result.points == 2

    @pytest.mark.parametrize(
        "grid_body",
        [
            *(
                f'1 2\n{key}\n"1"\n3 4'
                for key in [
                    "#POINTS",
                    "#ROWS",
                    "#PTSEPARATION",
                    "#RWSEPARATION",
                    "#XORIGIN",
                    "#YORIGIN",
                    "#ROTATION",
                    "#DUMMY",
                    "#GRID",
                ]
            ),
            '1 2\n##XMAX\n"9999"\n3 4',
            '1 2 3 4\n#ROTATION\n"45"',
        ],
    )
    def test_key_inside_grid_raises(
        self, grid_body: str, gxf_content_with_key_inside_grid: Callable[[str], str]
    ) -> None:
        """A key appearing inside the #GRID section must raise."""
        content = gxf_content_with_key_inside_grid(grid_body)
        with pytest.raises(ValueError, match="Unexpected key.*inside #GRID"):
            GXFData.from_file(gxf_stream(content))

    def test_comment_inside_grid_is_allowed(
        self, gxf_content_with_comment_inside_grid: str
    ) -> None:
        """Comments (lines starting with !) inside #GRID should be skipped."""
        result = GXFData.from_file(gxf_stream(gxf_content_with_comment_inside_grid))
        assert result.grid.count() == 4
        expected = np.array([[1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_allclose(result.grid.data, expected)


class TestGXFFileRoundtrip:
    """Tests for write-then-read roundtrips via file streams."""

    def test_roundtrip_stringio(self) -> None:
        """Writing to StringIO and reading back should preserve GXF data."""
        values = np.ma.array(
            [[1.0, 2.0, 3.0], [4.0, 9999.0, 6.0]],
            mask=[[False, False, False], [False, True, False]],
        )
        gxf = GXFData(
            points=3,
            rows=2,
            ptseparation=10.0,
            rwseparation=20.0,
            xorigin=100.0,
            yorigin=200.0,
            rotation=30.0,
            dummy=9999.0,
            grid=values,
        )

        stream = StringIO()
        gxf.to_file(stream)
        stream.seek(0)

        re_read = GXFData.from_file(stream)

        assert re_read.points == gxf.points
        assert re_read.rows == gxf.rows
        assert re_read.ptseparation == pytest.approx(gxf.ptseparation)
        assert re_read.rwseparation == pytest.approx(gxf.rwseparation)
        assert re_read.xorigin == pytest.approx(gxf.xorigin)
        assert re_read.yorigin == pytest.approx(gxf.yorigin)
        assert re_read.rotation == pytest.approx(gxf.rotation)
        assert re_read.dummy == pytest.approx(gxf.dummy)
        np.testing.assert_allclose(re_read.grid.data, gxf.grid.data)
        np.testing.assert_array_equal(re_read.grid.mask, gxf.grid.mask)

    def test_roundtrip_bytesio(self) -> None:
        """Writing to BytesIO and reading back should preserve grid data."""
        values = np.ma.array(
            [[1.0, 2.0], [3.0, 9999.0]],
            mask=[[False, False], [False, True]],
        )
        gxf = make_gxf_data(dummy=9999.0, grid=values)

        stream = BytesIO()
        gxf.to_file(stream)
        stream.seek(0)

        re_read = GXFData.from_file(stream)

        np.testing.assert_allclose(re_read.grid.data, gxf.grid.data)
        np.testing.assert_array_equal(re_read.grid.mask, gxf.grid.mask)

    def test_from_file_latin1_comments_and_free_text(
        self, tmp_path: Path, gxf_content_latin1_comments_and_free_text: str
    ) -> None:
        """Latin-1 GXF files should require the matching encoding."""
        path = tmp_path / "latin1_comments.gxf"
        path.write_bytes(gxf_content_latin1_comments_and_free_text.encode("latin-1"))

        with pytest.raises(UnicodeDecodeError):
            GXFData.from_file(path)

        result = GXFData.from_file(path, encoding="latin-1")

        assert result.points == 2
        assert result.rows == 2
        np.testing.assert_allclose(result.grid.data, [[1.0, 2.0], [3.0, 4.0]])

    def test_to_file_utf16_roundtrip(self, tmp_path: Path) -> None:
        """UTF-16 GXF output should roundtrip with explicit encoding."""
        values = np.ma.array(
            [[1.0, 2.0], [3.0, -999.0]],
            mask=[[False, False], [False, True]],
        )
        gxf = make_gxf_data(grid=values)
        path = tmp_path / "utf16_roundtrip.gxf"

        gxf.to_file(path, encoding="utf-16")

        assert path.read_bytes().startswith((b"\xff\xfe", b"\xfe\xff"))
        re_read = GXFData.from_file(path, encoding="utf-16")

        assert re_read.points == gxf.points
        assert re_read.rows == gxf.rows
        assert re_read.dummy == pytest.approx(gxf.dummy)
        np.testing.assert_array_equal(re_read.grid.mask, gxf.grid.mask)
        np.testing.assert_allclose(re_read.grid.data, gxf.grid.data)


class TestGXFWriter:
    """Tests for GXF output format constraints."""

    def test_scalar_headers_are_ordered_and_unquoted(self) -> None:
        """Scalar headers should be written in order without quotes."""
        values = np.ma.array(
            [[1.0, -9999.0], [3.0, 4.0]],
            mask=[[False, True], [False, False]],
        )
        gxf = make_gxf_data(
            ptseparation=1.25,
            rwseparation=-2.5,
            xorigin=10.0,
            yorigin=20.5,
            rotation=12.0,
            dummy=-9999.0,
            grid=values,
        )

        stream = StringIO()
        gxf.to_file(stream)
        header = stream.getvalue().split("#GRID", maxsplit=1)[0]
        header_lines = [
            line for line in header.splitlines() if line and not line.startswith("!")
        ]

        assert header_lines == [
            "#POINTS",
            "2",
            "#ROWS",
            "2",
            "#PTSEPARATION",
            "1.25",
            "#RWSEPARATION",
            "-2.5",
            "#XORIGIN",
            "10.0",
            "#YORIGIN",
            "20.5",
            "#ROTATION",
            "12.0",
            "#DUMMY",
            "-9999.0",
        ]
        for quoted_value in ('"2"', '"1.25"', '"-2.5"', '"10.0"', '"-9999.0"'):
            assert quoted_value not in header

    def test_line_length_at_most_80_chars(self) -> None:
        """GXF spec requires all lines <= 80 characters."""
        ncol = 20
        nrow = 3
        values = np.ma.array(
            [
                np.arange(1000.0, 1000.0 + ncol),
                np.arange(2000.0, 2000.0 + ncol),
                np.arange(3000.0, 3000.0 + ncol),
            ]
        )
        gxf = make_gxf_data(
            points=ncol,
            rows=nrow,
            grid=values,
        )

        stream = StringIO()
        gxf.to_file(stream)
        exported = stream.getvalue()
        stream.seek(0)
        for line in stream:
            assert len(line.rstrip("\n")) <= 80

        grid_lines = exported.split("#GRID\n", maxsplit=1)[1].splitlines()
        grid_tokens_by_line = [line.split() for line in grid_lines]
        assert len(grid_lines) > nrow

        for row in values:
            row_start = GXFData._format_number(float(row[0]))
            assert any(tokens[0] == row_start for tokens in grid_tokens_by_line)
            assert all(row_start not in tokens[1:] for tokens in grid_tokens_by_line)

        stream.seek(0)
        re_read = GXFData.from_file(stream)
        np.testing.assert_allclose(re_read.grid.data, gxf.grid.data)

    def test_to_file_nonexistent_folder_raises(self, tmp_path: Path) -> None:
        """Writing to a non-existent folder should raise OSError."""
        gxf = make_gxf_data()
        bad_path = tmp_path / "no_such_folder" / "output.gxf"
        with pytest.raises(OSError):
            gxf.to_file(bad_path)


class TestGXFDataclass:
    """Tests for GXFData dataclass properties."""

    def test_frozen(self) -> None:
        """GXFData instances should be immutable."""
        gxf = make_gxf_data(dummy=9999.0)

        with pytest.raises(FrozenInstanceError):
            gxf.points = 3

    def test_grid_data_and_mask_are_read_only(self) -> None:
        """Stored grid data and masks should be read-only."""
        gxf = make_gxf_data(
            dummy=9999.0,
            grid=np.ma.array(
                [[1.0, 9999.0], [3.0, 4.0]],
                mask=[[False, False], [False, True]],
            ),
        )

        assert gxf.grid.flags.writeable is False
        assert gxf.grid.data.flags.writeable is False
        assert gxf.grid.mask.flags.writeable is False

        with pytest.raises(ValueError):
            gxf.grid[0, 0] = 10.0
        with pytest.raises(ValueError):
            gxf.grid[0, 0] = np.ma.masked
        with pytest.raises(ValueError):
            gxf.grid.mask[0, 0] = True
        with pytest.raises(ValueError):
            gxf.grid.mask = [[True, True], [True, True]]

    def test_grid_freezing_does_not_mutate_input_array(self) -> None:
        """Freezing a GXFData grid should not mutate the input array."""
        values = np.ma.array([[1.0, 2.0], [3.0, 4.0]], mask=False)

        gxf = make_gxf_data(dummy=None, grid=values)

        values[0, 0] = 99.0

        assert values[0, 0] == pytest.approx(99.0)
        assert gxf.grid[0, 0] == pytest.approx(1.0)


class TestGXFDataComparison:
    """Tests for GXFData equality and tolerant comparisons."""

    def test_exact_equality_returns_notimplemented_for_other_type(self) -> None:
        """Exact equality should return NotImplemented for unrelated types."""
        gxf = make_gxf_data()

        assert gxf.__eq__(object()) is NotImplemented

    def test_allclose_returns_false_for_other_type(self) -> None:
        """Tolerant comparison should return false for unrelated types."""
        gxf = make_gxf_data()

        assert not gxf.allclose(object())

    def test_exact_equality_ignores_masked_grid_values(self) -> None:
        """Exact equality should ignore grid values behind matching masks."""
        first = make_gxf_data(
            grid=np.ma.array(
                [[1.0, 99.0], [3.0, 4.0]],
                mask=[[False, True], [False, False]],
            )
        )
        second = make_gxf_data(
            grid=np.ma.array(
                [[1.0, -123.0], [3.0, 4.0]],
                mask=[[False, True], [False, False]],
            )
        )

        assert first == second

    def test_exact_equality_rejects_unmasked_grid_value_differences(self) -> None:
        """Exact equality should reject differences in unmasked grid values."""
        first = make_gxf_data()
        second = make_gxf_data(grid=np.ma.array([[1.0, 2.1], [3.0, 4.0]]))

        assert first != second

    @pytest.mark.parametrize(
        "overrides",
        [
            {"points": 3, "grid": np.ma.array([[1.0, 2.0, 5.0], [3.0, 4.0, 6.0]])},
            {"rows": 3, "grid": np.ma.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])},
            {"ptseparation": 2.0},
            {"rwseparation": 2.0},
            {"xorigin": 1.0},
            {"yorigin": 1.0},
            {"rotation": 1.0},
            {"dummy": -998.0},
        ],
    )
    def test_exact_equality_rejects_core_value_differences(
        self, overrides: dict[str, object]
    ) -> None:
        """Exact equality should reject core value differences."""
        first = make_gxf_data()
        second = make_gxf_data(**overrides)

        assert first != second

    def test_exact_equality_rejects_mask_differences(self) -> None:
        """Exact equality should reject grid mask differences."""
        first = make_gxf_data(
            grid=np.ma.array(
                [[1.0, 2.0], [3.0, 4.0]],
                mask=[[False, True], [False, False]],
            )
        )
        second = make_gxf_data(
            grid=np.ma.array(
                [[1.0, 2.0], [3.0, 4.0]],
                mask=[[False, False], [True, False]],
            )
        )

        assert first != second

    def test_allclose_accepts_small_floating_differences(self) -> None:
        """Tolerant comparison should accept small floating-point differences."""
        eps = 1e-06
        first = make_gxf_data(
            ptseparation=1.0,
            rwseparation=2.0,
            xorigin=3.0,
            yorigin=4.0,
            rotation=5.0,
            dummy=-999.0,
            grid=np.ma.array([[1.0, 2.0], [3.0, 4.0]]),
        )
        second = make_gxf_data(
            ptseparation=1.0 + eps,
            rwseparation=2.0 + eps,
            xorigin=3.0 + eps,
            yorigin=4.0 + eps,
            rotation=5.0 + eps,
            dummy=-999.0 + eps,
            grid=np.ma.array([[1.0, 2.0 + eps], [3.0, 4.0]]),
        )

        assert first.allclose(second)

    def test_allclose_rejects_grid_shape_differences(self) -> None:
        """Tolerant comparison should reject grid shape differences."""
        first = make_gxf_data()
        second = make_gxf_data(
            points=3,
            rows=2,
            grid=np.ma.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        )

        assert not first.allclose(second)

    @pytest.mark.parametrize(
        "overrides",
        [
            {"ptseparation": 2.0},
            {"rwseparation": 2.0},
            {"xorigin": 1.0},
            {"yorigin": 1.0},
            {"rotation": 2.0},
            {"dummy": -998.0},
        ],
    )
    def test_allclose_rejects_large_floating_core_value_differences(
        self, overrides: dict[str, object]
    ) -> None:
        """Tolerant comparison should reject large core value differences."""
        first = make_gxf_data()
        second = make_gxf_data(**overrides)

        assert not first.allclose(second)

    def test_allclose_rejects_none_dummy_difference(self) -> None:
        """Tolerant comparison should reject None versus numeric dummy values."""
        first = make_gxf_data(dummy=None)
        second = make_gxf_data(dummy=-999.0)

        assert not first.allclose(second)

    def test_allclose_accepts_matching_none_dummy_values(self) -> None:
        """Tolerant comparison should accept matching None dummy values."""
        first = make_gxf_data(dummy=None)
        second = make_gxf_data(dummy=None)

        assert first.allclose(second)

    def test_allclose_rejects_large_dummy_difference(self) -> None:
        """Tolerant comparison should reject large dummy value differences."""
        first = make_gxf_data(dummy=-999.0)
        second = make_gxf_data(dummy=-998.0)

        assert not first.allclose(second)

    def test_allclose_rejects_mask_differences(self) -> None:
        """Tolerant comparison should reject grid mask differences."""
        first = make_gxf_data(
            grid=np.ma.array(
                [[1.0, 2.0], [3.0, 4.0]],
                mask=[[False, True], [False, False]],
            )
        )
        second = make_gxf_data(
            grid=np.ma.array(
                [[1.0, 2.0], [3.0, 4.0]],
                mask=[[False, False], [False, True]],
            )
        )

        assert not first.allclose(second)


class TestFileFormatVerification:
    """Tests for file format detection and validation."""

    def test_surface_from_file_gxf_content_missing_rows_raises(
        self, tmp_path: Path, gxf_content_missing_rows: str
    ) -> None:
        """
        surface_from_file should report parser validation errors for GXF input.
        """
        path = tmp_path / "test.gxf"
        path.write_text(gxf_content_missing_rows)

        with pytest.raises(
            ValueError,
            match=r"Missing mandatory keys: \['ROWS'\]\.",
        ):
            xtgeo.surface_from_file(path)

    def test_non_gxf_extension_and_content_raises(
        self, tmp_path: Path, non_gxf_content: str
    ) -> None:
        """Non-.gxf file path with non-GXF content should fail."""
        path = tmp_path / "test.txt"
        path.write_text(non_gxf_content)

        with pytest.raises(
            ValueError,
            match=r"Missing mandatory keys: \['POINTS', 'ROWS'\]",
        ):
            GXFData.from_file(path)

        with pytest.raises(
            ValueError,
            match=r"Missing mandatory keys: \['POINTS', 'ROWS'\]",
        ):
            xtgeo.surface_from_file(path, fformat="gxf")

    def test_surface_from_file_no_format_hints(
        self, gxf_path: Path, tmp_path: Path
    ) -> None:
        """
        Test that a GXF file with a large number of comments or free text
        at the beginning, and without a '.gxf' file extension, fails in
        the parsing stage due to missing mandatory keys.
        """

        outpath = tmp_path / "fdata_test.not_gxf"
        shutil.copy(gxf_path, outpath)

        # File suffix is not .gxf, no other format hints exist. So this should fail
        with pytest.raises(
            ValueError, match="File format None is unknown or unsupported"
        ):
            xtgeo.surface_from_file(outpath)

        # The same test, but with a format hint, should succeed
        surf = xtgeo.surface_from_file(outpath, fformat="gxf")
        assert surf.ncol == 330
        assert surf.nrow == 208


class TestDummyTypePreservation:
    """#DUMMY value should retain the type (int or float) from the input."""

    @pytest.mark.parametrize(
        "dummy, expected_value, expected_type",
        [
            (-9999, -9999, int),
            (-9999.0, -9999.0, float),
        ],
    )
    def test_dummy_type_roundtrip_file(
        self, dummy: int | float, expected_value: int | float, expected_type: type
    ) -> None:
        """Dummy type should survive a write-then-read roundtrip."""
        values = np.ma.array(
            [[1.0, 3.0], [2.0, -9999.0]],
            mask=[[False, False], [False, True]],
        )
        gxf = make_gxf_data(dummy=dummy, grid=values)

        stream = StringIO()
        gxf.to_file(stream)
        stream.seek(0)

        re_read = GXFData.from_file(stream)
        assert re_read.dummy == pytest.approx(expected_value)
        assert isinstance(re_read.dummy, expected_type)
        np.testing.assert_array_equal(re_read.grid.mask, gxf.grid.mask)


class TestRegularSurfaceIntegration:
    """Tests for xtgeo RegularSurface GXF integration."""

    def test_from_file(self, valid_gxf_content: str) -> None:
        """surface_from_file should import GXF content as a RegularSurface."""

        surf = xtgeo.surface_from_file(gxf_stream(valid_gxf_content), fformat="gxf")

        assert surf.ncol == 3
        assert surf.nrow == 2
        assert surf.xinc == pytest.approx(30.0)
        assert surf.yinc == pytest.approx(40.0)
        assert surf.xori == pytest.approx(1000.0)
        assert surf.yori == pytest.approx(2000.0)
        assert surf.rotation == pytest.approx(12.5)

        expected_data = np.array([[1.0, 4.0], [2.0, np.nan], [3.0, 6.0]])
        np.testing.assert_allclose(
            surf.values.filled(np.nan), expected_data, equal_nan=True
        )

    def test_import_gxf_always_returns_values(self, valid_gxf_content: str) -> None:
        """
        #GRID is a mandatory section in GXF, so
        GXF import should ignore values=False and include grid values.
        """
        args = import_gxf(FileWrapper(gxf_stream(valid_gxf_content)), values=False)

        assert args["ncol"] == 3
        assert args["nrow"] == 2
        assert args["xinc"] == pytest.approx(30.0)
        assert args["yinc"] == pytest.approx(40.0)
        assert args["xori"] == pytest.approx(1000.0)
        assert args["yori"] == pytest.approx(2000.0)
        assert args["rotation"] == pytest.approx(12.5)
        assert "undef" not in args
        expected_values = np.array([[1.0, 4.0], [2.0, np.nan], [3.0, 6.0]])
        np.testing.assert_allclose(
            args["values"].filled(np.nan), expected_values, equal_nan=True
        )
        assert args["values"].data[1, 1] == pytest.approx(xtgeo.UNDEF)

    def test_surface_from_file_gxf_values_false_still_loads_values(
        self, valid_gxf_content: str
    ) -> None:
        """
        #GRID is a mandatory section in GXF, so
        surface_from_file should load GXF values even with values=False.
        """
        surf = xtgeo.surface_from_file(
            gxf_stream(valid_gxf_content), fformat="gxf", values=False
        )

        assert surf._isloaded is True
        assert surf.values.shape == (3, 2)
        assert surf.values[1, 1] is np.ma.masked

    def test_to_file_constructs_gxfdata_from_regular_surface_parameters(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """RegularSurface export should construct GXFData from surface parameters."""
        values = np.ma.array(
            [[11.0, 44.0], [22.0, 55.0], [33.0, 66.0]],
            mask=[[False, False], [False, False], [False, True]],
        )
        surf = make_regular_surface(undef=-999.0, yflip=-1, values=values)
        captured: dict[str, object] = {}

        def capture_gxf_data(**kwargs: object) -> GXFData:
            captured.update(kwargs)
            return GXFData(**kwargs)

        monkeypatch.setattr(_regsurf_export, "GXFData", capture_gxf_data)

        surf.to_file(BytesIO(), fformat="gxf")

        assert captured.keys() == {
            "points",
            "rows",
            "xorigin",
            "yorigin",
            "ptseparation",
            "rwseparation",
            "rotation",
            "dummy",
            "grid",
        }
        assert captured["points"] == surf.ncol
        assert captured["rows"] == surf.nrow
        assert captured["xorigin"] == pytest.approx(surf.xori)
        assert captured["yorigin"] == pytest.approx(surf.yori)
        assert captured["ptseparation"] == pytest.approx(surf.xinc)
        assert captured["rwseparation"] == pytest.approx(-surf.yinc)
        assert captured["rotation"] == pytest.approx(surf.rotation)
        assert captured["dummy"] == pytest.approx(surf.undef)
        np.testing.assert_array_equal(captured["grid"], surf.values.T)

    def test_to_file_roundtrip(self) -> None:
        """RegularSurface GXF export and import should preserve values."""

        values = np.ma.array(
            [[11.0, 44.0], [22.0, 55.0], [33.0, np.nan]],
            mask=[[False, False], [False, False], [False, True]],
        )
        surf = make_regular_surface(values=values)

        stream = BytesIO()
        surf.to_file(stream, fformat="gxf")
        stream.seek(0)

        re_read = xtgeo.surface_from_file(stream, fformat="gxf")

        assert re_read.ncol == surf.ncol
        assert re_read.nrow == surf.nrow
        assert re_read.xinc == pytest.approx(surf.xinc)
        assert re_read.yinc == pytest.approx(surf.yinc)
        assert re_read.xori == pytest.approx(surf.xori)
        assert re_read.yori == pytest.approx(surf.yori)
        assert re_read.rotation == pytest.approx(surf.rotation)

        np.testing.assert_allclose(
            re_read.values.filled(np.nan),
            surf.values.filled(np.nan),
            equal_nan=True,
        )

    def test_to_file_path_roundtrip(self, tmp_path: Path) -> None:
        """RegularSurface GXF path export should return the path and roundtrip."""
        values = np.ma.array(
            [[11.0, 44.0], [22.0, np.nan], [33.0, 66.0]],
            mask=[[False, False], [False, True], [False, False]],
        )
        surf = make_regular_surface(undef=-999.0, values=values)
        path = tmp_path / "surface.gxf"

        result = surf.to_file(path, fformat="gxf")

        assert result == path
        re_read = xtgeo.surface_from_file(path, fformat="gxf")
        assert re_read.ncol == surf.ncol
        assert re_read.nrow == surf.nrow
        assert re_read.xinc == pytest.approx(surf.xinc)
        assert re_read.yinc == pytest.approx(surf.yinc)
        assert re_read.xori == pytest.approx(surf.xori)
        assert re_read.yori == pytest.approx(surf.yori)
        assert re_read.rotation == pytest.approx(surf.rotation)
        np.testing.assert_allclose(
            re_read.values.filled(np.nan),
            surf.values.filled(np.nan),
            equal_nan=True,
        )
        np.testing.assert_array_equal(re_read.values.mask, surf.values.mask)

    def test_to_file_omits_dummy_when_no_values_are_masked(self) -> None:
        """GXF export should omit #DUMMY when no grid values are masked."""
        values = np.ma.array(
            [[11.0, 44.0], [123.0, 55.0], [33.0, 66.0]],
            mask=False,
        )
        surf = make_regular_surface(undef=123.0, values=values)

        stream = BytesIO()
        surf.to_file(stream, fformat="gxf")

        exported = stream.getvalue().decode()
        assert "#DUMMY" not in exported

        stream.seek(0)
        re_read = xtgeo.surface_from_file(stream, fformat="gxf")

        assert re_read.nactive == surf.ncol * surf.nrow
        assert re_read.values[1, 0] == pytest.approx(123.0)

    def test_to_file_writes_dummy_when_values_are_masked(self) -> None:
        """GXF export should write #DUMMY when values are masked."""
        values = np.ma.array(
            [[11.0, 44.0], [22.0, 55.0], [33.0, 66.0]],
            mask=[[False, False], [False, False], [False, True]],
        )
        surf = make_regular_surface(undef=-999.0, values=values)

        stream = BytesIO()
        surf.to_file(stream, fformat="gxf")

        exported = stream.getvalue().decode()
        lines = exported.splitlines()
        dummy_index = lines.index("#DUMMY")
        assert lines[dummy_index + 1] == "-999.0"

        stream.seek(0)
        re_read = xtgeo.surface_from_file(stream, fformat="gxf")

        np.testing.assert_array_equal(re_read.values.mask, surf.values.mask)

    def test_to_file_preserves_unmasked_values_equal_to_undef(self) -> None:
        """GXF export should preserve unmasked values equal to surface undef."""
        values = np.ma.array(
            [[11.0, 44.0], [-999.0, 55.0], [33.0, 66.0]],
            mask=[[False, False], [False, False], [False, True]],
        )
        surf = make_regular_surface(undef=-999.0, values=values)

        stream = BytesIO()
        surf.to_file(stream, fformat="gxf")

        exported = stream.getvalue().decode()
        lines = exported.splitlines()
        dummy_index = lines.index("#DUMMY")
        assert float(lines[dummy_index + 1]) != pytest.approx(surf.undef)

        stream.seek(0)

        re_read = xtgeo.surface_from_file(stream, fformat="gxf")

        np.testing.assert_array_equal(re_read.values.mask, surf.values.mask)
        assert re_read.values[1, 0] == pytest.approx(-999.0)

    @pytest.mark.parametrize("surface_state", ["not_loaded", "empty_values"])
    def test_to_file_raises_without_grid_values(self, surface_state: str) -> None:
        """RegularSurface GXF export should require grid values."""
        if surface_state == "not_loaded":
            surf = xtgeo.RegularSurface(
                ncol=3,
                nrow=2,
                xinc=10.0,
                yinc=20.0,
                xori=100.0,
                yori=200.0,
                rotation=15.0,
            )
        else:
            surf = make_regular_surface(values=np.ones((3, 2)))
            surf._values = np.ma.array([])

        with (
            pytest.warns(
                UserWarning, match="fewer than 4 nodes"
            ),  # from RegularSurface
            pytest.raises(ValueError, match="Cannot export GXF without grid values"),
        ):
            surf.to_file(BytesIO(), fformat="gxf")

    def test_guess_format_by_extension(
        self, tmp_path: Path, gxf_content_small_grid: str
    ) -> None:
        """A .gxf suffix should allow surface_from_file to guess GXF format."""
        path = tmp_path / "small.gxf"
        path.write_text(gxf_content_small_grid)

        surf = xtgeo.surface_from_file(path)
        assert surf.ncol == 2
        assert surf.nrow == 2

    @pytest.mark.parametrize(
        "file_rotation, expected",
        [
            (0.0, 0.0),
            (90.0, 90.0),
            (359.9, 359.9),
            (360.0, 0.0),
            (450.0, 90.0),
            (-90.0, 270.0),
            (-360.0, 0.0),
            (720.0, 0.0),
        ],
    )
    def test_rotation_normalized_to_0_360(
        self,
        file_rotation: float,
        expected: float,
        gxf_content_with_rotation: Callable[[float], str],
    ) -> None:
        """Imported GXF rotations should be normalized to [0, 360)."""
        content = gxf_content_with_rotation(file_rotation)
        surf = xtgeo.surface_from_file(gxf_stream(content), fformat="gxf")
        assert surf.rotation == pytest.approx(expected)

    @pytest.mark.parametrize(
        "rwseparation, expected_yinc, expected_yflip",
        [
            (1.0, 1.0, 1),
            (20.5, 20.5, 1),
            (-1.0, 1.0, -1),
            (-20.5, 20.5, -1),
        ],
    )
    def test_negative_yinc_converted_to_yflip(
        self,
        rwseparation: float,
        expected_yinc: float,
        expected_yflip: int,
        gxf_content_with_rwseparation: Callable[[float], str],
    ) -> None:
        """Negative GXF row separation should map to positive yinc and yflip."""
        content = gxf_content_with_rwseparation(rwseparation)
        surf = xtgeo.surface_from_file(gxf_stream(content), fformat="gxf")
        assert surf.yinc == pytest.approx(expected_yinc)
        assert surf.yflip == expected_yflip


class TestDummyValue:
    """Tests for #DUMMY value handling: masking, edge cases, and propagation."""

    @pytest.mark.parametrize(
        (
            "dummy_literal, grid_values, expected_dummy, expected_type, "
            "expected_count, masked_positions"
        ),
        [
            ("-9999", "1 2 3 -9999", -9999, int, 3, [(1, 1)]),
            ("-9999.0", "1 2 3 -9999.0", -9999.0, float, 3, [(1, 1)]),
            ("0", "1 0 3 4", 0, int, 3, [(0, 1)]),
            ("-999", "10 -999 30 40", -999, int, 3, [(0, 1)]),
            ("1e30", "1.0 1e30 3.0 4.0", 1e30, float, 3, [(0, 1)]),
            (
                "-999",
                "-999 -999 -999 -999",
                -999,
                int,
                0,
                [(0, 0), (0, 1), (1, 0), (1, 1)],
            ),
            ("9999", "1 2 3 4", 9999, int, 4, []),
            ("-999", "-999 2 -999 4", -999, int, 2, [(0, 0), (1, 0)]),
        ],
    )
    def test_dummy_masking_from_file(
        self,
        dummy_literal: str,
        grid_values: str,
        expected_dummy: float,
        expected_type: type,
        expected_count: int,
        masked_positions: list[tuple[int, int]],
        gxf_content_with_dummy_value: Callable[[str, str], str],
    ) -> None:
        """#DUMMY values should parse with type preservation and mask matches."""
        content = gxf_content_with_dummy_value(dummy_literal, grid_values)
        result = GXFData.from_file(gxf_stream(content))

        assert result.dummy == pytest.approx(expected_dummy)
        assert isinstance(result.dummy, expected_type)
        assert result.grid.count() == expected_count
        for row, column in masked_positions:
            assert result.grid[row, column] is np.ma.masked
        if not masked_positions:
            assert not np.any(result.grid.mask)

    def test_no_dummy_key_means_default_value_is_valid(
        self, gxf_content_no_dummy_with_default_value: str
    ) -> None:
        """When #DUMMY is missing, there is no dummy value per the GXF spec."""

        result = GXFData.from_file(gxf_stream(gxf_content_no_dummy_with_default_value))

        assert result.dummy is None
        assert result.grid.count() == 4
        assert not np.any(result.grid.mask)
        assert result.grid[1, 0] == pytest.approx(9999999.0)

    def test_surface_from_file_missing_dummy_keeps_default_value_valid(
        self, gxf_content_no_dummy_with_default_value: str
    ) -> None:
        """
        RegularSurface import should keep default-like values valid without #DUMMY.
        """
        surf = xtgeo.surface_from_file(
            gxf_stream(gxf_content_no_dummy_with_default_value), fformat="gxf"
        )

        assert surf.nactive == 4
        assert surf.undef != pytest.approx(9999999.0)
        assert surf.values[0, 1] == pytest.approx(9999999.0)

    @pytest.mark.parametrize(
        "grid_values, input_mask, expected_masked_positions, expected_count",
        [
            pytest.param(
                [[1.0, 3.0], [2.0, -999.0]],
                False,
                [(1, 1)],
                3,
                id="mask_dummy",
            ),
            pytest.param(
                [[1.0, 3.0], [2.0, 5.0]],
                [[True, False], [False, False]],
                [(0, 0)],
                3,
                id="preserve_existing_mask",
            ),
            pytest.param(
                [[1.0, -999.0], [2.0, 5.0]],
                [[True, False], [False, False]],
                [(0, 0), (0, 1)],
                2,
                id="combine_dummy_and_existing_mask",
            ),
        ],
    )
    def test_direct_construction_masks_dummy_values(
        self,
        grid_values: list[list[float]],
        input_mask: bool | list[list[bool]],
        expected_masked_positions: list[tuple[int, int]],
        expected_count: int,
    ) -> None:
        """Direct construction should mask dummy values and preserve existing masks."""
        values: np.ma.MaskedArray = np.ma.array(grid_values, mask=input_mask)
        gxf = make_gxf_data(dummy=-999.0, grid=values)

        for row, column in expected_masked_positions:
            assert gxf.grid[row, column] is np.ma.masked
        assert gxf.grid.count() == expected_count

    def test_surface_from_file_masks_dummy_values(
        self, tmp_path: Path, gxf_content_with_dummy_value: Callable[[str, str], str]
    ) -> None:
        """GXF dummy values should be masked in the imported RegularSurface."""
        content = gxf_content_with_dummy_value("-9999.0", "1 2 3 -9999.0")
        path = tmp_path / "test.gxf"
        path.write_text(content)

        surf = xtgeo.surface_from_file(path, fformat="gxf")

        # The masked value should propagate as NaN in the RegularSurface
        filled = surf.values.filled(np.nan)
        assert np.isnan(filled).sum() == 1
        assert surf.nactive == 3

    def test_surface_from_file_stores_xtgeo_undef_under_masked_dummy_values(
        self, gxf_content_with_dummy_value: Callable[[str, str], str]
    ) -> None:
        """RegularSurface import should store xtgeo undef under masked dummies."""
        content = gxf_content_with_dummy_value("-9999.0", "1 2 3 -9999.0")
        surf = xtgeo.surface_from_file(gxf_stream(content), fformat="gxf")

        assert surf.undef == pytest.approx(xtgeo.UNDEF)
        assert surf.values[1, 1] is np.ma.masked
        assert surf.values.data[1, 1] == pytest.approx(xtgeo.UNDEF)

    def test_dummy_roundtrip_masking_preserved(self) -> None:
        """Masked values should survive a GXF write-read roundtrip."""
        # Use symmetric off-diagonal values so that the internal
        # transpose in to_file/from_file does not alter data.
        values = np.ma.array(
            [[10.0, 20.0], [20.0, -999.0]],
            mask=[[False, False], [False, True]],
        )
        gxf = make_gxf_data(
            ptseparation=5.0,
            rwseparation=5.0,
            xorigin=100.0,
            yorigin=200.0,
            grid=values,
        )

        stream = StringIO()
        gxf.to_file(stream)
        stream.seek(0)

        re_read = GXFData.from_file(stream)
        assert re_read.grid.count() == 3
        np.testing.assert_array_equal(re_read.grid.mask, gxf.grid.mask)
        np.testing.assert_allclose(
            re_read.grid.filled(np.nan),
            gxf.grid.filled(np.nan),
            equal_nan=True,
        )


class TestInRepoGXFVariantFixtures:
    """Tests using small inline GXF samples with producer-style variants."""

    def test_long_preamble_fixture_with_format_hint_without_gxf_suffix(
        self, tmp_path: Path, gxf_producer_long_preamble: str
    ) -> None:
        """Producer GXF with long preamble should import with explicit format hint."""
        path = tmp_path / "producer_long_preamble.dat"
        path.write_text(gxf_producer_long_preamble)

        surf = xtgeo.surface_from_file(path, fformat="gxf")

        assert surf.ncol == 2
        assert surf.nrow == 2
        assert surf.xinc == pytest.approx(25.0)
        assert surf.yinc == pytest.approx(50.0)
        assert surf.xori == pytest.approx(123.0)
        assert surf.yori == pytest.approx(456.0)
        assert surf.rotation == pytest.approx(10.0)
        assert surf.yflip == -1
        np.testing.assert_allclose(surf.values, [[1.0, 3.0], [2.0, 4.0]])

    def test_double_hash_keys(self, tmp_path: Path, gxf_double_hash_keys: str) -> None:
        """Double-hash keys should be ignored."""
        path = tmp_path / "double_hash_keys.gxf"
        path.write_text(gxf_double_hash_keys)

        gxf = GXFData.from_file(path)
        assert gxf.points == 2
        assert gxf.rows == 2
        assert gxf.dummy == pytest.approx(-999.0)
        assert gxf.grid[0, 1] is np.ma.masked

        surf = xtgeo.surface_from_file(path)
        assert surf.ncol == 2
        assert surf.nrow == 2
        assert surf.rotation == pytest.approx(12.5)
        expected = np.array([[10.0, 30.0], [np.nan, 40.0]])
        np.testing.assert_allclose(surf.values.filled(np.nan), expected, equal_nan=True)

    def test_minimal_defaults(self, tmp_path: Path, gxf_content_minimal: str) -> None:
        """Minimal GXF content should apply defaults and import as a surface."""
        path = tmp_path / "minimal_defaults.gxf"
        path.write_text(gxf_content_minimal)

        gxf = GXFData.from_file(path)
        assert gxf.ptseparation == pytest.approx(1.0)
        assert gxf.rwseparation == pytest.approx(1.0)
        assert gxf.xorigin == pytest.approx(0.0)
        assert gxf.yorigin == pytest.approx(0.0)
        assert gxf.rotation == pytest.approx(0.0)
        assert gxf.dummy is None

        surf = xtgeo.surface_from_file(path)
        assert surf.ncol == 3
        assert surf.nrow == 2
        assert surf.xinc == pytest.approx(1.0)
        assert surf.yinc == pytest.approx(1.0)
        np.testing.assert_allclose(
            surf.values.filled(np.nan), [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]
        )


class TestRealGXFData:
    """Tests using a real data file."""

    def test_parse_header_values(self, gxf_data_from_file: GXFData) -> None:
        """The real GXF fixture should expose expected header values."""
        assert gxf_data_from_file.points == 330
        assert gxf_data_from_file.rows == 208
        assert gxf_data_from_file.ptseparation == pytest.approx(30.4800600)
        assert gxf_data_from_file.rwseparation == pytest.approx(30.4800600)
        assert gxf_data_from_file.xorigin == pytest.approx(427391.726575)
        assert gxf_data_from_file.yorigin == pytest.approx(7250373.922731)
        assert gxf_data_from_file.rotation == pytest.approx(66.57993141719481)
        assert gxf_data_from_file.dummy == pytest.approx(9999999.0)

    def test_grid_shape(self, gxf_data_from_file: GXFData) -> None:
        """The real GXF fixture should have the expected grid shape."""
        assert gxf_data_from_file.grid.shape == (208, 330)
        assert gxf_data_from_file.grid.size == 208 * 330

    def test_real_fixture_masks_dummy_values(
        self,
        gxf_data_from_file: GXFData,
        regular_surface_from_gxf_file: xtgeo.RegularSurface,
    ) -> None:
        """The real GXF fixture should mask dummy values in both representations."""
        assert gxf_data_from_file.grid.count() < gxf_data_from_file.grid.size
        assert np.ma.count_masked(gxf_data_from_file.grid) > 0

        gxf_valid = gxf_data_from_file.grid.compressed()
        assert np.all(np.isfinite(gxf_valid))
        assert np.all(gxf_valid != gxf_data_from_file.dummy)

        surface_valid = regular_surface_from_gxf_file.values.compressed()
        assert np.all(np.isfinite(surface_valid))
        assert np.all(surface_valid != gxf_data_from_file.dummy)

    def test_roundtrip_preserves_data(
        self, gxf_data_from_file: GXFData, tmp_path: Path
    ) -> None:
        """Real GXF data should survive a GXFData file roundtrip."""
        original = gxf_data_from_file

        outpath = tmp_path / "roundtrip.gxf"
        original.to_file(outpath)
        reloaded = GXFData.from_file(outpath)

        assert reloaded.points == original.points
        assert reloaded.rows == original.rows
        assert reloaded.ptseparation == pytest.approx(original.ptseparation)
        assert reloaded.rwseparation == pytest.approx(original.rwseparation)
        assert reloaded.xorigin == pytest.approx(original.xorigin)
        assert reloaded.yorigin == pytest.approx(original.yorigin)
        assert reloaded.rotation == pytest.approx(original.rotation)
        assert reloaded.dummy == pytest.approx(original.dummy)
        np.testing.assert_array_equal(reloaded.grid.mask, original.grid.mask)
        np.testing.assert_allclose(
            reloaded.grid.compressed(), original.grid.compressed()
        )

    def test_surface_from_file_metadata(
        self, regular_surface_from_gxf_file: xtgeo.RegularSurface
    ) -> None:
        """The real GXF fixture should import expected RegularSurface values."""
        assert regular_surface_from_gxf_file.ncol == 330
        assert regular_surface_from_gxf_file.nrow == 208
        assert regular_surface_from_gxf_file.xinc == pytest.approx(30.4800600)
        assert regular_surface_from_gxf_file.yinc == pytest.approx(30.4800600)
        assert regular_surface_from_gxf_file.xori == pytest.approx(427391.726575)
        assert regular_surface_from_gxf_file.yori == pytest.approx(7250373.922731)
        assert regular_surface_from_gxf_file.rotation == pytest.approx(
            66.57993141719481
        )

    def test_surface_from_file_values_shape(
        self, regular_surface_from_gxf_file: xtgeo.RegularSurface
    ) -> None:
        """The real GXF fixture should import with expected surface value shape."""
        assert regular_surface_from_gxf_file.values.shape == (330, 208)
        assert (
            regular_surface_from_gxf_file.nactive
            < regular_surface_from_gxf_file.ncol * regular_surface_from_gxf_file.nrow
        )


class TestGXFDataValidation:
    """Tests for GXFData __post_init__ validation logic."""

    @pytest.mark.parametrize(
        "overrides, message",
        [
            ({"points": 0}, "positive integers"),
            ({"points": -1}, "positive integers"),
            ({"rows": 0}, "positive integers"),
            ({"ptseparation": 0.0}, "ptseparation.*positive"),
            ({"ptseparation": -1.0}, "ptseparation.*positive"),
            ({"rwseparation": 0.0}, "rwseparation.*non-zero"),
        ],
    )
    def test_invalid_constructor_arguments_raise(
        self, overrides: dict[str, object], message: str
    ) -> None:
        """Invalid GXFData constructor arguments should raise ValueError."""
        with pytest.raises(ValueError, match=message):
            GXFData(**valid_gxfdata_args(**overrides))

    def test_wrong_grid_shape_raises(self) -> None:
        """Grid shape not matching (rows, points) must raise ValueError."""
        with pytest.raises(ValueError, match="Invalid shape"):
            GXFData(**valid_gxfdata_args(points=3))

    def test_negative_rwseparation_is_valid(self) -> None:
        """
        Negative rwseparation is allowed
        (maps to yflip=-1 in RegularSurface).
        """
        gxf = GXFData(**valid_gxfdata_args(rwseparation=-1.0))
        assert gxf.rwseparation == -1.0

    @pytest.mark.parametrize(
        "kwargs, message",
        [
            ({"ptseparation": np.nan}, "ptseparation must be a finite number"),
            ({"rwseparation": np.inf}, "rwseparation must be a finite number"),
            ({"xorigin": -np.inf}, "xorigin must be a finite number"),
            ({"dummy": np.nan}, "dummy value must be a finite number"),
        ],
    )
    def test_non_finite_coredata_raises(
        self, kwargs: dict[str, float], message: str
    ) -> None:
        """Non-finite scalar constructor values should raise ValueError."""
        with pytest.raises(ValueError, match=message):
            GXFData(**valid_gxfdata_args(**kwargs))

    def test_masked_grid_without_dummy_raises(self) -> None:
        """Masked direct-construction grids should require a dummy value."""
        values = np.ma.array(
            [[1.0, 2.0], [3.0, 4.0]],
            mask=[[False, True], [False, False]],
        )

        with pytest.raises(ValueError, match="require a dummy value"):
            GXFData(**valid_gxfdata_args(dummy=None, grid=values))

    def test_direct_construction_non_finite_grid_values_without_dummy_raises(
        self,
    ) -> None:
        """Non-finite direct-construction grid values should require dummy."""
        with pytest.raises(ValueError, match="require a dummy value"):
            GXFData(
                **valid_gxfdata_args(
                    points=4,
                    rows=1,
                    dummy=None,
                    grid=np.ma.array([[1.0, np.nan, np.inf, -np.inf]]),
                )
            )

    def test_direct_construction_non_finite_grid_values_are_masked_with_dummy(
        self,
    ) -> None:
        """Non-finite direct-construction grid values should mask with dummy."""
        gxf = GXFData(
            **valid_gxfdata_args(
                points=4,
                rows=1,
                dummy=-999.0,
                grid=np.ma.array([[1.0, np.nan, np.inf, -np.inf]]),
            )
        )

        np.testing.assert_array_equal(gxf.grid.mask, [[False, True, True, True]])
        np.testing.assert_allclose(gxf.grid.compressed(), [1.0])

    @pytest.mark.parametrize("grid_token", ["nan", "inf", "-inf", 1e999, -1e999])
    def test_uncompressed_grid_values_must_be_finite_numbers(
        self, grid_token: str, gxf_content_with_grid_token: Callable[[object], str]
    ) -> None:
        """Uncompressed #GRID tokens should be finite numbers."""
        content = gxf_content_with_grid_token(grid_token)
        with pytest.raises(ValueError, match="Invalid value.*#GRID section"):
            GXFData.from_file(gxf_stream(content))


class TestGXFParserErrors:
    """Tests for parser error handling on malformed input."""

    @pytest.mark.parametrize(
        "file_factory, expected_reference",
        [
            (StringIO, "<_io.StringIO object at "),
            (lambda content: BytesIO(content.encode()), "<_io.BytesIO object at "),
        ],
    )
    def test_error_message_identifies_stream_input(
        self,
        file_factory: Callable[[str], IO],
        expected_reference: str,
        gxf_content_bad_grid_value: str,
    ) -> None:
        """Parser errors for stream inputs should identify the stream object."""
        with pytest.raises(ValueError) as exc_info:
            GXFData.from_file(file_factory(gxf_content_bad_grid_value))

        message = str(exc_info.value)
        assert f"In file {expected_reference}" in message
        assert "Invalid value 'abc' inside #GRID section." in message

    def test_error_message_identifies_path_input(
        self, tmp_path: Path, gxf_content_bad_grid_value: str
    ) -> None:
        """Parser errors for path inputs should identify the path."""
        path = tmp_path / "bad_grid.gxf"
        path.write_text(gxf_content_bad_grid_value)

        with pytest.raises(ValueError) as exc_info:
            GXFData.from_file(path)

        message = str(exc_info.value)
        assert f"In file {path}" in message
        assert "Invalid value 'abc' inside #GRID section." in message

    def test_invalid_value_for_scalar_key_raises(
        self, gxf_content_invalid_points_value: str
    ) -> None:
        """Non-numeric value for #POINTS must raise ValueError."""
        with pytest.raises(ValueError, match="Invalid value.*POINTS"):
            GXFData.from_file(gxf_stream(gxf_content_invalid_points_value))

    @pytest.mark.parametrize(
        "key, value",
        [
            ("PTSEPARATION", "nan"),
            ("XORIGIN", "inf"),
            ("ROTATION", "-inf"),
            ("DUMMY", "nan"),
        ],
    )
    def test_non_base10_scalar_key_raises(
        self,
        key: str,
        value: str,
        gxf_content_with_non_base10_scalar: Callable[[str, str], str],
    ) -> None:
        """Non-base-10 scalar values should raise parser errors."""
        content = gxf_content_with_non_base10_scalar(key, value)
        with pytest.raises(ValueError, match=f"Invalid value.*#{key}"):
            GXFData.from_file(gxf_stream(content))

    def test_overflowing_scalar_key_raises_non_finite(
        self, gxf_content_with_overflowing_rotation: str
    ) -> None:
        """Overflowing scalar values should be rejected as non-finite."""
        with pytest.raises(ValueError, match="Invalid value.*#ROTATION"):
            GXFData.from_file(gxf_stream(gxf_content_with_overflowing_rotation))

    def test_missing_value_line_after_key_eof(
        self, gxf_content_with_key_at_eof: str
    ) -> None:
        """Key at end of file with no value line must raise."""
        with pytest.raises(ValueError, match="Missing value for key '#ROWS'"):
            GXFData.from_file(gxf_stream(gxf_content_with_key_at_eof))

    def test_missing_value_line_key_follows_key(
        self, gxf_content_with_adjacent_keys: str
    ) -> None:
        """Key followed immediately by another key (no value line) must raise."""
        with pytest.raises(ValueError, match="Missing value for key '#POINTS'"):
            GXFData.from_file(gxf_stream(gxf_content_with_adjacent_keys))

    def test_empty_stream_raises(self, gxf_content_empty: str) -> None:
        """An empty file/stream must raise due to missing mandatory keys."""
        with pytest.raises(ValueError, match="Missing mandatory keys"):
            GXFData.from_file(gxf_stream(gxf_content_empty))

    def test_only_comments_raises(self, gxf_content_only_comments: str) -> None:
        """A file with only comments should raise due to missing keys."""
        with pytest.raises(ValueError, match="Missing mandatory"):
            GXFData.from_file(gxf_stream(gxf_content_only_comments))

    def test_float_value_for_int_key_raises(
        self, gxf_content_with_float_points: str
    ) -> None:
        """A float value like '3.5' for #POINTS (int key) must raise."""
        with pytest.raises(ValueError, match="Invalid value.*POINTS"):
            GXFData.from_file(gxf_stream(gxf_content_with_float_points))

    def test_scalar_values_accept_scientific_notation(
        self, gxf_content_with_scientific_notation: str
    ) -> None:
        """Scalar values should accept scientific notation."""
        result = GXFData.from_file(gxf_stream(gxf_content_with_scientific_notation))

        assert result.points == 2
        assert result.rows == 2
        assert result.ptseparation == pytest.approx(12.5)
        assert result.rwseparation == pytest.approx(25.0)
        assert result.xorigin == pytest.approx(-427391.726575)
        assert result.yorigin == pytest.approx(7250373.922731)
        assert result.rotation == pytest.approx(12.3456789)
        assert result.dummy == pytest.approx(-9999.0)
        assert result.grid[0, 1] is np.ma.masked

    @pytest.mark.parametrize("key", ["GTYPE", "RWSEPARATION", "DUMMY"])
    def test_scalar_value_must_be_single_token(
        self, key: str, gxf_content_with_scalar_trailing_token: Callable[[str], str]
    ) -> None:
        """Scalar values should not accept trailing tokens."""
        content = gxf_content_with_scalar_trailing_token(key)
        with pytest.raises(ValueError, match="Expected a single finite decimal number"):
            GXFData.from_file(gxf_stream(content))

    @pytest.mark.parametrize("value", ["1", "0.0", "left-handed", "0 0"])
    def test_sense_value_must_be_single_zero(
        self, value: str, gxf_content_with_sense_value: Callable[[str], str]
    ) -> None:
        """#SENSE values should be exactly a single zero token."""
        content = gxf_content_with_sense_value(value)
        with pytest.raises(ValueError, match="Invalid value for key '#SENSE'.*0"):
            GXFData.from_file(gxf_stream(content))

    @pytest.mark.parametrize("tail", ["", "#GRID\n1 2 3 4"])
    def test_sense_key_without_value_raises(
        self, tail: str, gxf_content_with_sense_tail: Callable[[str], str]
    ) -> None:
        """#SENSE without a value should raise a missing value error."""
        content = gxf_content_with_sense_tail(tail)
        with pytest.raises(ValueError, match="Missing value for key '#SENSE'"):
            GXFData.from_file(gxf_stream(content))

    @pytest.mark.parametrize(
        "invalid_key",
        ["#points", "#Rows", "#PtSeparation", "#grid", "##xmax", "##XMax"],
    )
    def test_lowercase_and_mixed_case_keys_raise(
        self, invalid_key: str, gxf_content_with_invalid_key_case: Callable[[str], str]
    ) -> None:
        """Lowercase and mixed-case GXF keys should raise parser errors."""
        content = gxf_content_with_invalid_key_case(invalid_key)

        with pytest.raises(ValueError, match="GXF keys must be uppercase"):
            GXFData.from_file(gxf_stream(content))


class TestGXFWriterDetails:
    """Tests for GXF writer output details."""

    def test_header_contains_xtgeo_comment(self) -> None:
        """Output should start with a comment containing 'xtgeo'."""
        gxf = make_gxf_data(rows=1, grid=np.ma.array([[1.0, 2.0]]))
        stream = StringIO()
        gxf.to_file(stream)
        stream.seek(0)
        first_line = stream.readline()
        assert first_line.startswith("!")
        assert "xtgeo" in first_line

    @pytest.mark.parametrize(
        "value, expected, contains_decimal, uses_scientific",
        [
            (42, "42", False, False),
            (-9999, "-9999", False, False),
            (0, "0", False, False),
            (-9999.0, None, True, False),
            (1.0, None, True, False),
            (1e33, None, False, True),
        ],
    )
    def test_format_number(
        self,
        value: int | float,
        expected: str | None,
        contains_decimal: bool,
        uses_scientific: bool,
    ) -> None:
        """_format_number should preserve int and float formatting rules."""
        result = GXFData._format_number(value)

        if expected is not None:
            assert result == expected
        if contains_decimal:
            assert "." in result
        if uses_scientific:
            assert "e" in result or "E" in result

    def test_single_column_grid(self) -> None:
        """A single-column grid (ncol=1) should write one value per row."""
        gxf = make_gxf_data(
            points=1,
            rows=3,
            grid=np.ma.array([[10.0], [20.0], [30.0]]),
        )
        stream = StringIO()
        gxf.to_file(stream)
        stream.seek(0)
        re_read = GXFData.from_file(stream)
        np.testing.assert_allclose(re_read.grid.data, gxf.grid.data)

    def test_all_masked_grid_writes_all_dummy(self) -> None:
        """A fully masked grid should write all dummy values."""
        values = np.ma.array(
            [[1.0, 2.0], [3.0, 4.0]],
            mask=[[True, True], [True, True]],
        )
        gxf = make_gxf_data(grid=values)
        stream = StringIO()
        gxf.to_file(stream)
        stream.seek(0)
        re_read = GXFData.from_file(stream)
        assert re_read.grid.count() == 0
        assert np.all(re_read.grid.mask)


class TestRegularSurfaceYflipRoundtrip:
    """Tests for yflip handling in RegularSurface GXF roundtrips."""

    @pytest.mark.parametrize(
        "surface_overrides, expected_yflip, expected_yinc, expected_rotation",
        [
            pytest.param(
                {
                    "rotation": 0.0,
                    "yflip": -1,
                    "values": np.ma.array(
                        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], mask=False
                    ),
                },
                -1,
                20.0,
                0.0,
                id="negative_yflip",
            ),
            pytest.param(
                {
                    "ncol": 2,
                    "nrow": 2,
                    "xinc": 5.0,
                    "yinc": 5.0,
                    "xori": 0.0,
                    "yori": 0.0,
                    "rotation": 45.0,
                    "yflip": 1,
                    "values": np.ma.array([[10.0, 20.0], [30.0, 40.0]], mask=False),
                },
                1,
                5.0,
                45.0,
                id="positive_yflip",
            ),
        ],
    )
    def test_yflip_roundtrip(
        self,
        surface_overrides: dict[str, object],
        expected_yflip: int,
        expected_yinc: float,
        expected_rotation: float,
    ) -> None:
        """A surface yflip should survive GXF export/import."""
        surf = make_regular_surface(**surface_overrides)
        stream = BytesIO()
        surf.to_file(stream, fformat="gxf")
        stream.seek(0)

        re_read = xtgeo.surface_from_file(stream, fformat="gxf")
        assert re_read.yflip == expected_yflip
        assert re_read.yinc == pytest.approx(expected_yinc)
        assert re_read.rotation == pytest.approx(expected_rotation)
        np.testing.assert_allclose(
            re_read.values.filled(np.nan),
            surf.values.filled(np.nan),
            equal_nan=True,
        )

    def test_negative_yflip_writes_negative_rwseparation(self) -> None:
        """Negative RegularSurface yflip should write negative #RWSEPARATION."""
        surf = make_regular_surface(yflip=-1)
        stream = BytesIO()

        surf.to_file(stream, fformat="gxf")

        lines = stream.getvalue().decode().splitlines()
        rwseparation_index = lines.index("#RWSEPARATION")
        assert lines[rwseparation_index + 1] == "-20.0"
