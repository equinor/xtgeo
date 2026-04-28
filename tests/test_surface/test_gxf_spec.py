import shutil
from dataclasses import FrozenInstanceError
from io import BytesIO, StringIO
from pathlib import Path

import numpy as np
import pytest

import xtgeo
from xtgeo.surface._regsurf_gxf_parser import GXFData


@pytest.fixture()
def gxf_path(testdata_path: str) -> Path:
    """Return the path to a GXF test file."""
    p = Path(testdata_path) / "surfaces/etc/fdata_test.gxf"
    if not p.exists():
        pytest.skip(f"Test data file not found: {p}")  # pragma: no cover
    return p


def gxf_stream(content: str) -> StringIO:
    return StringIO(content)


@pytest.fixture
def valid_gxf_content() -> str:
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
##XMAX
"9999.0"
##YMAX
"8888.0"
#GRID
1.0 2.0 3.0
4.0 9999999.0 6.0
"""


@pytest.fixture
def gxf_content_missing_grid() -> str:
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
"""


@pytest.fixture
def gxf_content_minimal_only() -> str:
    return """
#POINTS
"3"
#ROWS
"2"
#GRID
1 2 3 4 5 6
"""


@pytest.fixture
def gxf_content_no_dummy() -> str:
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
#GRID
1 2 3 4
"""


@pytest.fixture
def gxf_content_with_unknown_key() -> str:
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
def gxf_content_duplicate_points() -> str:
    return """
#POINTS
"3"
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
1 2 3 4 5 6
"""


@pytest.fixture
def gxf_content_with_sense_key() -> str:
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
#SENSE
"1"
#GRID
1 2 3 4
"""


@pytest.fixture
def gxf_content_valid_unquoted() -> str:
    return """#POINTS
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
1 2 3 4
"""


@pytest.fixture
def gxf_content_missing_rows() -> str:
    return """
#POINTS
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
1 2 3 4
"""


@pytest.fixture
def non_gxf_content() -> str:
    return """
This is not a GXF key
"""


@pytest.fixture
def gxf_content_small_grid() -> str:
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


class TestGXFParsing:
    """Tests for reading and parsing GXF content."""

    def test_valid_with_extension_keys(self, valid_gxf_content: str) -> None:
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

    def test_from_file_path_input(self, tmp_path, valid_gxf_content: str) -> None:
        path = tmp_path / "surface.gxf"
        path.write_text(valid_gxf_content)

        result = GXFData.from_file(path)

        assert result.points == 3

    def test_nonexistent_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError, match="does not exist"):
            GXFData.from_file("this_file_does_not_exist.gxf")

    @pytest.mark.parametrize("count", [5, 7])
    def test_grid_value_count_must_match_ncol_times_nrow(self, count: int) -> None:
        values = " ".join(str(i) for i in range(count))
        content = f"""
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

        with pytest.raises(ValueError, match="Number of values in #GRID section"):
            GXFData.from_file(gxf_stream(content))

    def test_missing_mandatory_key_raises(self, gxf_content_missing_grid: str) -> None:
        """Missing #GRID (a truly required key) must raise."""

        with pytest.raises(ValueError, match="Missing mandatory key"):
            GXFData.from_file(gxf_stream(gxf_content_missing_grid))

    def test_optional_keys_default_values_assigned(
        self, gxf_content_minimal_only: str
    ) -> None:
        """When optional keys are missing, defaults are applied."""

        result = GXFData.from_file(gxf_stream(gxf_content_minimal_only))

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

    def test_no_dummy_means_no_masking(self, gxf_content_no_dummy: str) -> None:
        """Without #DUMMY, no values should be masked."""

        result = GXFData.from_file(gxf_stream(gxf_content_no_dummy))

        assert result.grid.count() == 4
        assert not np.any(result.grid.mask)

    def test_unknown_single_hash_key_warns_and_skips(
        self, gxf_content_with_unknown_key: str
    ) -> None:

        result = GXFData.from_file(gxf_stream(gxf_content_with_unknown_key))

        assert result.points == 3
        assert result.rows == 2

    def test_duplicate_key_raises(self, gxf_content_duplicate_points: str) -> None:
        with pytest.raises(ValueError, match="Duplicate key"):
            GXFData.from_file(gxf_stream(gxf_content_duplicate_points))

    def test_missing_grid_key_raises(self, gxf_content_missing_grid: str) -> None:
        with pytest.raises(ValueError, match="Missing mandatory key '#GRID'"):
            GXFData.from_file(gxf_stream(gxf_content_missing_grid))

    def test_sense_key_warns_about_orientation(
        self, gxf_content_with_sense_key: str
    ) -> None:
        """#SENSE key should produce a specific orientation warning."""
        with pytest.warns(UserWarning, match="SENSE.*orientation"):
            result = GXFData.from_file(gxf_stream(gxf_content_with_sense_key))

        assert result.points == 2

    @pytest.mark.parametrize(
        "trailing_key",
        [
            "#POINTS",
            "#ROWS",
            "#PTSEPARATION",
            "#RWSEPARATION",
            "#XORIGIN",
            "#YORIGIN",
            "#ROTATION",
            "#DUMMY",
        ],
    )
    def test_valid_key_after_grid_raises(self, trailing_key: str) -> None:
        """A valid key appearing inside the #GRID section must raise."""
        content = f"""
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
{trailing_key}
"1"
3 4
"""
        with pytest.raises(ValueError, match="Unexpected key.*inside #GRID"):
            GXFData.from_file(gxf_stream(content))

    def test_extension_key_after_grid_raises(self) -> None:
        """An extension key (##) inside the #GRID section must also raise."""
        content = """
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
##XMAX
"9999"
3 4
"""
        with pytest.raises(ValueError, match="Unexpected key.*inside #GRID"):
            GXFData.from_file(gxf_stream(content))

    def test_valid_key_after_complete_grid_raises(self) -> None:
        """
        A key after all grid values are present still raises (inside #GRID section).
        """
        content = """
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
1 2 3 4
#ROTATION
"45"
"""
        with pytest.raises(ValueError, match="Unexpected key.*inside #GRID"):
            GXFData.from_file(gxf_stream(content))

    def test_comment_inside_grid_is_allowed(self) -> None:
        """Comments (lines starting with !) inside #GRID should be skipped."""
        content = """
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
        result = GXFData.from_file(gxf_stream(content))
        assert result.grid.count() == 4
        expected = np.array([[1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_allclose(result.grid.data, expected)


class TestGXFFileRoundtrip:
    """Tests for write-then-read roundtrips via file streams."""

    def test_roundtrip_stringio(self) -> None:
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
        values = np.ma.array(
            [[1.0, 2.0], [3.0, 9999.0]],
            mask=[[False, False], [False, True]],
        )
        gxf = GXFData(
            points=2,
            rows=2,
            ptseparation=1.0,
            rwseparation=1.0,
            xorigin=0.0,
            yorigin=0.0,
            rotation=0.0,
            dummy=9999.0,
            grid=values,
        )

        stream = BytesIO()
        gxf.to_file(stream)
        stream.seek(0)

        re_read = GXFData.from_file(stream)

        np.testing.assert_allclose(re_read.grid.data, gxf.grid.data)
        np.testing.assert_array_equal(re_read.grid.mask, gxf.grid.mask)

    def test_roundtrip_wide_grid(self) -> None:
        """Roundtrip a grid wide enough to require line wrapping."""
        ncol = 40
        nrow = 2
        values = np.ma.arange(nrow * ncol, dtype=np.float64).reshape((nrow, ncol))
        gxf = GXFData(
            points=ncol,
            rows=nrow,
            ptseparation=1.0,
            rwseparation=1.0,
            xorigin=0.0,
            yorigin=0.0,
            rotation=0.0,
            dummy=-999.0,
            grid=values,
        )

        stream = StringIO()
        gxf.to_file(stream)
        stream.seek(0)

        re_read = GXFData.from_file(stream)
        assert re_read.points == ncol
        assert re_read.rows == nrow
        np.testing.assert_allclose(re_read.grid.data, gxf.grid.data)


class TestGXFWriter:
    """Tests for GXF output format constraints."""

    def test_line_length_at_most_80_chars(self) -> None:
        """GXF spec requires all lines <= 80 characters."""
        ncol = 20
        nrow = 3
        values = np.ma.arange(nrow * ncol, dtype=np.float64).reshape((nrow, ncol))
        gxf = GXFData(
            points=ncol,
            rows=nrow,
            ptseparation=1.0,
            rwseparation=1.0,
            xorigin=0.0,
            yorigin=0.0,
            rotation=0.0,
            dummy=-999.0,
            grid=values,
        )

        stream = StringIO()
        gxf.to_file(stream)
        stream.seek(0)
        for line in stream:
            assert len(line.rstrip("\n")) <= 80

        stream.seek(0)
        re_read = GXFData.from_file(stream)
        np.testing.assert_allclose(re_read.grid.data, gxf.grid.data)

    def test_to_file_nonexistent_folder_raises(self, tmp_path) -> None:
        """Writing to a non-existent folder should raise OSError."""
        gxf = GXFData(
            points=2,
            rows=2,
            ptseparation=1.0,
            rwseparation=1.0,
            xorigin=0.0,
            yorigin=0.0,
            rotation=0.0,
            dummy=-999.0,
            grid=np.ma.array([[1.0, 2.0], [3.0, 4.0]]),
        )
        bad_path = tmp_path / "no_such_folder" / "output.gxf"
        with pytest.raises(OSError):
            gxf.to_file(bad_path)


class TestGXFDataclass:
    """Tests for GXFData dataclass properties."""

    def test_frozen(self) -> None:
        gxf = GXFData(
            points=2,
            rows=2,
            ptseparation=1.0,
            rwseparation=1.0,
            xorigin=0.0,
            yorigin=0.0,
            rotation=0.0,
            dummy=9999.0,
            grid=np.ma.array([[1.0, 2.0], [3.0, 4.0]]),
        )

        with pytest.raises(FrozenInstanceError):
            gxf.points = 3


class TestFileFormatVerification:
    """Tests for file format detection and validation."""

    def test_valid_gxf_file_path(
        self, tmp_path, gxf_content_valid_unquoted: str
    ) -> None:
        """Format verification should succeed for a valid .gxf file."""
        path = tmp_path / "test.gxf"
        path.write_text(gxf_content_valid_unquoted)

        result = GXFData.from_file(path)
        assert result.points == 2
        assert result.rows == 2

    def test_non_gxf_content_raises(
        self, tmp_path, gxf_content_missing_rows: str
    ) -> None:
        """
        Format verification should fail for non-GXF content (missing #ROW).
        Format verification checks for presence of some mandatory keys.
        There is also a check for mandatory keys, but that comes afterwards
        and will not be reached if format verification fails as expected.
        """
        path = tmp_path / "test.gxf"
        path.write_text(gxf_content_missing_rows)

        with pytest.raises(
            ValueError,
            match=r"Missing mandatory keys: \['ROWS'\]",
        ):
            GXFData.from_file(path)

        with pytest.raises(
            ValueError,
            match=r"Missing mandatory keys: \['ROWS'\]\.",
        ):
            xtgeo.surface_from_file(path)

    def test_non_gxf_extension_and_content_raises(
        self, tmp_path, non_gxf_content: str
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

    def test_surface_from_file_no_format_hints(self, gxf_path: Path, tmp_path) -> None:
        """
        Test that a real GXF file with a large number of comments or free text
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

        # File suffix is not .gxf, but a format hint is given
        surf = xtgeo.surface_from_file(outpath, fformat="gxf")
        assert surf.ncol == 330
        assert surf.nrow == 208


class TestDummyTypePreservation:
    """#DUMMY value should retain the type (int or float) from the input."""

    @staticmethod
    def _minimal_gxf(dummy_literal: str) -> str:
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
"{dummy_literal}"
#GRID
1 2 3 {dummy_literal}
"""

    def test_dummy_int_from_file(self) -> None:
        """An integer dummy like -9999 should be parsed as int."""
        content = self._minimal_gxf("-9999")
        result = GXFData.from_file(gxf_stream(content))

        assert result.dummy == -9999
        assert isinstance(result.dummy, int)

    def test_dummy_float_from_file(self) -> None:
        """A float dummy like -9999.0 should be parsed as float."""
        content = self._minimal_gxf("-9999.0")
        result = GXFData.from_file(gxf_stream(content))

        assert result.dummy == pytest.approx(-9999.0)
        assert isinstance(result.dummy, float)

    def test_dummy_float_scientific_from_file(self) -> None:
        """A scientific-notation dummy like 1e33 should be parsed as float."""
        content = self._minimal_gxf("1e33")
        result = GXFData.from_file(gxf_stream(content))

        assert result.dummy == pytest.approx(1e33)
        assert isinstance(result.dummy, float)

    def test_dummy_int_masking(self) -> None:
        """Grid values equal to an int dummy should be masked."""
        content = self._minimal_gxf("-9999")
        result = GXFData.from_file(gxf_stream(content))

        assert result.grid.count() == 3
        assert result.grid.mask.any()

    def test_dummy_float_masking(self) -> None:
        """Grid values equal to a float dummy should be masked."""
        content = self._minimal_gxf("-9999.0")
        result = GXFData.from_file(gxf_stream(content))

        assert result.grid.count() == 3
        assert result.grid.mask.any()

    def test_dummy_int_roundtrip_file(self) -> None:
        """Int dummy type should survive a write-then-read roundtrip."""
        values = np.ma.array(
            [[1.0, 3.0], [2.0, -9999.0]],
            mask=[[False, False], [False, True]],
        )
        gxf = GXFData(
            points=2,
            rows=2,
            ptseparation=1.0,
            rwseparation=1.0,
            xorigin=0.0,
            yorigin=0.0,
            rotation=0.0,
            dummy=-9999,
            grid=values,
        )

        stream = StringIO()
        gxf.to_file(stream)
        stream.seek(0)

        re_read = GXFData.from_file(stream)
        assert re_read.dummy == -9999
        assert isinstance(re_read.dummy, int)
        np.testing.assert_array_equal(re_read.grid.mask, gxf.grid.mask)

    def test_dummy_float_roundtrip_file(self) -> None:
        """Float dummy type should survive a write-then-read roundtrip."""
        values = np.ma.array(
            [[1.0, 3.0], [2.0, -9999.0]],
            mask=[[False, False], [False, True]],
        )
        gxf = GXFData(
            points=2,
            rows=2,
            ptseparation=1.0,
            rwseparation=1.0,
            xorigin=0.0,
            yorigin=0.0,
            rotation=0.0,
            dummy=-9999.0,
            grid=values,
        )

        stream = StringIO()
        gxf.to_file(stream)
        stream.seek(0)

        re_read = GXFData.from_file(stream)
        assert re_read.dummy == pytest.approx(-9999.0)
        assert isinstance(re_read.dummy, float)
        np.testing.assert_array_equal(re_read.grid.mask, gxf.grid.mask)


class TestRegularSurfaceIntegration:
    """Tests for xtgeo RegularSurface GXF integration."""

    def test_from_file(self, valid_gxf_content: str) -> None:

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

    def test_to_file_roundtrip(self) -> None:

        values = np.ma.array(
            [[11.0, 44.0], [22.0, 55.0], [33.0, np.nan]],
            mask=[[False, False], [False, False], [False, True]],
        )
        surf = xtgeo.RegularSurface(
            ncol=3,
            nrow=2,
            xinc=10.0,
            yinc=20.0,
            xori=100.0,
            yori=200.0,
            rotation=15.0,
            values=values,
        )

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

    def test_guess_format_by_extension(
        self, tmp_path, gxf_content_small_grid: str
    ) -> None:
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
    def test_rotation_normalized_to_0_360(self, file_rotation, expected) -> None:
        content = f"""
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
        self, rwseparation, expected_yinc, expected_yflip
    ) -> None:
        content = f"""
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
        surf = xtgeo.surface_from_file(gxf_stream(content), fformat="gxf")
        assert surf.yinc == pytest.approx(expected_yinc)
        assert surf.yflip == expected_yflip


class TestDummyValue:
    """Tests for #DUMMY value handling: masking, edge cases, and propagation."""

    @staticmethod
    def _make_gxf(dummy_literal: str, grid_values: str) -> str:
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

    def test_dummy_zero_masks_zeros(self) -> None:
        """A dummy of 0 should mask grid values that are exactly 0."""
        content = self._make_gxf("0", "1 0 3 4")
        result = GXFData.from_file(gxf_stream(content))

        assert result.dummy == 0
        assert isinstance(result.dummy, int)
        assert result.grid.count() == 3
        # Grid "1 0 3 4" reshaped (rows=2, points=2) -> [[1,0],[3,4]]
        # dummy=0 masks position [0, 1]
        assert result.grid[0, 1] is np.ma.masked

    def test_dummy_negative_value(self) -> None:
        """A negative dummy should mask matching negative grid values."""
        content = self._make_gxf("-999", "10 -999 30 40")
        result = GXFData.from_file(gxf_stream(content))

        assert result.dummy == -999
        assert result.grid.count() == 3

    def test_dummy_large_float(self) -> None:
        """A very large float dummy should correctly mask matching values."""
        content = self._make_gxf("1e30", "1.0 1e30 3.0 4.0")
        result = GXFData.from_file(gxf_stream(content))

        assert result.dummy == pytest.approx(1e30)
        assert isinstance(result.dummy, float)
        assert result.grid.count() == 3

    def test_all_values_are_dummy(self) -> None:
        """When every grid value equals the dummy, all should be masked."""
        content = self._make_gxf("-999", "-999 -999 -999 -999")
        result = GXFData.from_file(gxf_stream(content))

        assert result.grid.count() == 0
        assert np.all(result.grid.mask)

    def test_no_values_match_dummy(self) -> None:
        """When no grid value equals the dummy, nothing should be masked."""
        content = self._make_gxf("9999", "1 2 3 4")
        result = GXFData.from_file(gxf_stream(content))

        assert result.grid.count() == 4
        assert not np.any(result.grid.mask)

    def test_multiple_dummy_occurrences(self) -> None:
        """Multiple grid values equal to the dummy should all be masked."""
        content = self._make_gxf("-999", "-999 2 -999 4")
        result = GXFData.from_file(gxf_stream(content))

        assert result.grid.count() == 2

    def test_default_dummy_when_key_missing(
        self, gxf_content_no_dummy_with_default_value: str
    ) -> None:
        """When #DUMMY is missing, the default (9999999.0) is used."""

        result = GXFData.from_file(gxf_stream(gxf_content_no_dummy_with_default_value))

        assert result.dummy == pytest.approx(9999999.0)
        # The default dummy should mask the matching value
        assert result.grid.count() == 3

    def test_direct_construction_masks_dummy(self) -> None:
        """Constructing GXFData directly should mask values equal to dummy."""
        values = np.ma.array(
            [[1.0, 3.0], [2.0, -999.0]],
        )
        gxf = GXFData(
            points=2,
            rows=2,
            ptseparation=1.0,
            rwseparation=1.0,
            xorigin=0.0,
            yorigin=0.0,
            rotation=0.0,
            dummy=-999.0,
            grid=values,
        )

        assert gxf.grid[1, 1] is np.ma.masked
        assert gxf.grid.count() == 3

    def test_direct_construction_preserves_existing_mask(self) -> None:
        """When grid already has a mask, __post_init__ should keep it."""
        values = np.ma.array(
            [[1.0, 3.0], [2.0, 5.0]],
            mask=[[True, False], [False, False]],
        )
        gxf = GXFData(
            points=2,
            rows=2,
            ptseparation=1.0,
            rwseparation=1.0,
            xorigin=0.0,
            yorigin=0.0,
            rotation=0.0,
            dummy=-999.0,
            grid=values,
        )

        # The pre-existing mask on [0,0] should still be there
        assert gxf.grid[0, 0] is np.ma.masked
        assert gxf.grid.count() == 3

    def test_import_gxf_passes_dummy_as_undef(self, tmp_path) -> None:
        """import_gxf should pass the dummy value as 'undef' in args dict."""
        content = self._make_gxf("-9999.0", "1 2 3 -9999.0")
        path = tmp_path / "test.gxf"
        path.write_text(content)

        surf = xtgeo.surface_from_file(path, fformat="gxf")

        # The masked value should propagate as NaN in the RegularSurface
        filled = surf.values.filled(np.nan)
        assert np.isnan(filled).sum() == 1
        assert surf.nactive == 3

    def test_dummy_roundtrip_masking_preserved(self) -> None:
        """Masked values should survive a GXF write-read roundtrip."""
        # Use symmetric off-diagonal values so that the internal
        # transpose in to_file/from_file does not alter data.
        values = np.ma.array(
            [[10.0, 20.0], [20.0, -999.0]],
            mask=[[False, False], [False, True]],
        )
        gxf = GXFData(
            points=2,
            rows=2,
            ptseparation=5.0,
            rwseparation=5.0,
            xorigin=100.0,
            yorigin=200.0,
            rotation=0.0,
            dummy=-999.0,
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


class TestRealGXFData:
    """Tests using the real data file surfaces/etc/fdata_test.gxf."""

    def test_parse_header_values(self, gxf_path: Path) -> None:
        gxf = GXFData.from_file(gxf_path)

        assert gxf.points == 330
        assert gxf.rows == 208
        assert gxf.ptseparation == pytest.approx(30.4800600)
        assert gxf.rwseparation == pytest.approx(30.4800600)
        assert gxf.xorigin == pytest.approx(427391.726575)
        assert gxf.yorigin == pytest.approx(7250373.922731)
        assert gxf.rotation == pytest.approx(66.57993141719481)
        assert gxf.dummy == pytest.approx(9999999.0)

    def test_grid_shape(self, gxf_path: Path) -> None:
        gxf = GXFData.from_file(gxf_path)

        assert gxf.grid.shape == (208, 330)
        assert gxf.grid.size == 208 * 330

    def test_grid_has_masked_values(self, gxf_path: Path) -> None:
        gxf = GXFData.from_file(gxf_path)

        assert gxf.grid.count() < gxf.grid.size
        assert np.ma.count_masked(gxf.grid) > 0

    def test_grid_valid_values_finite(self, gxf_path: Path) -> None:
        gxf = GXFData.from_file(gxf_path)

        valid = gxf.grid.compressed()
        assert np.all(np.isfinite(valid))
        assert np.all(valid != gxf.dummy)

    def test_roundtrip_preserves_data(self, gxf_path: Path, tmp_path) -> None:
        original = GXFData.from_file(gxf_path)

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

    def test_surface_from_file_metadata(self, gxf_path: Path) -> None:
        surf = xtgeo.surface_from_file(gxf_path)

        assert surf.ncol == 330
        assert surf.nrow == 208
        assert surf.xinc == pytest.approx(30.4800600)
        assert surf.yinc == pytest.approx(30.4800600)
        assert surf.xori == pytest.approx(427391.726575)
        assert surf.yori == pytest.approx(7250373.922731)
        assert surf.rotation == pytest.approx(66.57993141719481)

    def test_surface_from_file_values_shape(self, gxf_path: Path) -> None:
        surf = xtgeo.surface_from_file(gxf_path)

        assert surf.values.shape == (330, 208)
        assert surf.nactive < surf.ncol * surf.nrow

    def test_surface_from_file_no_dummy_in_valid(self, gxf_path: Path) -> None:
        surf = xtgeo.surface_from_file(gxf_path)

        filled = surf.values.filled(np.nan)
        finite_vals = filled[np.isfinite(filled)]
        assert np.all(finite_vals != 9999999.0)
