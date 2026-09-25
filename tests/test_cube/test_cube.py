from __future__ import annotations

import io
import logging
import pathlib

import hypothesis.strategies as st
import numpy as np
import pytest
import segyio
from hypothesis import HealthCheck, given, settings
from xtgeo._cxtgeo import XTGeoCLibError

import xtgeo.cube._cube_import as cube_import
import xtgeo.cube._cube_window_attributes as cube_window_attributes
import xtgeo.cube.cube1 as cube1_module
from xtgeo.common.exceptions import InvalidFileFormatError
from xtgeo.cube import Cube, cube_from_file, cube_from_roxar
from xtgeo.cube._cube_import import (
    _import_segy_all_traces,
    _import_segy_incomplete_traces,
)
from xtgeo.surface.regular_surface import RegularSurface, surface_from_cube
from xtgeo.xyz.polygons import Polygons

logger = logging.getLogger(__name__)

SFILE1 = pathlib.Path("cubes/reek/syntseis_20000101_seismic_depth_stack.segy")
SFILE3 = pathlib.Path("cubes/reek/syntseis_20000101_seismic_depth_stack.storm")
SFILE4 = pathlib.Path("cubes/etc/ib_test_cube2.segy")
SFILE5 = pathlib.Path("cubes/etc/ex2_complete_3d_75x71x26.segy")
SFILE6 = pathlib.Path("cubes/etc/ex1_missing_traces_75x71x26.segy")


_EXPECTED_ATTRS = {
    "min",
    "max",
    "mean",
    "var",
    "rms",
    "maxpos",
    "maxneg",
    "maxabs",
    "meanpos",
    "meanneg",
    "meanabs",
    "sumpos",
    "sumneg",
    "sumabs",
    "upper",
    "lower",
}


# =========================================================================
# Shared fixtures and helpers
# =========================================================================


@pytest.fixture(name="smallcube")
def fixture_smallcube() -> Cube:
    """Fixture for making a small cube"""
    return Cube(ncol=3, nrow=2, nlay=5, xinc=10, yinc=10, zinc=1)


@pytest.fixture(name="loadsfile1")
def fixture_loadsfile1(testdata_path: str) -> Cube:
    """Fixture for loading a SFILE1"""
    return cube_from_file(testdata_path / SFILE1)


def _make_cube(seed: int = 42) -> Cube:
    """Create a small reproducible cube with positive and negative values."""
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(6, 5, 20)).astype(np.float32)
    return Cube(
        ncol=6,
        nrow=5,
        nlay=20,
        xinc=25.0,
        yinc=25.0,
        zinc=2.0,
        zori=1000.0,
        values=values,
    )


def _make_fence(npoints: int = 15) -> np.ndarray:
    """Build a straight X, Y, Z, HLEN fence inside the cube from _make_cube."""
    return np.column_stack(
        [
            np.linspace(10.0, 110.0, npoints),  # X
            np.full(npoints, 50.0),  # Y
            np.zeros(npoints),  # Z (unused by the sampler)
            np.linspace(0.0, 100.0, npoints),  # HLEN
        ]
    ).astype(np.float64)


# =========================================================================
# cube1.py - construction and value normalization
# =========================================================================


def test_create() -> None:
    """Create default cube instance."""
    xcu = Cube(ncol=5, nrow=3, nlay=4, xinc=10, yinc=10, zinc=1)
    assert xcu.ncol == 5, "NCOL"
    assert xcu.nrow == 3, "NROW"
    vec = xcu.values
    xdim, _ydim, _zdim = vec.shape
    assert xdim == 5, "NX from numpy shape "
    assert xcu.zslices.tolist() == [0, 1, 2, 3]


@pytest.mark.parametrize(
    "input, behaviour",
    [
        (np.ones((2, 3, 1)), None),
        (np.ma.ones((2, 3, 1)), UserWarning),
        (np.ones((2, 3, 99)), ValueError),
        (np.ones((3, 2, 1)), ValueError),
        (np.ones((3, 2)), ValueError),
        (np.ones((1, 6)), ValueError),
        ([1, 2, 3, 4, 5, 6], None),
        ([1, 2, 3, 4, 5, 6, 7], ValueError),
        (99, None),
        ("a", ValueError),
    ],
    ids=[
        "np array",
        "masked np array (warn)",
        "np array right dims but wrong size (err)",
        "np array right dims but flipped row col (err)",
        "np array wrong dims as 2D ex 1 (err)",
        "np array wrong dims as 2D ex 2 (err)",
        "list",
        "list_wrong_length (err)",
        "scalar",
        "letter (err)",
    ],
)
def test_create_cube_with_values(
    input: np.ndarray | list | int | str, behaviour: type[BaseException] | None
) -> None:
    """Create cube with various input values, both correct and incorrect formats."""

    if behaviour is None:
        Cube(ncol=2, nrow=3, nlay=1, xinc=1, yinc=1, zinc=1, values=input)  # type: ignore[arg-type]

    elif behaviour is UserWarning:
        with pytest.warns(behaviour):
            Cube(ncol=2, nrow=3, nlay=1, xinc=1, yinc=1, zinc=1, values=input)  # type: ignore[arg-type]

    elif behaviour is ValueError:
        with pytest.raises(behaviour):
            Cube(ncol=2, nrow=3, nlay=1, xinc=1, yinc=1, zinc=1, values=input)  # type: ignore[arg-type]


def test_values_none_and_bool_default_to_zero() -> None:
    """Test that None and bool value inputs default to a zero cube.

    What is tested:
        Cubes are created with ``values=None`` and with ``values=True``.

    Expected behaviour:
        Both cubes contain only zeros.
    """
    cube_none = Cube(ncol=2, nrow=2, nlay=2, xinc=1, yinc=1, zinc=1, values=None)
    assert bool((cube_none.values == 0.0).all())

    cube_bool = Cube(ncol=2, nrow=2, nlay=2, xinc=1, yinc=1, zinc=1, values=True)
    assert bool((cube_bool.values == 0.0).all())


def test_values_non_contiguous_array_is_made_contiguous() -> None:
    """Test that non-C-contiguous input values are converted to C-contiguous.

    What is tested:
        A Fortran-ordered ndarray is passed as ``values`` to the Cube constructor.

    Expected behaviour:
        Stored cube values are C-contiguous and equal to the input values.
    """
    values = np.asfortranarray(np.arange(24, dtype=np.float32).reshape(2, 3, 4))
    assert not values.flags.c_contiguous

    cube = Cube(ncol=2, nrow=3, nlay=4, xinc=1, yinc=1, zinc=1, values=values)

    assert cube.values.flags.c_contiguous
    np.testing.assert_array_equal(cube.values, values)


# =========================================================================
# cube1.py - geometry, metadata, and properties
# =========================================================================


def test_geometry_setters_update_values() -> None:
    """Test that the geometry property setters update stored values.

    What is tested:
        The ``xori``, ``yori``, ``zori``, ``xinc``, ``yinc``, ``zinc`` and
        ``rotation`` setters are each assigned a new value.

    Expected behaviour:
        Each property returns the newly assigned value.
    """
    cube = Cube(ncol=2, nrow=2, nlay=2, xinc=1, yinc=1, zinc=1)

    cube.xori = 10.0
    cube.yori = 20.0
    cube.zori = 30.0
    cube.xinc = 2.0
    cube.yinc = 3.0
    cube.zinc = 4.0
    cube.rotation = 45.0

    assert cube.xori == 10.0
    assert cube.yori == 20.0
    assert cube.zori == 30.0
    assert cube.xinc == 2.0
    assert cube.yinc == 3.0
    assert cube.zinc == 4.0
    assert cube.rotation == 45.0


def test_ilines_xlines_traceidcodes_setters() -> None:
    """Test the ilines, xlines and traceidcodes setters.

    What is tested:
        ``ilines`` and ``xlines`` are set from arrays, and ``traceidcodes`` is set
        both from a scalar and from a 2D array.

    Expected behaviour:
        The line vectors reflect the assigned arrays, the scalar assignment fills the
        whole traceidcodes grid, and the array assignment is stored as given.
    """
    cube = Cube(ncol=2, nrow=2, nlay=2, xinc=1, yinc=1, zinc=1)

    cube.ilines = np.array([5, 6], dtype=np.int32)
    cube.xlines = np.array([7, 8], dtype=np.int32)
    assert cube.ilines.tolist() == [5, 6]
    assert cube.xlines.tolist() == [7, 8]

    cube.traceidcodes = 3  # scalar -> filled array
    np.testing.assert_array_equal(cube.traceidcodes, np.full((2, 2), 3))

    cube.traceidcodes = np.array([[1, 2], [3, 4]], dtype=np.int32)
    assert cube.traceidcodes.tolist() == [[1, 2], [3, 4]]


def test_metadata_setter_rejects_wrong_type() -> None:
    """Test that the metadata setter validates the object type.

    What is tested:
        A non ``MetaDataRegularCube`` object is assigned to ``metadata``.

    Expected behaviour:
        A ValueError is raised indicating the object is of the wrong type.
    """
    cube = _make_cube()

    with pytest.raises(ValueError, match="not an instance of MetaDataRegularCube"):
        cube.metadata = "not a metadata object"  # type: ignore[assignment]


def test_metadata_setter_accepts_existing_metadata() -> None:
    """Test that metadata setter accepts a valid MetaDataRegularCube instance.

    What is tested:
        The existing metadata object is assigned back via the metadata setter.

    Expected behaviour:
        Assignment succeeds and metadata references the assigned object.
    """
    cube = _make_cube()
    metadata = cube.metadata

    cube.metadata = metadata

    assert cube.metadata is metadata


def test_zflip_and_filesrc_getters() -> None:
    """Test read-only zflip and filesrc getters.

    What is tested:
        ``zflip`` and ``filesrc`` are read from a freshly created cube.

    Expected behaviour:
        ``zflip`` defaults to 1 and ``filesrc`` defaults to None.
    """
    cube = Cube(ncol=2, nrow=2, nlay=2, xinc=1, yinc=1, zinc=1)

    assert cube.zflip == 1
    assert cube.filesrc is None


# =========================================================================
# cube1.py - hashing, description, copying, and dead traces
# =========================================================================


def test_generate_hash_is_deterministic_and_sensitive() -> None:
    """Test that generate_hash is deterministic and content/method sensitive.

    What is tested:
        ``generate_hash`` is called twice on the same cube, with two different hash
        methods, and on a second cube built from different values.

    Expected behaviour:
        Repeated calls give an identical hash, different methods give different
        hashes, and a different cube gives a different hash.
    """
    cube = _make_cube()

    assert cube.generate_hash() == cube.generate_hash()
    assert cube.generate_hash("sha256") != cube.generate_hash("md5")

    other = _make_cube(seed=7)
    assert cube.generate_hash() != other.generate_hash()


def test_describe_flush_and_return(capsys: pytest.CaptureFixture[str]) -> None:
    """Test the describe method for both flush modes.

    What is tested:
        ``describe`` is called with ``flush=False`` (return text) and with
        ``flush=True`` (print to stdout).

    Expected behaviour:
        The ``flush=False`` call returns a descriptive string, and the ``flush=True``
        call prints the description and returns None.
    """
    cube = _make_cube()

    text = cube.describe(flush=False)
    assert isinstance(text, str)
    assert "Description of Cube instance" in text
    assert capsys.readouterr().out == ""

    assert cube.describe(flush=True) is None
    assert "Description of Cube instance" in capsys.readouterr().out


def test_copy_is_independent_deep_copy() -> None:
    """Test that copy returns an independent deep copy.

    What is tested:
        A cube is copied with ``copy`` and the clone's values are mutated in place.

    Expected behaviour:
        The clone is a distinct object with equal dimensions and values, and mutating
        the clone does not affect the original.
    """
    cube = Cube(
        ncol=3, nrow=2, nlay=5, xinc=10, yinc=10, zinc=1, values=list(range(30))
    )

    clone = cube.copy()

    assert clone is not cube
    assert clone.dimensions == cube.dimensions
    np.testing.assert_array_equal(clone.values, cube.values)

    original_values = cube.values.copy()
    clone.values[:] += 1.0
    np.testing.assert_array_equal(cube.values, original_values)
    np.testing.assert_array_equal(clone.values, original_values + 1.0)


def test_values_dead_traces_sets_value_and_returns_average() -> None:
    """Test that values_dead_traces overwrites dead-trace values.

    What is tested:
        Two traces are flagged as dead (traceidcode 2) and ``values_dead_traces`` is
        called with a replacement value.

    Expected behaviour:
        The midpoint of the original dead-trace minimum and maximum is returned,
        dead-trace samples equal the new value, and live traces are unchanged.
    """
    cube = Cube(ncol=2, nrow=2, nlay=2, xinc=1, yinc=1, zinc=1, values=list(range(8)))
    cube.traceidcodes = [1, 2, 1, 2]

    result = cube.values_dead_traces(999.0)

    assert result == pytest.approx(4.5)
    traceidcodes = np.asarray(cube.traceidcodes)
    np.testing.assert_array_equal(
        cube.values[traceidcodes == 2], np.full((2, 2), 999.0)
    )
    np.testing.assert_array_equal(cube.values[traceidcodes == 1], [[0, 1], [4, 5]])


def test_values_dead_traces_without_dead_traces_returns_none() -> None:
    """Test values_dead_traces when there are no dead traces.

    What is tested:
        ``values_dead_traces`` is called on a cube whose traceidcodes are all alive.

    Expected behaviour:
        The method returns None.
    """
    cube = Cube(ncol=2, nrow=2, nlay=2, xinc=1, yinc=1, zinc=1)

    assert cube.values_dead_traces(0.0) is None


# =========================================================================
# cube1.py / _cube_import.py - file import and format validation
# =========================================================================


def test_import_wrong_format(tmp_path: pathlib.Path) -> None:
    (tmp_path / "test.EGRID").write_text("hello")
    with pytest.raises(ValueError, match="File format"):
        cube_from_file(tmp_path / "test.EGRID", fformat="egrid")


indices = st.integers(min_value=2, max_value=4)
coordinates = st.floats(min_value=-100, max_value=100, allow_nan=False)
increments = st.floats(min_value=1, max_value=2, allow_nan=False)

cubes = st.builds(Cube, *([indices] * 3), *([increments] * 3), *([coordinates] * 3))


@given(cubes)
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_import_guess_segy(tmp_path: pathlib.Path, cube: Cube) -> None:
    filepath = tmp_path / "test.segy"
    cube.to_file(filepath)
    cube2 = cube_from_file(filepath)
    assert cube.xori == pytest.approx(cube2.xori, abs=1)
    assert cube.yori == pytest.approx(cube2.yori, abs=1)
    assert cube.zori == pytest.approx(cube2.zori, abs=1)
    assert cube.ncol == cube2.ncol
    assert cube.nrow == cube2.nrow
    assert cube.nlay == cube2.nlay


def test_import_segy_garbage_raises_oserror(tmp_path: pathlib.Path) -> None:
    """Test that a non-SEGY file surfaces an OSError.

    What is tested:
        ``cube_from_file`` is called with ``fformat="segy"`` on a file containing
        garbage bytes.

    Expected behaviour:
        An OSError explaining the SEGY parse failure is raised.
    """
    bad = tmp_path / "bad.segy"
    bad.write_bytes(b"this is not a valid segy file" * 100)

    with pytest.raises(OSError, match="Cannot parse SEGY"):
        cube_from_file(bad, fformat="segy")


def test_import_xtgregcube_invalid_magic_raises(tmp_path: pathlib.Path) -> None:
    """Test that an xtgregcube file with a bad header is rejected.

    What is tested:
        ``cube_from_file`` is called with ``fformat="xtgregcube"`` on a file whose
        header has an invalid swap id / magic number.

    Expected behaviour:
        A ValueError about the invalid file format is raised.
    """
    bad = tmp_path / "bad.xtgregcube"
    bad.write_bytes(b"\x00" * 64)

    with pytest.raises(ValueError, match="Invalid file format"):
        cube_from_file(bad, fformat="xtgregcube")


# =========================================================================
# cube1.py / _cube_roxapi.py - Roxar import metadata
# =========================================================================


def test_cube_from_roxar_sets_metadata_required(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that cube_from_roxar sets metadata.required on the returned cube.

    What is tested:
        ``cube_from_roxar`` is invoked with Roxar import patched to a no-op.

    Expected behaviour:
        A Cube instance is returned and its required metadata matches its dimensions.
    """
    monkeypatch.setattr(
        cube1_module._cube_roxapi,
        "import_cube_roxapi",
        lambda obj, project, name, folder=None: None,
    )

    cube = cube_from_roxar(project=object(), name="dummy", folder="a/b")

    assert isinstance(cube, Cube)
    assert cube.metadata.required["ncol"] == cube.ncol
    assert cube.metadata.required["nrow"] == cube.nrow
    assert cube.metadata.required["nlay"] == cube.nlay


# =========================================================================
# _cube_import.py / _cube_export.py - SEGY round-trip and Storm import
# =========================================================================


def test_segy_export_import(tmp_path: pathlib.Path, testdata_path: str) -> None:
    cube = cube_from_file(testdata_path / SFILE4)
    assert cube.zinc == 4

    # change to study rounding effect
    cube.zinc = 3.99
    fname = tmp_path / "small1.segy"
    cube.to_file(fname, fformat="segy")
    cube2 = cube_from_file(fname)

    np.testing.assert_equal(cube.values, cube2.values)
    assert cube.zinc == pytest.approx(cube2.zinc)
    assert cube.xinc == pytest.approx(cube2.xinc, abs=0.01)
    assert cube.yinc == pytest.approx(cube2.yinc, abs=0.01)
    assert cube.ncol == cube2.ncol
    assert cube.nrow == cube2.nrow
    assert cube.nlay == cube2.nlay
    assert cube.xori == cube2.xori
    assert cube.yori == cube2.yori
    assert cube.zori == cube2.zori
    assert cube.ilines.all() == cube2.ilines.all()


def test_storm_import(testdata_path: str) -> None:
    """Import Cube using Storm format (case Reek)."""

    acube = cube_from_file(testdata_path / SFILE3, fformat="storm")
    assert acube.ncol == 280, "NCOL"
    vals = acube.values
    assert vals[180, 185, 4] == pytest.approx(0.117074, 0.0001)


# @skipsegyio
# @skiplargetest
def test_segy_import(loadsfile1: Cube) -> None:
    """Import SEGY using internal reader (case 1 Reek)."""

    xcu = loadsfile1

    assert xcu.ncol == 408, "NCOL"

    dim = xcu.values.shape

    assert dim == (408, 280, 70), "Dimensions 3D"

    assert xcu.values.max() == pytest.approx(7.42017, 0.001)


def test_segyio_import(loadsfile1: Cube) -> None:
    """Import SEGY (case 1 Reek) via SegIO library."""

    xcu = loadsfile1

    assert xcu.ncol == 408, "NCOL"
    dim = xcu.values.shape

    assert dim == (408, 280, 70), "Dimensions 3D"
    assert xcu.values.max() == pytest.approx(7.42017, 0.001)


# =========================================================================
# _cube_import.py - value processing and incomplete-trace geometry
# =========================================================================


def test_process_cube_values_converts_to_float32(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that non-float32 cube values are converted to float32.

    What is tested:
        ``_process_cube_values`` is called with a float64 array.

    Expected behaviour:
        The values are converted to float32 and a user warning is emitted.
    """
    warnings: list[str] = []
    monkeypatch.setattr(cube_import.xtg, "warnuser", warnings.append)

    result = cube_import._process_cube_values(np.array([1.0, 2.0], dtype=np.float64))

    assert result.dtype == np.float32
    assert warnings


def test_process_cube_values_with_nan_raises() -> None:
    """Test that NaN values are rejected during cube-value processing.

    What is tested:
        ``_process_cube_values`` is called with an array containing NaN.

    Expected behaviour:
        A ValueError is raised indicating NaN input is unsupported.
    """
    with pytest.raises(ValueError, match="contains NaN values"):
        cube_import._process_cube_values(np.array([1.0, np.nan], dtype=np.float32))


def test_find_long_line_keeps_longest_available_vector() -> None:
    """Test that _find_long_line falls back to the longest available vector.

    What is tested:
        ``_find_long_line`` is called with no vector long enough to reach the
        minimum threshold.

    Expected behaviour:
        The longest vector endpoints found so far are returned.
    """
    result = cube_import._find_long_line({"a": [2, 3], "b": [5, 6, 7]}, nany=10)

    assert result == [5, 7]


def test_find_long_line_empty_input_raises() -> None:
    """Test that _find_long_line rejects empty input.

    What is tested:
        ``_find_long_line`` is called with no candidate vectors.

    Expected behaviour:
        A RuntimeError is raised indicating geometry vectors could not be found.
    """
    with pytest.raises(RuntimeError, match="Not able to get inline or xline vector"):
        cube_import._find_long_line({}, nany=10)


def test_internal_segy_import_full_vs_partial(testdata_path: str) -> None:
    """Test variants of private internal import of segy for compatibility.

    Most prominent, a full SEGY cube should be possible to parse using the same routine
    as used for cubes with missing traces.
    """

    with segyio.open(testdata_path / SFILE5, "r") as f:
        attrs1 = _import_segy_all_traces(f)

    with segyio.open(testdata_path / SFILE5, "r") as f:
        attrs2 = _import_segy_incomplete_traces(f)

    attrs2_by_key = dict(attrs2)
    for key, val in attrs1.items():
        if isinstance(val, np.ndarray):
            np.testing.assert_array_equal(val, attrs2_by_key[key])
        else:
            assert val == attrs2_by_key[key]


def test_segyio_import_ex2(testdata_path: str) -> None:
    """Import small SEGY (ex2 case) via segyio library."""

    cube = cube_from_file(testdata_path / SFILE5)
    assert cube.ncol == 75
    assert list(cube.ilines)[0:4] == [10750, 10752, 10754, 10756]
    assert cube.rotation == pytest.approx(90.0)
    assert cube.xinc == pytest.approx(12.5, abs=0.01)
    assert cube.yinc == pytest.approx(12.5, abs=0.01)


def test_segyio_import_ex1(testdata_path: str) -> None:
    """Import small SEGY (ex1 case with missing traces!) via segyio library."""

    with pytest.warns(UserWarning, match=r"Missing or inconsistent"):
        cube = cube_from_file(testdata_path / SFILE6)
        assert cube.ncol == 75
        assert list(cube.ilines)[0:4] == [11352, 11354, 11356, 11358]
        assert cube.rotation == pytest.approx(90.0)
        assert cube.xinc == pytest.approx(12.5, abs=0.01)
        assert cube.yinc == pytest.approx(12.5, abs=0.01)


# =========================================================================
# cube1.py / _cube_export.py - export formats, headers, and errors
# =========================================================================


@pytest.mark.parametrize("pristine", [True, False])
def test_segyio_import_export(
    tmp_path: pathlib.Path, pristine: bool, smallcube: Cube
) -> None:
    """Import and export SEGY (case 1 Reek) via SegIO library."""
    input_cube = smallcube
    input_cube.values = list(range(30))
    input_cube.to_file(tmp_path / "reek_cube.segy", pristine=pristine)

    # reread that file
    read_cube = cube_from_file(tmp_path / "reek_cube.segy")
    assert input_cube.dimensions == read_cube.dimensions
    assert input_cube.values.flatten().tolist() == read_cube.values.flatten().tolist()


def test_segy_export_to_bytesio_raises() -> None:
    """SEGY cube export to BytesIO is not supported."""
    cube = Cube(ncol=3, nrow=2, nlay=5, xinc=10, yinc=10, zinc=1)
    cube.values = list(range(30))

    stream = io.BytesIO()
    with pytest.raises(TypeError, match="filesystem path"):
        cube.to_file(stream, fformat="segy")


def test_segy_export_sanitizes_text_header_last_byte(
    tmp_path: pathlib.Path,
) -> None:
    """Text header last byte is a safe EBCDIC space after export."""
    cube = Cube(ncol=3, nrow=2, nlay=5, xinc=10, yinc=10, zinc=1)
    cube.values = list(range(30))
    outfile = tmp_path / "cube_text_header_sanitized.segy"

    cube.to_file(outfile, fformat="segy")
    raw = outfile.read_bytes()
    assert raw[3199] == 0x40  # EBCDIC space
    assert raw[3199] != 0x80

    with segyio.open(outfile, "r") as seg:
        assert seg.bin[segyio.BinField.Interval] == 1000
        assert seg.bin[segyio.BinField.Samples] == 5


def test_to_file_rms_regular(tmp_path: pathlib.Path) -> None:
    """Test export to the RMS regular format.

    What is tested:
        A cube is exported with ``to_file(fformat="rms_regular")``.

    Expected behaviour:
        A non-empty file is written to disk.
    """
    cube = Cube(
        ncol=3, nrow=2, nlay=5, xinc=10, yinc=10, zinc=1, values=list(range(30))
    )
    outfile = tmp_path / "cube.rmsreg"

    cube.to_file(outfile, fformat="rms_regular")

    assert outfile.exists()
    assert outfile.stat().st_size > 0


def test_to_file_rms_regular_export_failure_raises(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that RMS regular export failures surface as RuntimeError.

    What is tested:
        The low-level RMS regular exporter is patched to return a non-zero status,
        and ``to_file(fformat="rms_regular")`` is called.

    Expected behaviour:
        A ``RuntimeError`` is raised indicating the RMS regular export failed.
    """
    cube = Cube(
        ncol=3, nrow=2, nlay=5, xinc=10, yinc=10, zinc=1, values=list(range(30))
    )

    monkeypatch.setattr(
        "xtgeo.cube._cube_export._cxtgeo.cube_export_rmsregular",
        lambda *args: 1,
    )

    with pytest.raises(RuntimeError, match="Error when exporting to RMS regular"):
        cube.to_file(tmp_path / "cube_fail.rmsreg", fformat="rms_regular")


def test_to_file_invalid_format_raises(tmp_path: pathlib.Path) -> None:
    """Test that an unsupported export format is rejected.

    What is tested:
        ``to_file`` is called with an unknown ``fformat``.

    Expected behaviour:
        An InvalidFileFormatError is raised.
    """
    cube = Cube(ncol=3, nrow=2, nlay=5, xinc=10, yinc=10, zinc=1)

    with pytest.raises(InvalidFileFormatError):
        cube.to_file(tmp_path / "cube.bogus", fformat="nonsense")


def test_to_file_engine_argument_warns(tmp_path: pathlib.Path) -> None:
    """Test that passing a deprecated ``engine`` argument warns.

    What is tested:
        ``to_file`` is called with an explicit ``engine`` value.

    Expected behaviour:
        A UserWarning about the unsupported ``engine`` argument is emitted while the
        file is still written.
    """
    cube = Cube(
        ncol=3, nrow=2, nlay=5, xinc=10, yinc=10, zinc=1, values=list(range(30))
    )
    outfile = tmp_path / "cube_engine.segy"

    with pytest.warns(UserWarning, match="engine"):
        cube.to_file(outfile, fformat="segy", engine="xtgeo")

    assert outfile.exists()


# =========================================================================
# _cube_utils.py - resampling
# =========================================================================


def test_cube_resampling(loadsfile1: Cube) -> None:
    """Import a cube, then make a smaller and resample, then export the new"""

    logger.info("Import SEGY format via SEGYIO")

    incube = loadsfile1

    newcube = Cube(
        xori=460500,
        yori=5926100,
        zori=1540,
        xinc=40,
        yinc=40,
        zinc=5,
        ncol=200,
        nrow=100,
        nlay=100,
        rotation=incube.rotation,
        yflip=incube.yflip,
    )

    newcube.resample(incube, sampling="trilinear", outside_value=10.0)

    assert newcube.values.mean() == pytest.approx(5.3107, 0.0001)
    assert newcube.values[20, 20, 20] == pytest.approx(10.0, 0.0001)


def test_resample_nearest_default_outside_value() -> None:
    """Test resample with nearest sampling and the default outside_value.

    What is tested:
        A smaller cube that overlaps a larger cube is resampled with the default
        (nearest) sampling and ``outside_value=None``.

    Expected behaviour:
        The target cube keeps its dimensions and contains the matching source samples.
    """
    rng = np.random.default_rng(1)
    incube = Cube(
        ncol=10,
        nrow=10,
        nlay=10,
        xinc=10,
        yinc=10,
        zinc=2,
        xori=0.0,
        yori=0.0,
        zori=1000.0,
        values=rng.normal(size=(10, 10, 10)).astype(np.float32),
    )
    newcube = Cube(
        ncol=5,
        nrow=5,
        nlay=5,
        xinc=10,
        yinc=10,
        zinc=2,
        xori=10.0,
        yori=10.0,
        zori=1004.0,
        yflip=incube.yflip,
        rotation=incube.rotation,
    )

    newcube.resample(incube)

    assert newcube.dimensions == (5, 5, 5)
    np.testing.assert_array_equal(newcube.values, incube.values[1:6, 1:6, 2:7])


def test_resample_warns_when_few_samples_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that resample warns when less than 10 percent is sampled.

    What is tested:
        ``resample`` is called with the low-level routine patched to return ``-4``.

    Expected behaviour:
        A ``RuntimeWarning`` is emitted.
    """
    cube = Cube(ncol=2, nrow=2, nlay=2, xinc=1, yinc=1, zinc=1)
    other = Cube(ncol=2, nrow=2, nlay=2, xinc=1, yinc=1, zinc=1)

    monkeypatch.setattr(
        "xtgeo.cube._cube_utils._cxtgeo.cube_resample_cube",
        lambda *args: -4,
    )

    with pytest.warns(RuntimeWarning, match="Less than 10%"):
        cube.resample(other)


def test_resample_non_overlapping_cubes_raises() -> None:
    """Test that resample raises when the cubes do not overlap.

    What is tested:
        ``resample`` is called with two cubes separated along the X axis.

    Expected behaviour:
        An ``XTGeoCLibError`` is raised indicating no cube overlap.
    """
    cube = Cube(ncol=2, nrow=2, nlay=2, xinc=1, yinc=1, zinc=1, xori=0)
    other = Cube(ncol=2, nrow=2, nlay=2, xinc=1, yinc=1, zinc=1, xori=100)

    with pytest.raises(XTGeoCLibError, match="No cube overlap"):
        cube.resample(other)


# =========================================================================
# _cube_utils.py - thinning
# =========================================================================


def test_cube_thinning(tmp_path: pathlib.Path, loadsfile1: Cube) -> None:
    """Import a cube, then make a smaller by thinning every N line"""

    logger.info("Import SEGY format via SEGYIO")

    incube = loadsfile1
    logger.info(incube)

    # thinning to evey second column and row, but not vertically
    incube.do_thinning(2, 2, 1)
    logger.info(incube)

    incube.to_file(tmp_path / "cube_thinned.segy")

    incube2 = cube_from_file(tmp_path / "cube_thinned.segy")
    logger.info(incube2)


def test_do_thinning_non_integer_raises() -> None:
    """Test that non-integer thinning factors are rejected.

    What is tested:
        ``do_thinning`` is called with a float column factor.

    Expected behaviour:
        A ValueError is raised indicating the input is not integer.
    """
    cube = Cube(ncol=10, nrow=10, nlay=10, xinc=1, yinc=1, zinc=1)

    with pytest.raises(ValueError, match="not integer"):
        cube.do_thinning(2.0, 2, 1)  # type: ignore[arg-type]


def test_do_thinning_too_large_raises() -> None:
    """Test that too-large thinning factors are rejected.

    What is tested:
        ``do_thinning`` is called with factors larger than half the cube range.

    Expected behaviour:
        A ValueError is raised indicating the numbers are too large.
    """
    cube = Cube(ncol=10, nrow=10, nlay=10, xinc=1, yinc=1, zinc=1)

    with pytest.raises(ValueError, match="too large"):
        cube.do_thinning(9, 9, 1)


# =========================================================================
# cube1.py / _cube_utils.py - cropping
# =========================================================================


def test_cube_cropping(loadsfile1: Cube) -> None:
    """Import a cube, then make a smaller by cropping"""

    logger.info("Import SEGY format via SEGYIO")

    incube = loadsfile1
    assert incube.dimensions == (408, 280, 70)
    # thinning to evey second column and row, but not vertically
    incube.do_cropping((2, 13), (10, 22), (30, 0))
    assert incube.dimensions == (393, 248, 40)
    assert incube.values.mean() == pytest.approx(0.0003633049)


@pytest.mark.xfail(
    raises=TypeError,
    reason=(
        "do_cropping(mode='inclusive') computes the crop offsets as numpy.int32 "
        "(arithmetic with the int32 ilines/xlines arrays), which the C routine "
        "cube_xy_from_ij rejects because it expects a Python int."
    ),
    strict=True,
)
def test_do_cropping_inclusive_mode() -> None:
    """Test that inclusive-mode cropping keeps the requested index ranges.

    What is tested:
        ``do_cropping`` is called with ``mode="inclusive"`` using inline, xline and
        depth ranges.

    Expected behaviour:
        The retained inline/xline vectors, dimensions and origin match the requested
        inclusive ranges. This is currently an xfail: inclusive mode passes
        numpy.int32 offsets that the C routine rejects (see the xfail marker).
    """
    cube = Cube(ncol=10, nrow=10, nlay=10, xinc=1, yinc=1, zinc=1, zori=0.0)

    cube.do_cropping((3, 8), (2, 9), (2, 7), mode="inclusive")

    assert cube.ilines.tolist() == [3, 4, 5, 6, 7, 8]
    assert cube.xlines.tolist() == [2, 3, 4, 5, 6, 7, 8, 9]
    assert cube.dimensions == (6, 8, 6)
    assert cube.zori == pytest.approx(2.0)


def test_do_cropping_clib_error_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that do_cropping surfaces C-library failures.

    What is tested:
        ``do_cropping`` is called with low-level XY origin update patched to
        return a non-zero status.

    Expected behaviour:
        A ``RuntimeError`` is raised containing the returned status code.
    """
    cube = Cube(ncol=4, nrow=4, nlay=4, xinc=1, yinc=1, zinc=1)

    monkeypatch.setattr(
        "xtgeo.cube._cube_utils._cxtgeo.cube_xy_from_ij",
        lambda *args: (7, 0.0, 0.0),
    )

    with pytest.raises(RuntimeError, match="code is 7"):
        cube.do_cropping((1, 1), (1, 1), (0, 0))


# =========================================================================
# _cube_utils.py - coordinate lookup
# =========================================================================


def test_cube_get_xy_from_ij(loadsfile1: Cube) -> None:
    """Import a cube, then report XY for a given IJ"""

    logger.info("Checking get xy from IJ")

    incube = loadsfile1

    # thinning to evey second column and row, but not vertically
    xpos, ypos = incube.get_xy_value_from_ij(0, 0, zerobased=True)
    assert xpos == incube.xori
    assert ypos == incube.yori

    xpos, ypos = incube.get_xy_value_from_ij(1, 1, zerobased=False)
    assert xpos == incube.xori
    assert ypos == incube.yori

    xpos, ypos = incube.get_xy_value_from_ij(0, 0, ixline=True)
    assert xpos == incube.xori
    assert ypos == incube.yori

    xpos, ypos = incube.get_xy_value_from_ij(200, 200, zerobased=True)

    assert xpos == pytest.approx(463327.8811957213, 0.01)
    assert ypos == pytest.approx(5933633.598034564, 0.01)


def test_get_xy_value_from_ij_out_of_bounds_raises() -> None:
    """Test that get_xy_value_from_ij rejects out-of-bounds indices.

    What is tested:
        ``get_xy_value_from_ij`` is called with i/j indices outside the cube.

    Expected behaviour:
        A ValueError is raised indicating the index is out of bounds.
    """
    cube = Cube(ncol=3, nrow=3, nlay=2, xinc=1, yinc=1, zinc=1)

    with pytest.raises(ValueError, match="out of bounds"):
        cube.get_xy_value_from_ij(100, 100)


def test_get_xy_value_from_ij_ixline_lookup() -> None:
    """Test get_xy_value_from_ij using inline/xline lookup.

    What is tested:
        ``get_xy_value_from_ij`` is called with ``ixline=True`` and compared with the
        equivalent column/row lookup using labels distinct from positional indices.

    Expected behaviour:
        Both calls return the same coordinate pair.
    """
    cube = Cube(ncol=3, nrow=3, nlay=2, xinc=10, yinc=10, zinc=1)
    cube.ilines = np.array([10, 20, 30], dtype=np.int32)
    cube.xlines = np.array([100, 200, 300], dtype=np.int32)

    by_index = cube.get_xy_value_from_ij(2, 2)
    by_ixline = cube.get_xy_value_from_ij(20, 200, ixline=True)

    assert by_index == pytest.approx(by_ixline)


def test_get_xy_value_from_ij_zero_spacing_raises() -> None:
    """Test that get_xy_value_from_ij rejects degenerate cube geometry.

    What is tested:
        ``get_xy_value_from_ij`` is called away from the origin on a cube
        with zero X and Y spacing.

    Expected behaviour:
        An ``XTGeoCLibError`` is raised for the undefined angle calculation.
    """
    cube = Cube(ncol=3, nrow=3, nlay=2, xinc=0, yinc=0, zinc=1)

    with pytest.raises(XTGeoCLibError, match="Unvalid value for beta"):
        cube.get_xy_value_from_ij(2, 2)


# =========================================================================
# _cube_utils.py - axis-swap round-trip
# =========================================================================


def test_cube_swapaxes(testdata_path: str) -> None:
    """Import a cube, do axes swapping back and forth"""

    logger.info("Import SEGY format via SEGYIO")

    incube = cube_from_file(testdata_path / SFILE4)
    logger.info(incube)
    val1 = incube.values.copy()

    incube.swapaxes()
    logger.info(incube)

    incube.swapaxes()
    val2 = incube.values.copy()
    logger.info(incube)

    np.testing.assert_array_equal(val1, val2)


# =========================================================================
# cube1.py / _cube_utils.py - randomline sampling
# =========================================================================


def test_cube_randomline(show_plot: bool, testdata_path: str) -> None:
    """Import a cube, and compute a randomline given a simple Polygon"""

    incube = cube_from_file(testdata_path / SFILE4)

    poly = Polygons([[778133, 6737650, 2000, 1], [776880, 6738820, 2000, 1]])

    logger.info("Generate random line...")
    hmin, hmax, vmin, vmax, random = incube.get_randomline(poly)

    assert hmin == pytest.approx(-15.7, 0.1)
    assert random.mean() == pytest.approx(-12.5, 0.1)

    if show_plot:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.imshow(
            random,
            cmap="seismic",
            interpolation="sinc",
            extent=(hmin, hmax, vmax, vmin),
        )
        plt.axis("tight")
        plt.colorbar()
        plt.show()


def test_get_randomline_invalid_fencespec_raises() -> None:
    """Test that get_randomline rejects an unsupported fencespec type.

    What is tested:
        ``get_randomline`` is called with a plain string.

    Expected behaviour:
        A ValueError is raised indicating a numpy array or Polygons is required.
    """
    cube = _make_cube()

    with pytest.raises(ValueError, match="must be a numpy or a Polygons"):
        cube.get_randomline("not a fence")  # type: ignore[arg-type]


def test_get_randomline_1d_fence_raises() -> None:
    """Test that get_randomline rejects a non-2D numpy fence.

    What is tested:
        ``get_randomline`` is called with a 1D numpy array.

    Expected behaviour:
        A ValueError is raised indicating the fence is not a 2D numpy.
    """
    cube = _make_cube()

    with pytest.raises(ValueError, match="Fence is not a 2D numpy"):
        cube.get_randomline(np.array([1.0, 2.0, 3.0]))


def test_get_randomline_from_numpy_fence_defaults() -> None:
    """Test get_randomline with a numpy fence and default z-range.

    What is tested:
        ``get_randomline`` is called with a 2D numpy fence and no z arguments, so
        the cube z-origin and z-maximum defaults are used.

    Expected behaviour:
        A 2D array is returned and the vertical range spans the full cube.
    """
    cube = _make_cube()

    hmin, hmax, vmin, vmax, arr = cube.get_randomline(_make_fence())

    assert arr.ndim == 2
    assert vmin == pytest.approx(cube.zori)
    assert vmax == pytest.approx(cube.zori + (cube.nlay - 1) * cube.zinc)


def test_get_randomline_trilinear_with_explicit_z() -> None:
    """Test get_randomline with trilinear sampling and explicit z-range.

    What is tested:
        ``get_randomline`` is called with ``sampling="trilinear"`` and explicit
        ``zmin``, ``zmax`` and ``zincrement``.

    Expected behaviour:
        A 2D array is returned and the vertical range matches the explicit values.
    """
    cube = _make_cube()

    _, _, vmin, vmax, arr = cube.get_randomline(
        _make_fence(),
        zmin=1005.0,
        zmax=1030.0,
        zincrement=1.0,
        sampling="trilinear",
    )

    assert arr.ndim == 2
    assert vmin == pytest.approx(1005.0)
    assert vmax == pytest.approx(1030.0)


def test_get_randomline_from_polygons_with_hincrement() -> None:
    """Test get_randomline from a Polygons fence with an explicit hincrement.

    What is tested:
        ``get_randomline`` is called with a ``Polygons`` fence and an explicit
        ``hincrement`` (exercising the non-default resampling distance branch).

    Expected behaviour:
        The five-element (hmin, hmax, vmin, vmax, ndarray) tuple is returned.
    """
    cube = _make_cube()
    poly = Polygons([[10.0, 50.0, 1000.0, 1], [110.0, 50.0, 1000.0, 1]])

    result = cube.get_randomline(poly, hincrement=25.0)

    assert len(result) == 5
    assert result[4].ndim == 2


# =========================================================================
# _cube_window_attributes.py - window attributes and depth boundaries
# =========================================================================


def test_compute_attributes_in_window_returns_all_maps() -> None:
    """Test that compute_attributes_in_window returns every attribute map.

    What is tested:
        ``compute_attributes_in_window`` is called with two constant depth levels.
        The returned dictionary is inspected for all statistical and sum attribute
        keys plus the ``upper``/``lower`` surfaces, and each value is checked to be a
        ``RegularSurface`` with the cube's map (ncol, nrow) topology.

    Expected behaviour:
        All expected keys are present, every value is a RegularSurface on the cube
        map grid, and no ``max`` map value is below the corresponding ``min`` value.
    """
    cube = _make_cube()

    attrs = cube.compute_attributes_in_window(1010.0, 1030.0)

    assert _EXPECTED_ATTRS.issubset(attrs)
    for surf in attrs.values():
        assert isinstance(surf, RegularSurface)
        assert surf.ncol == cube.ncol
        assert surf.nrow == cube.nrow

    assert np.ma.all(attrs["max"].values >= attrs["min"].values)


def test_compute_attributes_in_window_with_surface_inputs() -> None:
    """Test compute_attributes_in_window with RegularSurface inputs.

    What is tested:
        The upper and lower boundaries are supplied as ``RegularSurface`` instances
        (created via ``surface_from_cube``) instead of constant levels.

    Expected behaviour:
        The resulting ``mean`` and ``rms`` attributes are RegularSurface instances.
    """
    cube = _make_cube()
    upper = surface_from_cube(cube, 1010.0)
    lower = surface_from_cube(cube, 1030.0)

    attrs = cube.compute_attributes_in_window(upper, lower)

    assert isinstance(attrs["mean"], RegularSurface)
    assert isinstance(attrs["rms"], RegularSurface)


def test_compute_attributes_in_window_linear_interpolation() -> None:
    """Test compute_attributes_in_window with linear interpolation.

    What is tested:
        ``compute_attributes_in_window`` is called with ``interpolation="linear"``
        and a custom ``ndiv`` to exercise the linear signal-interpolation path.

    Expected behaviour:
        The ``rms`` attribute is returned as a RegularSurface.
    """
    cube = _make_cube()

    attrs = cube.compute_attributes_in_window(
        1010.0, 1030.0, interpolation="linear", ndiv=5
    )

    assert isinstance(attrs["rms"], RegularSurface)


def test_compute_attributes_in_window_reversed_surfaces_raises() -> None:
    """Test that reversed upper/lower surfaces raise a ValueError.

    What is tested:
        ``compute_attributes_in_window`` is called with the upper level below the
        lower level.

    Expected behaviour:
        A ValueError is raised indicating the upper surface is below the lower.
    """
    cube = _make_cube()

    with pytest.raises(ValueError, match="upper surface is below the lower"):
        cube.compute_attributes_in_window(1030.0, 1010.0)


def test_compute_attributes_in_window_minimum_thickness_too_large_raises() -> None:
    """Test that an over-large minimum thickness raises a ValueError.

    What is tested:
        ``compute_attributes_in_window`` is called with a ``minimum_thickness`` that
        exceeds the window thickness everywhere.

    Expected behaviour:
        A ValueError is raised indicating the minimum thickness is too large.
    """
    cube = _make_cube()

    with pytest.raises(ValueError, match="minimum thickness is too large"):
        cube.compute_attributes_in_window(1010.0, 1030.0, minimum_thickness=10000.0)


def test_compute_attributes_upper_fully_below_cube_raises() -> None:
    """Test that an upper surface entirely below the cube raises a ValueError.

    What is tested:
        ``compute_attributes_in_window`` is called with both levels deeper than the
        cube depth range (so the upper surface is below the cube).

    Expected behaviour:
        A ValueError is raised indicating the upper surface is fully below the cube.
    """
    cube = _make_cube()

    with pytest.raises(ValueError, match="Upper surface is fully below the cube"):
        cube.compute_attributes_in_window(2000.0, 2001.0)


def test_compute_attributes_lower_fully_above_cube_raises() -> None:
    """Test that a lower surface entirely above the cube raises a ValueError.

    What is tested:
        ``compute_attributes_in_window`` is called with both levels shallower than
        the cube.

    Expected behaviour:
        A ValueError is raised indicating the lower surface is fully above the cube.
    """
    cube = _make_cube()

    with pytest.raises(ValueError, match="Lower surface is fully above the cube"):
        cube.compute_attributes_in_window(500.0, 600.0)


def test_compute_attributes_upper_fully_above_cube_warns() -> None:
    """Test the warning when the upper surface is fully above the cube.

    What is tested:
        ``compute_attributes_in_window`` is called with the upper level just above
        the cube top and the lower level within the cube.

    Expected behaviour:
        A UserWarning is emitted and the attribute maps are still returned.
    """
    cube = _make_cube()

    with pytest.warns(UserWarning, match="Upper surface is fully above the cube"):
        attrs = cube.compute_attributes_in_window(990.0, 1030.0)

    assert "mean" in attrs


def test_compute_attributes_lower_fully_below_cube_warns() -> None:
    """Test the warning when the lower surface is fully below the cube.

    What is tested:
        ``compute_attributes_in_window`` is called with the lower level just below
        the cube base and the upper level within the cube.

    Expected behaviour:
        A UserWarning is emitted and the attribute maps are still returned.
    """
    cube = _make_cube()

    with pytest.warns(UserWarning, match="Lower surface is fully below the cube"):
        attrs = cube.compute_attributes_in_window(1010.0, 1050.0)

    assert "mean" in attrs


def test_determine_slice_indices_without_valid_depth_raises_runtime_error() -> None:
    """Test _determine_slice_indices when no valid depth samples exist.

    What is tested:
        ``CubeAttrs._determine_slice_indices`` is called with a depth array where
        all values are outside the valid range.

    Expected behaviour:
        A RuntimeError is raised indicating no valid depth-cube data was found.
    """
    attrs = cube_window_attributes.CubeAttrs.__new__(cube_window_attributes.CubeAttrs)
    attrs._depth_array = np.array([1000.0, 1002.0], dtype=np.float32)
    attrs._outside_depth = 999.0

    with pytest.raises(RuntimeError, match="No valid data found in the depth cube"):
        attrs._determine_slice_indices()


# =========================================================================
# _cube_utils.py - axis-swap values, geometry, and trace metadata
# =========================================================================


def test_swapaxis() -> None:
    cube = Cube(
        xori=0.0,
        yori=0.0,
        zori=0.0,
        ncol=2,
        nrow=2,
        nlay=2,
        xinc=1.0,
        yinc=1.0,
        zinc=1.0,
        yflip=1,
        values=[1, 2, 3, 4, 5, 6, 7, 8],
    )

    assert cube.values.flatten().tolist() == [1, 2, 3, 4, 5, 6, 7, 8]

    cube.swapaxes()

    assert cube.values.flatten().tolist() == [1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0]


def test_swapaxis_traceidcodes() -> None:
    cube = Cube(
        xori=0.0,
        yori=0.0,
        zori=0.0,
        ncol=2,
        nrow=2,
        nlay=2,
        xinc=1.0,
        yinc=1.0,
        zinc=1.0,
        yflip=1,
        values=[1, 2, 3, 4, 5, 6, 7, 8],
    )
    assert cube.traceidcodes.flatten().tolist() == [1, 1, 1, 1]
    cube.traceidcodes = [1, 2, 3, 4]

    cube.swapaxes()

    assert cube.traceidcodes.flatten().tolist() == [1, 3, 2, 4]


@pytest.mark.parametrize(
    "rotation, expected_rotation",
    [
        (-1, 89),
        (0, 90),
        (90, 180),
        (180, 270),
        (270, 0),
        (360, 90),
        (361, 91),
    ],
)
def test_swapaxis_rotation(rotation: int, expected_rotation: int) -> None:
    cube = Cube(
        xori=0.0,
        yori=0.0,
        zori=0.0,
        ncol=2,
        nrow=2,
        nlay=2,
        xinc=1.0,
        yinc=1.0,
        zinc=1.0,
        yflip=1,
        rotation=rotation,
        values=[1, 2, 3, 4, 5, 6, 7, 8],
    )

    cube.swapaxes()

    assert cube.rotation == expected_rotation


def test_swapaxis_ilines() -> None:
    cube = Cube(
        xori=0.0,
        yori=0.0,
        zori=0.0,
        ncol=2,
        nrow=2,
        nlay=2,
        xinc=1.0,
        yinc=1.0,
        zinc=1.0,
        yflip=1,
        values=[1, 2, 3, 4, 5, 6, 7, 8],
    )
    assert cube.ilines.tolist() == [1, 2]

    cube.swapaxes()

    assert cube.ilines.tolist() == [1, 2]


def test_swapaxis_ncol_nrow() -> None:
    cube = Cube(
        xori=0.0,
        yori=0.0,
        zori=0.0,
        ncol=2,
        nrow=3,
        nlay=2,
        xinc=1.0,
        yinc=1.0,
        zinc=1.0,
        yflip=1,
    )

    cube.swapaxes()

    assert (cube.nrow, cube.ncol) == (2, 3)


def test_swapaxis_xinc_yinc() -> None:
    cube = Cube(
        xori=0.0,
        yori=0.0,
        zori=0.0,
        ncol=2,
        nrow=3,
        nlay=2,
        xinc=1.0,
        yinc=2.0,
        zinc=1.0,
        yflip=1,
    )

    cube.swapaxes()

    assert (cube.xinc, cube.yinc) == (2, 1)


# =========================================================================
# _cube_import.py / _cube_export.py - SEGY byte positions and sorting
# =========================================================================


def test_segy_io_roundtrip(tmp_path: pathlib.Path) -> None:
    """Round-trip SEGY with non-standard byte positions (il-byte 193, xl-byte 189).

    Some vendor software writes inline numbers to byte 193 and crossline numbers to
    byte 189 instead of the SEGY-standard positions (189/193).  xtgeo must:

      - read the file correctly when the caller supplies the actual byte positions
      - warn when the default positions are used (file then appears crossline-sorted
        because the physical crosslines occupy the fast-varying byte 189)
      - export to a standard-bytes file that round-trips values and labels correctly
    """
    n_ilines, n_xlines, n_samples = 4, 6, 8
    xori, yori, zori = 500.0, 1000.0, 0.0
    xinc_xl, yinc_il, zinc = 25.0, 50.0, 4.0

    segyfile = tmp_path / "nonstandard_bytes.segy"
    iline_nums = np.arange(1, n_ilines + 1, dtype=np.int32)
    xline_nums = np.arange(1, n_xlines + 1, dtype=np.int32)

    # Write inline numbers to byte 193 and crossline to byte 189.
    # segyio.TraceField.INLINE_3D == 189, CROSSLINE_3D == 193 (fixed byte offsets),
    # so we intentionally assign the values to the *opposite* fields.
    spec = segyio.spec()
    spec.sorting = None  # unstructured — headers written explicitly
    spec.format = 5  # IEEE float32
    spec.samples = np.arange(n_samples, dtype=np.float32) * zinc + zori
    spec.tracecount = n_ilines * n_xlines

    rng = np.random.default_rng(7)
    trace_data = rng.random((n_ilines, n_xlines, n_samples), dtype=np.float32)

    with segyio.create(str(segyfile), spec) as f:
        tr = 0
        for il_idx, il in enumerate(iline_nums):
            for xl_idx, xl in enumerate(xline_nums):
                x = xori + xl_idx * xinc_xl
                y = yori + il_idx * yinc_il
                f.header[tr] = {
                    segyio.TraceField.CROSSLINE_3D: int(il),  # byte 193 ← inline
                    segyio.TraceField.INLINE_3D: int(xl),  # byte 189 ← crossline
                    segyio.TraceField.CDP_X: int(round(x * 100)),
                    segyio.TraceField.CDP_Y: int(round(y * 100)),
                    segyio.TraceField.SourceGroupScalar: -100,
                    segyio.TraceField.TRACE_SAMPLE_INTERVAL: int(zinc * 1000),
                    segyio.TraceField.TRACE_SAMPLE_COUNT: n_samples,
                    segyio.TraceField.DelayRecordingTime: int(zori),
                }
                f.trace[tr] = trace_data[il_idx, xl_idx, :]
                tr += 1
        f.bin[segyio.BinField.Interval] = int(zinc * 1000)
        f.bin[segyio.BinField.Samples] = n_samples

    # --- Correct import: supply the actual byte positions ---
    cube = cube_from_file(str(segyfile), iline=193, xline=189)
    assert cube.ncol == n_ilines
    assert cube.nrow == n_xlines
    assert cube.ilines.tolist() == iline_nums.tolist()
    assert cube.xlines.tolist() == xline_nums.tolist()

    # --- Export to a standard-bytes file and verify round-trip ---
    outfile = tmp_path / "roundtrip_cube.segy"
    cube.to_file(outfile)
    cube_rt = cube_from_file(outfile)
    assert cube_rt.ncol == n_ilines
    assert cube_rt.nrow == n_xlines
    assert cube_rt.ilines.tolist() == iline_nums.tolist()
    np.testing.assert_array_almost_equal(cube.values, cube_rt.values, decimal=4)

    # --- Default bytes: must warn — byte 189 carries xline (fast) values ---
    with pytest.warns(UserWarning, match="crossline-sorted"):
        cube2 = cube_from_file(str(segyfile))
    # With default bytes the labels are transposed: xl values appear as ilines, etc.
    assert cube2.ilines.tolist() == xline_nums.tolist()
    assert cube2.xlines.tolist() == iline_nums.tolist()


def test_segy_crossline_sorted_import_export(tmp_path: pathlib.Path) -> None:
    """Crossline-sorted SEGY must import and re-export with correct orientation.

    segyio.tools.cube() returns (n_xlines, n_ilines, nsamples) for crossline-sorted
    files. xtgeo must transpose to its convention (axis-0 = inlines) so that:
      - exported INLINE_3D / CROSSLINE_3D headers are NOT swapped
      - the CDP_X / CDP_Y for every (INLINE, CROSSLINE) pair is identical to
        the original file

    Uses deliberately asymmetric inline/xline spacings (12.5 vs 25.0) so a
    xinc/yinc swap would produce wrong coordinates, catching the bug the user
    reported (inlines appearing perpendicular in RMS after a round-trip).
    """
    n_ilines, n_xlines, n_samples = 3, 5, 4
    xori, yori, zori = 1000.0, 2000.0, 0.0
    # Deliberately different so a xinc/yinc swap is detectable
    xinc_xline = 12.5  # spacing between adjacent xlines (East direction)
    yinc_inline = 25.0  # spacing between adjacent inlines (North direction)
    zinc = 4.0

    segyfile = tmp_path / "xline_sorted.segy"

    iline_nums = np.arange(1, n_ilines + 1, dtype=np.int32)
    xline_nums = np.arange(1, n_xlines + 1, dtype=np.int32)

    spec = segyio.spec()
    spec.sorting = 1  # crossline sorting
    spec.format = 5
    spec.samples = np.arange(n_samples) * zinc + zori
    spec.ilines = iline_nums
    spec.xlines = xline_nums

    rng = np.random.default_rng(42)
    trace_data = rng.random((n_ilines, n_xlines, n_samples), dtype=np.float32)

    # Build a lookup: (il, xl) -> (cdpx, cdpy) to check round-trip preservation
    original_coords: dict[tuple[int, int], tuple[float, float]] = {}

    with segyio.create(str(segyfile), spec) as f:
        tr = 0
        for xl_idx, xl in enumerate(xline_nums):
            for il_idx, il in enumerate(iline_nums):
                # xlines run East (+X), inlines run North (+Y)
                x = xori + xl_idx * xinc_xline
                y = yori + il_idx * yinc_inline
                original_coords[(int(il), int(xl))] = (x, y)
                f.header[tr] = {
                    segyio.TraceField.INLINE_3D: int(il),
                    segyio.TraceField.CROSSLINE_3D: int(xl),
                    segyio.TraceField.CDP_X: int(round(x * 100)),
                    segyio.TraceField.CDP_Y: int(round(y * 100)),
                    segyio.TraceField.SourceGroupScalar: -100,
                    segyio.TraceField.TRACE_SAMPLE_INTERVAL: int(zinc * 1000),
                    segyio.TraceField.TRACE_SAMPLE_COUNT: n_samples,
                    segyio.TraceField.DelayRecordingTime: int(zori),
                }
                f.trace[tr] = trace_data[il_idx, xl_idx, :]
                tr += 1
        f.bin[segyio.BinField.Interval] = int(zinc * 1000)
        f.bin[segyio.BinField.Samples] = n_samples
        f.bin[segyio.BinField.SortingCode] = 1

    with pytest.warns(UserWarning, match="crossline-sorted"):
        cube = cube_from_file(str(segyfile))

    # After normalising to xtgeo convention: axis-0 = inlines
    assert cube.ncol == n_ilines
    assert cube.nrow == n_xlines
    assert cube.nlay == n_samples
    assert len(cube.ilines) == cube.ncol
    assert len(cube.xlines) == cube.nrow

    outfile = tmp_path / "roundtrip.segy"
    cube.to_file(outfile)

    # Read back exported file and verify every (INLINE, CROSSLINE) pair has
    # the same CDP_X/CDP_Y as the original -- this is what RMS would check.
    with segyio.open(str(outfile), "r", ignore_geometry=True) as f:
        for hdr in f.header:
            il = hdr[segyio.TraceField.INLINE_3D]
            xl = hdr[segyio.TraceField.CROSSLINE_3D]
            scaler = hdr[segyio.TraceField.SourceGroupScalar]
            raw_x = hdr[segyio.TraceField.CDP_X]
            raw_y = hdr[segyio.TraceField.CDP_Y]
            scale = -1.0 / scaler if scaler < 0 else float(scaler)
            exported_x = raw_x * scale
            exported_y = raw_y * scale

            expected_x, expected_y = original_coords[(il, xl)]
            assert exported_x == pytest.approx(expected_x, abs=0.1), (
                f"CDP_X mismatch for IL={il} XL={xl}"
            )
            assert exported_y == pytest.approx(expected_y, abs=0.1), (
                f"CDP_Y mismatch for IL={il} XL={xl}"
            )


@pytest.mark.parametrize(
    "iline, xline",
    [
        (189, 189),
        (193, 193),
        (181, 193),
        (189, 197),
        (0, 0),
    ],
)
def test_invalid_iline_xline_raises(
    tmp_path: pathlib.Path, iline: int, xline: int
) -> None:
    """Only (189, 193) and (193, 189) are accepted for iline/xline."""
    cube = Cube(ncol=2, nrow=2, nlay=2, xinc=1, yinc=1, zinc=1)
    outfile = tmp_path / "tiny.segy"
    cube.to_file(outfile)
    with pytest.raises(
        ValueError, match=r"Only \(189, 193\) and \(193, 189\) are accepted\."
    ):
        cube_from_file(outfile, iline=iline, xline=xline)  # type: ignore[arg-type]


def test_iline_xline_ignored_for_non_segy(
    tmp_path: pathlib.Path, testdata_path: str
) -> None:
    """Passing non-default iline/xline for a non-SEGY file should warn."""
    with pytest.warns(UserWarning, match="only used for SEGY"):
        cube_from_file(testdata_path / SFILE3, fformat="storm", iline=193, xline=189)
