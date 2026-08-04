"""Additional unit tests for Cube methods to improve coverage."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from xtgeo.common.exceptions import InvalidFileFormatError
from xtgeo.cube import Cube, cube_from_file
from xtgeo.surface.regular_surface import RegularSurface, surface_from_cube
from xtgeo.xyz.polygons import Polygons

if TYPE_CHECKING:
    from pathlib import Path


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
# compute_attributes_in_window (src/xtgeo/cube/_cube_window_attributes.py)
# =========================================================================

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


def test_compute_attributes_in_window_returns_all_maps() -> None:
    """Test that compute_attributes_in_window returns every attribute map.

    What is tested:
        ``compute_attributes_in_window`` is called with two constant depth levels.
        The returned dictionary is inspected for all statistical and sum attribute
        keys plus the ``upper``/``lower`` surfaces, and each value is checked to be a
        ``RegularSurface`` with the cube's map (ncol, nrow) topology.

    Expected behaviour:
        All expected keys are present, every value is a RegularSurface on the cube
        map grid, and the mean of the ``max`` map is not below the ``min`` map.
    """
    cube = _make_cube()

    attrs = cube.compute_attributes_in_window(1010.0, 1030.0)

    assert _EXPECTED_ATTRS.issubset(attrs)
    for surf in attrs.values():
        assert isinstance(surf, RegularSurface)
        assert surf.ncol == cube.ncol
        assert surf.nrow == cube.nrow

    # max must never be below min
    assert attrs["max"].values.mean() >= attrs["min"].values.mean()


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


# =========================================================================
# generate_hash, describe, copy (src/xtgeo/cube/cube1.py)
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


def test_describe_flush_and_return() -> None:
    """Test the describe method for both flush modes.

    What is tested:
        ``describe`` is called with ``flush=False`` (return text) and with
        ``flush=True`` (print to stdout).

    Expected behaviour:
        The ``flush=False`` call returns a descriptive string, and the ``flush=True``
        call returns None.
    """
    cube = _make_cube()

    text = cube.describe(flush=False)
    assert isinstance(text, str)
    assert "Description of Cube instance" in text

    assert cube.describe(flush=True) is None


def test_copy_is_independent_deep_copy() -> None:
    """Test that copy returns an independent deep copy.

    What is tested:
        A cube is copied with ``copy`` and the clone's values are then mutated.

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

    clone.values = clone.values + 1.0
    # mutating the clone must not affect the original
    assert not np.array_equal(clone.values, cube.values)


def test_values_dead_traces_sets_value_and_returns_average() -> None:
    """Test that values_dead_traces overwrites dead-trace values.

    What is tested:
        Two traces are flagged as dead (traceidcode 2) and ``values_dead_traces`` is
        called with a replacement value.

    Expected behaviour:
        A non-None average is returned and all dead-trace samples equal the new
        value.
    """
    cube = Cube(ncol=2, nrow=2, nlay=2, xinc=1, yinc=1, zinc=1, values=list(range(8)))
    cube.traceidcodes = [1, 2, 1, 2]

    result = cube.values_dead_traces(999.0)

    assert result is not None
    assert bool((cube.values[cube.traceidcodes == 2] == 999.0).all())


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
# property setters (src/xtgeo/cube/cube1.py)
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
    assert bool((cube.traceidcodes == 3).all())

    cube.traceidcodes = np.array([[1, 2], [3, 4]], dtype=np.int32)
    assert cube.traceidcodes.tolist() == [[1, 2], [3, 4]]


# =========================================================================
# cube1.py API routed via _cube_utils.py
# =========================================================================


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


# =========================================================================
# _cube_utils.py
# =========================================================================


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
        equivalent column/row lookup on a cube with default inline/xline vectors.

    Expected behaviour:
        Both calls return the same coordinate pair.
    """
    cube = Cube(ncol=3, nrow=3, nlay=2, xinc=10, yinc=10, zinc=1)

    by_index = cube.get_xy_value_from_ij(2, 2)
    by_ixline = cube.get_xy_value_from_ij(2, 2, ixline=True)

    assert by_index == pytest.approx(by_ixline)


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
# resample nearest / default outside_value (src/xtgeo/cube/_cube_utils.py)
# =========================================================================


def test_resample_nearest_default_outside_value() -> None:
    """Test resample with nearest sampling and the default outside_value.

    What is tested:
        A smaller cube that overlaps a larger cube is resampled with the default
        (nearest) sampling and ``outside_value=None``.

    Expected behaviour:
        The target cube keeps its own dimensions after resampling in place.
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


# =========================================================================
# _cube_export.py via cube1.to_file
# =========================================================================


def test_to_file_rms_regular(tmp_path: Path) -> None:
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


def test_to_file_invalid_format_raises(tmp_path: Path) -> None:
    """Test that an unsupported export format is rejected.

    What is tested:
        ``to_file`` is called with an unknown ``fformat``.

    Expected behaviour:
        An InvalidFileFormatError is raised.
    """
    cube = Cube(ncol=3, nrow=2, nlay=5, xinc=10, yinc=10, zinc=1)

    with pytest.raises(InvalidFileFormatError):
        cube.to_file(tmp_path / "cube.bogus", fformat="nonsense")


def test_to_file_engine_argument_warns(tmp_path: Path) -> None:
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


# =========================================================================
# import error paths (src/xtgeo/cube/_cube_import.py)
# =========================================================================


def test_import_segy_garbage_raises_oserror(tmp_path: Path) -> None:
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


def test_import_xtgregcube_invalid_magic_raises(tmp_path: Path) -> None:
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
