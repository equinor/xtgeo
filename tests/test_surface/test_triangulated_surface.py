"""Test xtgeo.surface.triangulated_surface."""

from __future__ import annotations

import logging
import warnings
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pytest

import xtgeo
from xtgeo import TriangulatedSurface
from xtgeo.common.exceptions import InvalidFileFormatError
from xtgeo.io._file import FileFormat
from xtgeo.metadata.metadata import MetaDataTriangulatedSurface
from xtgeo.surface.triangulated_surface import (
    triangulated_surface_from_file,
    triangulated_surface_from_rms,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _square_vertices() -> npt.NDArray[np.float64]:
    """Four vertices forming a unit square in the z=0 plane."""
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )


def _square_triangles() -> npt.NDArray[np.int64]:
    """Two 0-based triangles covering the unit square."""
    return np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)


def _create_surface(**kwargs: Any) -> TriangulatedSurface:
    """Convenience helper to create a surface with default square geometry."""
    defaults = {"vertices": _square_vertices(), "triangles": _square_triangles()}
    defaults.update(kwargs)
    return TriangulatedSurface(**defaults)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tsurf_path(testdata_path: str) -> Path:
    """Return the path to a TSurf test file."""
    p = Path(testdata_path) / "surfaces/drogon/3/F5.ts"
    if not p.exists():
        pytest.skip(f"Test data file not found: {p}")
    return p


@pytest.fixture()
def tsurf_file(tsurf_path: Path) -> str:
    return str(tsurf_path)


@pytest.fixture()
def tsurf_bytes_io(tsurf_path: Path) -> BytesIO:
    return BytesIO(tsurf_path.read_bytes())


@pytest.fixture()
def tsurf_string_io(tsurf_path: Path) -> StringIO:
    return StringIO(tsurf_path.read_text())


# ===================================================================
# 1. Constructor (__init__)
# ===================================================================


class TestConstructor:
    """Tests for TriangulatedSurface.__init__."""

    def test_top_level_access(self) -> None:
        surf = xtgeo.TriangulatedSurface(
            vertices=_square_vertices(),
            triangles=_square_triangles(),
        )
        assert isinstance(surf, TriangulatedSurface)

    def test_minimal_construction(self) -> None:
        surf = _create_surface()
        assert surf.num_vertices == 4
        assert surf.num_triangles == 2
        assert surf.name == "unknown"
        assert surf.filesrc is None
        assert surf.fformat == "unknown"

    def test_construction_with_all_params(self) -> None:
        surf = TriangulatedSurface(
            vertices=_square_vertices(),
            triangles=_square_triangles(),
            filesrc="test.ts",
            fformat="tsurf",
            name="my_surface",
        )
        assert surf.name == "my_surface"
        assert surf.filesrc == "test.ts"
        assert surf.fformat == "tsurf"

    def test_construction_with_none_name(self) -> None:
        surf = TriangulatedSurface(
            vertices=_square_vertices(),
            triangles=_square_triangles(),
            name=None,
        )
        assert surf.name is None

    def test_rejects_non_string_filesrc(self) -> None:
        with pytest.raises(ValueError, match="filesrc must be a string or None"):
            TriangulatedSurface(
                vertices=_square_vertices(),
                triangles=_square_triangles(),
                filesrc=123,
            )

    def test_rejects_non_fileformat_fformat(self) -> None:
        with pytest.raises(ValueError, match="file format must be a string or None"):
            TriangulatedSurface(
                vertices=_square_vertices(),
                triangles=_square_triangles(),
                fformat=123,
            )

    def test_rejects_non_string_name(self) -> None:
        with pytest.raises(ValueError, match="name must be a string or None"):
            TriangulatedSurface(
                vertices=_square_vertices(),
                triangles=_square_triangles(),
                name=42,
            )

    def test_validates_triangulation_data(self) -> None:
        """Constructor delegates to validate_triangulation_data."""
        bad_vertices = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        with pytest.raises(ValueError, match="At least three vertices"):
            TriangulatedSurface(vertices=bad_vertices, triangles=_square_triangles())

    def test_metadata_initialised(self) -> None:
        surf = _create_surface()
        assert isinstance(surf.metadata, MetaDataTriangulatedSurface)


# ===================================================================
# 2. Properties
# ===================================================================


class TestProperties:
    """Tests for simple properties and their setters."""

    def test_modify_vertices(self) -> None:
        surf = _create_surface()
        v = surf.vertices
        orig_val = surf.vertices[0, 0]
        # Verify in-place edits raises error
        with pytest.raises(ValueError):
            v[0, 0] = 999.0

        assert surf.vertices[0, 0] == orig_val
        v_new = np.array(
            [[55.0, 0.0, 0.0], [55.0, 1.0, 0.0], [55.0, 2.0, 0.0], [55.0, 3.0, 0.0]],
            dtype=np.float64,
        )
        surf.vertices = v_new
        assert surf.metadata._required["num_vertices"] == v_new.shape[0]

    def test_modify_triangles(self) -> None:
        surf = _create_surface()
        t = surf.triangles
        orig_val = surf.triangles[0, 0]
        # Verify in-place edits raises error
        with pytest.raises(ValueError):
            t[0, 0] = 999

        assert surf.triangles[0, 0] == orig_val

        t_new = np.array([[0, 1, 2], [0, 2, 3], [0, 1, 2]], dtype=np.int64)
        surf.triangles = t_new
        assert surf.metadata._required["num_triangles"] == t_new.shape[0]

    def test_vertices_content(self) -> None:
        verts = _square_vertices()
        surf = TriangulatedSurface(vertices=verts, triangles=_square_triangles())
        np.testing.assert_array_equal(surf.vertices, verts)

    def test_triangles_content(self) -> None:
        tris = _square_triangles()
        surf = TriangulatedSurface(vertices=_square_vertices(), triangles=tris)
        np.testing.assert_array_equal(surf.triangles, tris)

    def test_num_vertices(self) -> None:
        assert _create_surface().num_vertices == 4

    def test_num_triangles(self) -> None:
        assert _create_surface().num_triangles == 2

    def test_name_setter(self) -> None:
        surf = _create_surface()
        surf.name = "new_name"
        assert surf.name == "new_name"

    def test_name_setter_rejects_blank(self) -> None:
        surf = _create_surface(name="original")
        with pytest.raises(ValueError, match="non-empty string"):
            surf.name = "   "
        assert surf.name == "original"

    def test_filesrc_setter(self) -> None:
        surf = _create_surface()
        surf.filesrc = "new_source.ts"
        assert surf.filesrc == "new_source.ts"

    def test_fformat_setter(self) -> None:
        surf = _create_surface()
        surf.fformat = FileFormat.TSURF.value[1]
        assert surf.fformat == FileFormat.TSURF.value[1]


# ===================================================================
# 3. Metadata
# ===================================================================


class TestMetadata:
    """Tests for the metadata property and setter."""

    def test_metadata_has_required_fields(self) -> None:
        surf = _create_surface()
        req = surf.metadata.required
        assert req["num_vertices"] == 4
        assert req["num_triangles"] == 2

    def test_metadata_setter_rejects_wrong_type(self) -> None:
        surf = _create_surface()
        with pytest.raises(
            ValueError,
            match="not an instance of MetaDataTriangulatedSurface",
        ):
            surf.metadata = "not_metadata"

    def test_metadata_setter_rejects_mismatched(self) -> None:
        surf = _create_surface()
        meta = MetaDataTriangulatedSurface()
        # Give it wrong dimensions
        meta.update_from({"num_vertices": 999, "num_triangles": 999})
        with pytest.raises(ValueError, match="Metadata does not match"):
            surf.metadata = meta

    def test_required_setter_accepts_valid_dict(self) -> None:
        meta = MetaDataTriangulatedSurface()
        meta.required = {"num_vertices": 10, "num_triangles": 5}
        assert meta.required["num_vertices"] == 10
        assert meta.required["num_triangles"] == 5

    def test_required_setter_rejects_non_dict(self) -> None:
        meta = MetaDataTriangulatedSurface()
        with pytest.raises(TypeError, match="required must be a dict"):
            meta.required = "not_a_dict"

    def test_required_setter_rejects_wrong_keys(self) -> None:
        meta = MetaDataTriangulatedSurface()
        with pytest.raises(ValueError, match="Expected keys"):
            meta.required = {"wrong_key": 1}


# ===================================================================
# 4. Bounding box
# ===================================================================


class TestBoundingBox:
    """Tests for bounding_box / _compute_bounding_box."""

    def test_bounding_box_square(self) -> None:
        surf = _create_surface()
        bbox = surf.bounding_box
        assert bbox.min_x == pytest.approx(0.0)
        assert bbox.max_x == pytest.approx(1.0)
        assert bbox.min_y == pytest.approx(0.0)
        assert bbox.max_y == pytest.approx(1.0)
        assert bbox.min_z == pytest.approx(0.0)
        assert bbox.max_z == pytest.approx(0.0)

    def test_bounding_box_3d(self) -> None:
        verts = np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ],
            dtype=np.float64,
        )
        tris = np.array([[0, 1, 2]], dtype=np.int64)
        surf = TriangulatedSurface(vertices=verts, triangles=tris)
        bbox = surf.bounding_box
        assert bbox.min_x == pytest.approx(1.0)
        assert bbox.max_x == pytest.approx(7.0)
        assert bbox.min_y == pytest.approx(2.0)
        assert bbox.max_y == pytest.approx(8.0)
        assert bbox.min_z == pytest.approx(3.0)
        assert bbox.max_z == pytest.approx(9.0)

    def test_raises_on_nan_vertices(self) -> None:
        surf = _create_surface()
        surf._vertices[:] = np.nan
        with pytest.raises(ValueError, match="Bounding box cannot be computed"):
            _ = surf.bounding_box

    def test_raises_on_degenerate_surface(self) -> None:
        surf = _create_surface()
        surf._vertices[:] = np.array([0.0, 0.0, 0.0])
        with pytest.raises(
            ValueError, match="Bounding box cannot be computed for degenerate surface"
        ):
            _ = surf.bounding_box


# ===================================================================
# 5. Computation methods
# ===================================================================


class TestComputations:
    """Tests for compute_* methods."""

    def test_triangle_areas_square(self) -> None:
        surf = _create_surface()
        areas = surf.compute_triangle_areas()
        assert areas.shape == (2,)
        # Two right triangles, each with area 0.5
        np.testing.assert_allclose(areas, [0.5, 0.5])

    def test_surface_area_square(self) -> None:
        surf = _create_surface()
        assert surf.compute_surface_area() == pytest.approx(1.0)

    def test_triangle_normals_square(self) -> None:
        surf = _create_surface()
        normals = surf.compute_triangle_normals()
        assert normals.shape == (2, 3)
        # All normals should point in the z direction for a flat square
        for normal in normals:
            assert abs(normal[2]) == pytest.approx(1.0)
            assert normal[0] == pytest.approx(0.0)
            assert normal[1] == pytest.approx(0.0)

    def test_triangle_centroids_square(self) -> None:
        surf = _create_surface()
        centroids = surf.compute_triangle_centroids()
        assert centroids.shape == (2, 3)
        # Centroid of triangle [0,1,2]: mean of (0,0,0),(1,0,0),(1,1,0)
        np.testing.assert_allclose(centroids[0], [2.0 / 3.0, 1.0 / 3.0, 0.0])
        # Centroid of triangle [0,2,3]: mean of (0,0,0),(1,1,0),(0,1,0)
        np.testing.assert_allclose(centroids[1], [1.0 / 3.0, 2.0 / 3.0, 0.0])

    def test_centroid_square(self) -> None:
        surf = _create_surface()
        c = surf.compute_centroid()
        np.testing.assert_allclose(c, [0.5, 0.5, 0.0])

    def test_compute_triangle_normals_3d(self) -> None:
        """Verify normals for a non-flat triangle."""
        verts = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 1.0],
            ],
            dtype=np.float64,
        )
        tris = np.array([[0, 1, 2]], dtype=np.int64)
        surf = TriangulatedSurface(vertices=verts, triangles=tris)
        normals = surf.compute_triangle_normals()
        # Cross product of (1,0,0) and (0,1,1) = (0,-1,1), normalised
        expected = np.array([0.0, -1.0, 1.0]) / np.sqrt(2.0)
        np.testing.assert_allclose(normals[0], expected, atol=1e-10)

    def test_compute_triangle_normals_degenerate(self) -> None:
        """Degenerate triangle (collinear vertices) should produce zero normal."""
        verts = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],  # collinear
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
        tris = np.array([[0, 1, 2], [0, 1, 3]], dtype=np.int64)
        surf = TriangulatedSurface(vertices=verts, triangles=tris)
        normals = surf.compute_triangle_normals()
        # First triangle is degenerate — should be zero-length normal
        np.testing.assert_allclose(normals[0], [0.0, 0.0, 0.0], atol=1e-10)
        # Second triangle is well-formed — should be unit normal
        assert np.linalg.norm(normals[1]) == pytest.approx(1.0)


# ===================================================================
# 6. validate_triangulation_data (static method)
# ===================================================================


class TestValidateTriangulationData:
    """Tests for TriangulatedSurface.validate_triangulation_data."""

    def test_accepts_valid_input(self) -> None:
        TriangulatedSurface.validate_triangulation_data(
            _square_vertices(), _square_triangles()
        )

    def test_rejects_non_array_vertices(self) -> None:
        with pytest.raises(ValueError, match="Vertices must be a numpy array"):
            TriangulatedSurface.validate_triangulation_data(
                [[0.0, 0.0, 0.0]], np.array([[0, 1, 2]], dtype=np.int64)
            )

    def test_rejects_non_array_triangles(self) -> None:
        with pytest.raises(ValueError, match="Triangles must be a numpy array"):
            TriangulatedSurface.validate_triangulation_data(
                _square_vertices(), [[0, 1, 2]]
            )

    def test_rejects_integer_vertices(self) -> None:
        with pytest.raises(
            ValueError, match="Vertices array must consist of floating point numbers"
        ):
            TriangulatedSurface.validate_triangulation_data(
                _square_vertices().astype(np.int64), _square_triangles()
            )

    def test_rejects_float_triangles(self) -> None:
        with pytest.raises(
            ValueError, match="Triangles array must consist of integers"
        ):
            TriangulatedSurface.validate_triangulation_data(
                _square_vertices(), _square_triangles().astype(np.float64)
            )

    def test_rejects_1d_triangles(self) -> None:
        with pytest.raises(ValueError, match="Triangles array must be 2-dimensional"):
            TriangulatedSurface.validate_triangulation_data(
                _square_vertices(), np.array([0, 1, 2], dtype=np.int64)
            )

    def test_rejects_wrong_cols_triangles(self) -> None:
        with pytest.raises(ValueError, match="Triangles array must be 2-dimensional"):
            TriangulatedSurface.validate_triangulation_data(
                _square_vertices(), np.array([[0, 1], [2, 3]], dtype=np.int64)
            )

    def test_rejects_1d_vertices(self) -> None:
        with pytest.raises(ValueError, match="Vertices array must be 2-dimensional"):
            TriangulatedSurface.validate_triangulation_data(
                np.array([0.0, 1.0, 2.0], dtype=np.float64), _square_triangles()
            )

    def test_rejects_empty_triangles(self) -> None:
        with pytest.raises(ValueError, match="No triangles found"):
            TriangulatedSurface.validate_triangulation_data(
                _square_vertices(), np.empty((0, 3), dtype=np.int64)
            )

    def test_rejects_too_few_vertices(self) -> None:
        verts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
        with pytest.raises(ValueError, match="At least three vertices"):
            TriangulatedSurface.validate_triangulation_data(
                verts, np.array([[0, 0, 1]], dtype=np.int64)
            )

    def test_rejects_nan_vertices(self) -> None:
        verts = _square_vertices()
        verts[0, 0] = np.nan
        with pytest.raises(ValueError, match="Vertices array must not contain NaN"):
            TriangulatedSurface.validate_triangulation_data(verts, _square_triangles())

    def test_rejects_negative_indices(self) -> None:
        with pytest.raises(ValueError, match="Triangle vertex indices must be >= 0"):
            TriangulatedSurface.validate_triangulation_data(
                _square_vertices(), np.array([[0, -1, 2]], dtype=np.int64)
            )

    def test_rejects_out_of_range_indices(self) -> None:
        with pytest.raises(
            ValueError, match="Triangle vertex indices must be < number of vertices"
        ):
            TriangulatedSurface.validate_triangulation_data(
                _square_vertices(), np.array([[0, 1, 10]], dtype=np.int64)
            )

    def test_rejects_duplicate_indices(self) -> None:
        with pytest.raises(
            ValueError, match="Triangles must not contain duplicate vertex indices"
        ):
            TriangulatedSurface.validate_triangulation_data(
                _square_vertices(), np.array([[0, 0, 1]], dtype=np.int64)
            )


# ===================================================================
# 8. copy
# ===================================================================


class TestCopy:
    """Tests for TriangulatedSurface.copy (deep copy)."""

    def test_deep_copy_vertices(self) -> None:
        surf = _create_surface()
        cp = surf.copy()
        cp._vertices[0, 0] = 999.0
        assert surf.vertices[0, 0] == 0.0

    def test_deep_copy_triangles(self) -> None:
        surf = _create_surface()
        cp = surf.copy()
        cp._triangles[0, 0] = 3
        assert surf.triangles[0, 0] == 0

    def test_deep_copy_preserves_metadata_fields(self) -> None:
        surf = TriangulatedSurface(
            vertices=_square_vertices(),
            triangles=_square_triangles(),
            name="original",
            filesrc="source.ts",
            fformat="tsurf",
        )
        cp = surf.copy()
        assert cp.name == "original"
        assert cp.filesrc == "source.ts"
        assert cp.fformat == "tsurf"

    def test_deep_copy_scalar_independence(self) -> None:
        surf = TriangulatedSurface(
            vertices=_square_vertices(),
            triangles=_square_triangles(),
            name="original",
            filesrc="source.ts",
            fformat="tsurf",
        )
        cp = surf.copy()
        surf.name = "modified"
        surf.filesrc = "modified.ts"
        surf.fformat = "unknown"
        assert cp.name == "original"
        assert cp.filesrc == "source.ts"
        assert cp.fformat == "tsurf"

    def test_deep_copy_metadata_independence(self) -> None:
        surf = _create_surface()
        surf.metadata.opt.shortname = "short"
        surf.metadata.freeform["custom"] = "value"

        cp = surf.copy()
        assert cp.metadata.opt.shortname == "short"
        assert cp.metadata.freeform["custom"] == "value"

        surf.metadata.opt.shortname = "modified"
        assert cp.metadata.opt.shortname == "short"
        cp.metadata.freeform["custom"] = "modified"
        assert surf.metadata.freeform["custom"] == "value"


# ===================================================================
# 9. __eq__
# ===================================================================


class TestEquality:
    """Tests for TriangulatedSurface.__eq__."""

    def test_equal_surfaces(self) -> None:
        a = _create_surface()
        b = _create_surface()
        assert a == b

    def test_self_equality(self) -> None:
        a = _create_surface()
        assert a == a

    def test_not_equal_different_vertices(self) -> None:
        a = _create_surface()
        b = _create_surface()
        b._vertices[0, 0] = 999.0
        assert a != b

    def test_not_equal_different_triangles(self) -> None:
        a = _create_surface()
        b = _create_surface()
        b._triangles[0] = [0, 3, 2]
        assert a != b

    def test_not_equal_different_name(self) -> None:
        a = _create_surface(name="a")
        b = _create_surface(name="b")
        assert a != b

    def test_not_equal_different_filesrc(self) -> None:
        a = _create_surface(filesrc="a.ts")
        b = _create_surface(filesrc="b.ts")
        assert a != b

    def test_not_equal_different_fformat(self) -> None:
        a = _create_surface(fformat="tsurf")
        b = _create_surface(fformat="unknown")
        assert a != b

    def test_not_equal_non_surface(self) -> None:
        assert _create_surface().__eq__("not a surface") is NotImplemented


# ===================================================================
# 10. __repr__ / __str__ / describe
# ===================================================================


class TestStringRepresentations:
    """Tests for __repr__, __str__, describe."""

    def test_repr_contains_key_info(self) -> None:
        surf = _create_surface()
        r = repr(surf)
        assert "TriangulatedSurface" in r
        assert "num_vertices" in r
        assert "num_triangles" in r

    def test_repr_is_well_formed(self) -> None:
        surf = _create_surface(name="test", filesrc="test.ts")
        r = repr(surf)
        assert r.startswith("TriangulatedSurface")
        assert "TriangulatedSurfaceTriangulatedSurface" not in r

    def test_str_returns_description(self) -> None:
        surf = _create_surface()
        s = str(surf)
        assert "Description" in s or "TriangulatedSurface" in s

    def test_describe_flush_false(self) -> None:
        surf = _create_surface()
        desc = surf.describe(flush=False)
        assert isinstance(desc, str)
        assert "Number of vertices" in desc
        assert "Number of triangles" in desc

    def test_describe_flush_true(self, capsys: pytest.CaptureFixture[str]) -> None:
        surf = _create_surface()
        surf.describe(flush=True)
        captured = capsys.readouterr()
        assert "Number of vertices" in captured.out


# ===================================================================
# 11. generate_hash
# ===================================================================


class TestGenerateHash:
    """Tests for TriangulatedSurface.generate_hash."""

    def test_hash_is_string(self) -> None:
        assert isinstance(_create_surface().generate_hash(), str)

    def test_same_data_same_hash(self) -> None:
        a = _create_surface()
        b = _create_surface()
        assert a.generate_hash() == b.generate_hash()

    def test_different_data_different_hash(self) -> None:
        a = _create_surface()
        b = _create_surface()
        b._vertices[0, 0] = 999.0
        assert a.generate_hash() != b.generate_hash()

    @pytest.mark.parametrize("method", ["md5", "sha256", "blake2b"])
    def test_hash_methods(self, method: Literal["md5", "sha256", "blake2b"]) -> None:
        h = _create_surface().generate_hash(hashmethod=method)
        assert isinstance(h, str) and len(h) > 0


# ===================================================================
# 12. to_dict
# ===================================================================


class TestToDict:
    """Tests for TriangulatedSurface.to_dict."""

    def test_keys_present(self) -> None:
        d = _create_surface().to_dict()
        assert "vertices" in d
        assert "triangles" in d
        assert "filesrc" in d
        assert "name" in d

    def test_defaults(self) -> None:
        d = _create_surface().to_dict()
        assert d["filesrc"] == "unknown"
        assert d["name"] == "unknown"

    def test_custom_values(self) -> None:
        surf = TriangulatedSurface(
            vertices=_square_vertices(),
            triangles=_square_triangles(),
            filesrc="test.ts",
            name="my_surf",
        )
        d = surf.to_dict()
        assert d["filesrc"] == "test.ts"
        assert d["name"] == "my_surf"

    def test_includes_tsurf_coord_sys_freeform(self) -> None:
        surf = _create_surface()
        coord_sys = {
            "name": "Default",
            "axis_name": ["X", "Y", "Z"],
            "axis_unit": ["m", "m", "m"],
            "zpositive": "Depth",
        }
        surf.metadata.freeform["tsurf_coord_sys"] = coord_sys
        d = surf.to_dict()
        assert "free_form_metadata" in d
        assert d["free_form_metadata"]["tsurf_coord_sys"] is coord_sys


# ===================================================================
# 12b. from_dict
# ===================================================================


class TestFromDict:
    """Tests for TriangulatedSurface.from_dict."""

    def test_round_trip_to_dict_from_dict(self) -> None:
        """to_dict -> from_dict produces an equal surface."""
        original = _create_surface(filesrc="src.ts", name="surf1")
        restored = TriangulatedSurface.from_dict(original.to_dict())
        assert np.array_equal(restored.vertices, original.vertices)
        assert np.array_equal(restored.triangles, original.triangles)
        assert restored.name == original.name
        assert restored.filesrc == original.filesrc

    def test_minimal_keys(self) -> None:
        """Only vertices and triangles are required."""
        data = {
            "vertices": _square_vertices(),
            "triangles": _square_triangles(),
        }
        surf = TriangulatedSurface.from_dict(data)
        assert surf.num_vertices == 4
        assert surf.num_triangles == 2
        assert surf.name == "unknown"

    def test_fformat_parameter(self) -> None:
        """fformat is passed through as a separate parameter, not in the dict."""
        data = {
            "vertices": _square_vertices(),
            "triangles": _square_triangles(),
        }
        surf = TriangulatedSurface.from_dict(data, fformat="tsurf")
        assert surf.fformat == "tsurf"

    def test_overrides_take_precedence(self) -> None:
        """Keyword overrides win over values in the dict."""
        data = {
            "vertices": _square_vertices(),
            "triangles": _square_triangles(),
            "name": "from_dict_name",
            "filesrc": "from_dict_src",
        }
        surf = TriangulatedSurface.from_dict(
            data, filesrc="override_src", name="override_name"
        )
        assert surf.filesrc == "override_src"
        assert surf.name == "override_name"

    def test_free_form_metadata_populates_freeform(self) -> None:
        """free_form_metadata in the dict is stored in metadata.freeform."""
        coord_sys = {
            "name": "Default",
            "axis_name": ["X", "Y", "Z"],
            "axis_unit": ["m", "m", "m"],
            "zpositive": "Depth",
        }
        data = {
            "vertices": _square_vertices(),
            "triangles": _square_triangles(),
            "free_form_metadata": {"tsurf_coord_sys": coord_sys},
        }
        surf = TriangulatedSurface.from_dict(data)
        assert "tsurf_coord_sys" in surf.metadata.freeform
        assert surf.metadata.freeform["tsurf_coord_sys"] == coord_sys

    def test_no_free_form_metadata_key(self) -> None:
        """When free_form_metadata is absent, metadata.freeform stays empty."""
        data = {
            "vertices": _square_vertices(),
            "triangles": _square_triangles(),
        }
        surf = TriangulatedSurface.from_dict(data)
        assert "tsurf_coord_sys" not in surf.metadata.freeform


# ===================================================================
# 13. File I/O — reading (integration tests requiring testdata)
# ===================================================================


class TestReadFile:
    """Tests for _read_file / triangulated_surface_from_file."""

    @pytest.mark.parametrize("fformat", ["tsurf", None])
    @pytest.mark.parametrize("input_type", ["str", "path", "bytes_io", "string_io"])
    def test_from_file_multiple_inputs(
        self,
        tsurf_file: str,
        tsurf_path: str,
        tsurf_bytes_io: bytes,
        tsurf_string_io: str,
        input_type: str,
        fformat: Literal["tsurf", None],
    ) -> None:
        source = {
            "str": tsurf_file,
            "path": tsurf_path,
            "bytes_io": tsurf_bytes_io,
            "string_io": tsurf_string_io,
        }[input_type]
        surface = triangulated_surface_from_file(source, fformat=fformat)
        assert isinstance(surface, TriangulatedSurface)
        assert surface.num_vertices > 0
        assert surface.num_triangles > 0

    def test_raises_for_missing_file(self, tsurf_path: Path) -> None:
        missing = tsurf_path.parent / "nonexistent.ts"
        with pytest.raises(ValueError, match="does not exist or cannot be accessed"):
            triangulated_surface_from_file(missing)

    def test_unknown_extension_known_format(self, tsurf_path: Path) -> None:
        # Test that if we provide a known format explicitly, it works even
        # if the extension is wrong.
        import shutil

        unknown_extension = tsurf_path.with_suffix(".unknown")
        try:
            shutil.copy2(tsurf_path, unknown_extension)
            surface = triangulated_surface_from_file(unknown_extension, fformat="tsurf")
            assert isinstance(surface, TriangulatedSurface)
            assert surface.num_vertices > 0
            assert surface.num_triangles > 0
        finally:
            unknown_extension.unlink(missing_ok=True)

    def test_raises_for_invalid_format(self, tsurf_path: Path) -> None:
        # If format is explicitly given but invalid, should raise error
        # even if file exists
        with pytest.raises(InvalidFileFormatError):
            TriangulatedSurface._read_file(tsurf_path, fformat="invalid_format")

    def test_raises_for_unsupported_format(self, tsurf_path: Path) -> None:
        # A format known to xtgeo but not supported by TriangulatedSurface
        with pytest.raises(InvalidFileFormatError, match="invalid for type"):
            TriangulatedSurface._read_file(tsurf_path, fformat="irap_binary")

    def test_basic_geometry(self, tsurf_path: Path) -> None:
        surface = triangulated_surface_from_file(tsurf_path)
        assert surface.num_vertices == 44
        assert surface.num_triangles == 65
        areas = surface.compute_triangle_areas()
        assert areas.shape == (65,)
        assert np.all(areas > 0.0)
        assert surface.compute_surface_area() == pytest.approx(1292640.272423, rel=1e-4)

    def test_bounding_box_from_file(self, tsurf_path: Path) -> None:
        surface = triangulated_surface_from_file(tsurf_path)
        bbox = surface.bounding_box
        assert bbox.min_x == pytest.approx(459252.895264)
        assert bbox.max_x == pytest.approx(460106.177490)
        assert bbox.min_y == pytest.approx(5934226.964478)
        assert bbox.max_y == pytest.approx(5936393.925537)
        assert bbox.min_z == pytest.approx(1437.976074)
        assert bbox.max_z == pytest.approx(1839.967407)


# ===================================================================
# 14. File I/O — reading coord sys
# ===================================================================


class TestFromTSurfCoordSys:
    """Tests for coordinate system handling during TSurf import."""

    def test_coord_sys_stored_in_freeform_metadata(self, tsurf_path: Path) -> None:
        surface = triangulated_surface_from_file(tsurf_path, fformat="tsurf")
        coord_sys = surface.metadata.freeform.get("tsurf_coord_sys")
        assert coord_sys is not None
        assert coord_sys["name"] == "Default"
        assert coord_sys["axis_name"] == ("X", "Y", "Z")
        assert coord_sys["axis_unit"] == ("m", "m", "m")
        assert coord_sys["zpositive"] == "Depth"

    def test_elevation_zpositive_inverts_z(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When ZPOSITIVE is Elevation, z-values must be negated."""
        # Build a fake tsurf import result with Elevation coord sys
        fake_result = {
            "vertices": np.array(
                [[0.0, 0.0, 10.0], [1.0, 0.0, 20.0], [0.0, 1.0, 30.0]],
                dtype=np.float64,
            ),
            "triangles": np.array([[0, 1, 2]], dtype=np.int64),  # 1-based
            "name": "elevation_surface",
            "filesrc": "dummy.ts",
            "fformat": FileFormat.TSURF,
            "free_form_metadata": {
                "tsurf_coord_sys": {
                    "name": "Default",
                    "axis_name": ["X", "Y", "Z"],
                    "axis_unit": ["m", "m", "m"],
                    "zpositive": "Elevation",
                }
            },
        }

        monkeypatch.setattr(
            "xtgeo.surface.triangulated_surface.import_tsurf",
            lambda _mfile: fake_result,
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            surface = triangulated_surface_from_file(StringIO("dummy"), fformat="tsurf")
            # Should have emitted a warning about z inversion
            assert any("ZPOSITIVE" in str(warning.message) for warning in w)

        # z-values should be negated
        np.testing.assert_allclose(surface.vertices[:, 2], [-10.0, -20.0, -30.0])
        # Metadata should be updated to Depth
        assert surface.metadata.freeform["tsurf_coord_sys"]["zpositive"] == "Depth"

    def test_depth_zpositive_no_inversion(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When ZPOSITIVE is Depth, z-values remain unchanged."""
        original_z = np.array([10.0, 20.0, 30.0])
        fake_result = {
            "vertices": np.array(
                [[0.0, 0.0, 10.0], [1.0, 0.0, 20.0], [0.0, 1.0, 30.0]],
                dtype=np.float64,
            ),
            "triangles": np.array([[0, 1, 2]], dtype=np.int64),
            "name": "depth_surface",
            "filesrc": "dummy.ts",
            "fformat": FileFormat.TSURF,
            "free_form_metadata": {
                "tsurf_coord_sys": {
                    "name": "Default",
                    "axis_name": ["X", "Y", "Z"],
                    "axis_unit": ["m", "m", "m"],
                    "zpositive": "Depth",
                }
            },
        }
        monkeypatch.setattr(
            "xtgeo.surface.triangulated_surface.import_tsurf",
            lambda _mfile: fake_result,
        )
        surface = triangulated_surface_from_file(StringIO("dummy"), fformat="tsurf")
        np.testing.assert_allclose(surface.vertices[:, 2], original_z)

    def test_metric_units_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When axis_unit is 'm', metadata.optional['units'] should be 'metric'."""
        fake_result = {
            "vertices": np.array(
                [[0.0, 0.0, 1.0], [1.0, 0.0, 2.0], [0.0, 1.0, 3.0]],
                dtype=np.float64,
            ),
            "triangles": np.array([[0, 1, 2]], dtype=np.int64),
            "name": "metric_surface",
            "filesrc": "dummy.ts",
            "fformat": FileFormat.TSURF,
            "free_form_metadata": {
                "tsurf_coord_sys": {
                    "name": "Default",
                    "axis_name": ["X", "Y", "Z"],
                    "axis_unit": ["m", "m", "m"],
                    "zpositive": "Depth",
                }
            },
        }
        monkeypatch.setattr(
            "xtgeo.surface.triangulated_surface.import_tsurf",
            lambda _mfile: fake_result,
        )
        surface = triangulated_surface_from_file(StringIO("dummy"), fformat="tsurf")
        assert surface.metadata.optional.get("units") == "metric"


# ===================================================================
# 15. File I/O — writing
# ===================================================================


class TestWriteFile:
    """Tests for TriangulatedSurface.to_file."""

    def test_round_trip_tsurf(self, tmp_path: Path) -> None:
        verts = _square_vertices()
        tris = _square_triangles()
        surf = TriangulatedSurface(
            vertices=verts,
            triangles=tris,
            fformat=FileFormat.TSURF.value[1],
            name="round_trip",
        )
        filepath = tmp_path / "test.ts"
        result = surf.to_file(filepath)
        assert result == filepath

        loaded = triangulated_surface_from_file(filepath, fformat="tsurf")
        np.testing.assert_array_almost_equal(loaded.vertices, verts)
        assert loaded.num_vertices == surf.num_vertices
        assert loaded.num_triangles == surf.num_triangles

    def test_to_file_invalid_format(self, tmp_path: Path) -> None:
        surf = _create_surface()
        filepath = tmp_path / "test.xyz"
        with pytest.raises(InvalidFileFormatError):
            surf.to_file(filepath, fformat="irap_binary")

    def test_to_file_bytes_io(self) -> None:
        surf = _create_surface(fformat=FileFormat.TSURF.value[0], name="bytes_io_test")
        buf = BytesIO()
        result = surf.to_file(buf, fformat=FileFormat.TSURF.value[1])
        # For memstreams, returns None
        assert result is None
        buf.seek(0)
        content = buf.read()
        assert len(content) > 0
        assert b"GOCAD TSurf" in content

    def test_round_trip_preserves_coord_sys(self, tmp_path: Path) -> None:
        """Round-trip should preserve coord sys metadata if present."""
        surf = _create_surface(fformat=FileFormat.TSURF.value[0], name="coord_test")
        surf.metadata.freeform["tsurf_coord_sys"] = {
            "name": "Default",
            "axis_name": ["X", "Y", "Z"],
            "axis_unit": ["m", "m", "m"],
            "zpositive": "Depth",
        }
        filepath = tmp_path / "coord_test.ts"
        surf.to_file(filepath)

        loaded = triangulated_surface_from_file(filepath, fformat="tsurf")
        cs = loaded.metadata.freeform.get("tsurf_coord_sys")
        assert cs is not None
        assert cs["name"] == "Default"
        assert cs["zpositive"] == "Depth"


# ===================================================================
# 16. Not-implemented methods
# ===================================================================


class TestNotImplemented:
    """Tests for methods that raise NotImplementedError."""

    def test_from_rms_not_implemented(self) -> None:
        with pytest.raises(
            NotImplementedError,
            match="Import from RMS not implemented",
        ):
            triangulated_surface_from_rms()

    def test_to_rms_not_implemented(self) -> None:
        surf = _create_surface()
        with pytest.raises(NotImplementedError, match="Export to RMS not implemented"):
            surf.to_rms()

    def test_to_hdf_not_implemented(self) -> None:
        surf = _create_surface()
        with pytest.raises(NotImplementedError, match="Export to HDF not implemented"):
            surf.to_hdf()
