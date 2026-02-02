from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from xtgeo.io.triangulated_surface.triangulated_surface import (
    GxfInfoKeyType,
    TriangulatedSurface,
    TriangulatedSurfaceFileFormat,
    TSurfInfoKeyType,
    TSurfInfoValueType,
)
from xtgeo.io.tsurf._tsurf_io import TSurfCoordSys, TSurfHeader


def _square_vertices() -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )


def _square_triangles() -> np.ndarray:
    """Return 0-based triangles for a square surface made of two triangles."""
    return np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)


def _require_testdata_file(path: Path) -> None:
    if not path.exists():
        pytest.skip("Required testdata file not available")


def test_basic_geometry():
    vertices = _square_vertices()
    triangles = _square_triangles()

    surface = TriangulatedSurface(vertices, triangles, {})

    assert surface.num_vertices == 4
    assert surface.num_triangles == 2

    bbox = surface.bounding_box
    assert bbox.min_x == 0.0
    assert bbox.max_x == 1.0
    assert bbox.min_y == 0.0
    assert bbox.max_y == 1.0
    assert bbox.min_z == 0.0
    assert bbox.max_z == 0.0

    areas = TriangulatedSurface._compute_triangle_areas(vertices, triangles)
    assert areas.shape == (2,)
    assert areas[0] == pytest.approx(0.5)
    assert areas[1] == pytest.approx(0.5)
    assert float(np.sum(areas)) == pytest.approx(1.0)


def test_validate_accepts_valid_input():
    vertices = _square_vertices()
    triangles = _square_triangles()

    TriangulatedSurface.validate_triangulation_data(vertices, triangles)


def test_validate_rejects_non_arrays():
    vertices = [[0.0, 0.0, 0.0]]
    triangles = [[0, 1, 2]]

    with pytest.raises(ValueError, match="Vertices must be a numpy array"):
        TriangulatedSurface.validate_triangulation_data(vertices, np.array(triangles))

    with pytest.raises(ValueError, match="Triangles must be a numpy array"):
        TriangulatedSurface.validate_triangulation_data(np.array(vertices), triangles)


def test_validate_rejects_wrong_dtypes():
    vertices = _square_vertices().astype(np.int64)
    triangles = _square_triangles().astype(np.float64)

    with pytest.raises(
        ValueError, match="Vertices array must consist of floating point numbers"
    ):
        TriangulatedSurface.validate_triangulation_data(vertices, _square_triangles())

    with pytest.raises(ValueError, match="Triangles array must consist of integers"):
        TriangulatedSurface.validate_triangulation_data(_square_vertices(), triangles)


def test_validate_rejects_empty_triangles():
    vertices = _square_vertices()
    triangles = np.empty((0, 3), dtype=np.int64)

    with pytest.raises(ValueError, match="No triangles found"):
        TriangulatedSurface.validate_triangulation_data(vertices, triangles)


def test_validate_rejects_too_few_vertices():
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    dummy_triangles = np.array([[0, -1, -1]], dtype=np.int64)

    with pytest.raises(ValueError, match="At least three vertices"):
        TriangulatedSurface.validate_triangulation_data(vertices, dummy_triangles)


def test_validate_rejects_negative_indices():
    vertices = _square_vertices()
    triangles = np.array([[0, -1, 2]], dtype=np.int64)

    with pytest.raises(ValueError, match="Triangle vertex indices must be >= 0"):
        TriangulatedSurface.validate_triangulation_data(vertices, triangles)


def test_validate_rejects_out_of_range_indices():
    vertices = _square_vertices()
    triangles = np.array([[0, 1, 10]], dtype=np.int64)

    with pytest.raises(
        ValueError, match="Triangle vertex indices must be < number of vertices"
    ):
        TriangulatedSurface.validate_triangulation_data(vertices, triangles)


def test_validate_rejects_bad_shapes():
    vertices = _square_vertices()

    triangles_bad_shape = np.array([0, 1, 2], dtype=np.int64)
    with pytest.raises(ValueError, match="Triangles array must be 2-dimensional"):
        TriangulatedSurface.validate_triangulation_data(vertices, triangles_bad_shape)

    vertices_bad_shape = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    with pytest.raises(ValueError, match="Vertices array must be 2-dimensional"):
        TriangulatedSurface.validate_triangulation_data(
            vertices_bad_shape, _square_triangles()
        )

    triangles_bad_last_dim = np.array([[0, 1], [1, 2]], dtype=np.int64)
    with pytest.raises(ValueError, match="Triangles array must be 2-dimensional"):
        TriangulatedSurface.validate_triangulation_data(
            vertices, triangles_bad_last_dim
        )


def test_validate_rejects_duplicate_indices():
    vertices = _square_vertices()
    triangles = np.array([[0, 0, 1]], dtype=np.int64)

    with pytest.raises(
        ValueError, match="Triangles must not contain duplicate vertex indices"
    ):
        TriangulatedSurface.validate_triangulation_data(vertices, triangles)


def test_validate_rejects_degenerate_triangles():
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    triangles = np.array([[0, 1, 2]], dtype=np.int64)

    with pytest.raises(ValueError, match="Triangles must not be degenerate"):
        TriangulatedSurface.validate_triangulation_data(vertices, triangles)


def test_validate_rejects_nan_vertices():
    vertices = _square_vertices()
    triangles = _square_triangles()
    vertices[0, 0] = np.nan

    with pytest.raises(ValueError, match="Vertices array must not contain NaN"):
        TriangulatedSurface.validate_triangulation_data(vertices, triangles)


def test_validate_rejects_nan_triangles():
    vertices = _square_vertices()
    triangles = _square_triangles().astype(np.float64)
    triangles[0, 0] = np.nan

    with pytest.raises(ValueError, match="Triangles array must consist of integers"):
        TriangulatedSurface.validate_triangulation_data(vertices, triangles)


def test_from_file_gxf_not_implemented():
    with pytest.raises(NotImplementedError, match="GXF format is not yet supported"):
        TriangulatedSurface.from_file("dummy.gxf", TriangulatedSurfaceFileFormat.GXF)


def test_tsurf_info_returns_dict():
    vertices = _square_vertices()
    triangles = _square_triangles()
    info: dict[TSurfInfoKeyType, TSurfInfoValueType] = {
        TSurfInfoKeyType.HEADER: TSurfHeader(name="test_surface")
    }

    surface = TriangulatedSurface(vertices, triangles, info)

    assert surface.info is info
    assert surface.tsurf_info is info
    assert surface.gxf_info == {}


def test_tsurf_info_contains_expected_keys():
    vertices = _square_vertices()
    triangles = _square_triangles()
    header = TSurfHeader(name="test_surface")
    coordsys = TSurfCoordSys(
        name="Default",
        axis_name=["X", "Y", "Z"],
        axis_unit=["m", "m", "m"],
        zpositive="Depth",
    )
    info = {
        TSurfInfoKeyType.HEADER: header,
        TSurfInfoKeyType.COORDSYS: coordsys,
    }

    surface = TriangulatedSurface(vertices, triangles, info)

    assert surface.tsurf_info[TSurfInfoKeyType.HEADER] is header
    assert surface.tsurf_info[TSurfInfoKeyType.COORDSYS] is coordsys


def test_from_tsurf_file_reads_coord_sys(monkeypatch: pytest.MonkeyPatch):
    vertices = _square_vertices()
    triangles = _square_triangles()
    triangles += 1  # Simulate 1-based indexing in TSurf files
    coord_sys = TSurfCoordSys(
        name="Default",
        axis_name=["X", "Y", "Z"],
        axis_unit=["m", "m", "m"],
        zpositive="Depth",
    )
    header = TSurfHeader(name="test_surface")

    fake_tsurf_data = SimpleNamespace(
        header=header,
        vertices=vertices,
        triangles=triangles,
        coord_sys=coord_sys,
    )

    def _fake_from_file(file: str) -> SimpleNamespace:
        return fake_tsurf_data

    monkeypatch.setattr(
        "xtgeo.io.triangulated_surface.triangulated_surface.TSurfData.from_file",
        _fake_from_file,
    )

    surface = TriangulatedSurface._from_tsurf_file("dummy.tsurf")

    assert surface.tsurf_info[TSurfInfoKeyType.COORDSYS] is coord_sys


def test_from_tsurf_file_reads_and_validates_coord_sys(testdata_path: Path):
    file_path = Path(testdata_path) / "surfaces/drogon/3/F5.ts"
    _require_testdata_file(file_path)

    surface = TriangulatedSurface._from_tsurf_file(file_path)

    coord_sys = surface.tsurf_info[TSurfInfoKeyType.COORDSYS]
    assert coord_sys is not None
    assert coord_sys.name == "Default"
    assert coord_sys.axis_name == ["X", "Y", "Z"]
    assert coord_sys.axis_unit == ["m", "m", "m"]
    assert coord_sys.zpositive == "Depth"

    assert isinstance(coord_sys, TSurfCoordSys)


def test_raises_if_bounding_box_cannot_be_computed():
    vertices = _square_vertices()
    triangles = _square_triangles()

    surface = TriangulatedSurface(vertices, triangles)

    surface._vertices[:] = np.nan

    with pytest.raises(ValueError, match="Bounding box cannot be computed"):
        _ = surface.bounding_box


def test_raises_if_bounding_box_computed_for_degenerate_surface():
    vertices = _square_vertices()
    triangles = _square_triangles()

    surface = TriangulatedSurface(vertices, triangles, {})

    # Set vertices to the same point to simulate degenerate surface
    surface._vertices[:] = np.array([0.0, 0.0, 0.0])

    with pytest.raises(
        ValueError, match="Bounding box cannot be computed for degenerate surface"
    ):
        _ = surface.bounding_box


def test_raises_if_info_is_not_dict():

    vertices = _square_vertices()
    triangles = _square_triangles()

    with pytest.raises(ValueError, match="Info must be a dictionary."):
        TriangulatedSurface(vertices, triangles, "not a dict")


def test_require_testdata_file_skips_if_missing(tmp_path: Path):
    missing_path = tmp_path / "missing.tsurf"

    with pytest.raises(pytest.skip.Exception):
        _require_testdata_file(missing_path)


def test_tsurf_info_only_accepts_valid_keys():
    vertices = _square_vertices()
    triangles = _square_triangles()
    aux = {
        TSurfInfoKeyType.HEADER: TSurfHeader(name="test_surface"),
        TSurfInfoKeyType.COORDSYS: TSurfCoordSys(
            name="Default",
            axis_name=["X", "Y", "Z"],
            axis_unit=["m", "m", "m"],
            zpositive="Depth",
        ),
        "invalid_key": "some value",
    }

    with pytest.raises(ValueError, match="Invalid info key:"):
        TriangulatedSurface(vertices, triangles, aux)


def test_tsurf_info_only_accepts_valid_types():
    vertices = _square_vertices()
    triangles = _square_triangles()
    aux = {
        TSurfInfoKeyType.HEADER: "not a TSurfHeader",
        TSurfInfoKeyType.COORDSYS: "not a TSurfCoordSys",
    }

    with pytest.raises(ValueError, match="Invalid tsurf info value for key"):
        TriangulatedSurface(vertices, triangles, aux)


def test_copy_is_deep_copy():
    vertices = _square_vertices()
    triangles = _square_triangles()
    aux = {
        TSurfInfoKeyType.HEADER: TSurfHeader(name="original"),
        TSurfInfoKeyType.COORDSYS: TSurfCoordSys(
            name="Default",
            axis_name=["X", "Y", "Z"],
            axis_unit=["m", "m", "m"],
            zpositive="Depth",
        ),
    }

    surface = TriangulatedSurface(vertices, triangles, aux)
    surface_copy = surface.copy()

    surface_copy.tsurf_info[TSurfInfoKeyType.HEADER] = TSurfHeader(name="modified")

    assert surface.tsurf_info[TSurfInfoKeyType.HEADER] == TSurfHeader(name="original")


def test_default_info_is_empty_dict():
    vertices = _square_vertices()
    triangles = _square_triangles()

    surface = TriangulatedSurface(vertices, triangles)

    assert surface.tsurf_info == {}


def test_triangles_property():
    vertices = _square_vertices()
    triangles = _square_triangles()

    surface = TriangulatedSurface(vertices, triangles)

    np.testing.assert_array_equal(surface.triangles, triangles)


def test_gxf_info_property_with_gxf_keys():
    vertices = _square_vertices()
    triangles = _square_triangles()
    gxf_data = {GxfInfoKeyType.DUMMYTYPE_1: object()}

    surface = TriangulatedSurface(vertices, triangles, gxf_data)

    assert surface.gxf_info is gxf_data
    assert GxfInfoKeyType.DUMMYTYPE_1 in surface.gxf_info


def test_tsurf_info_returns_empty_when_gxf_data_present():
    vertices = _square_vertices()
    triangles = _square_triangles()
    gxf_data = {GxfInfoKeyType.DUMMYTYPE_1: object()}

    surface = TriangulatedSurface(vertices, triangles, gxf_data)

    assert surface.tsurf_info == {}


def test_gxf_info_returns_empty_when_tsurf_data_present():
    vertices = _square_vertices()
    triangles = _square_triangles()
    tsurf_data = {TSurfInfoKeyType.HEADER: TSurfHeader(name="test")}

    surface = TriangulatedSurface(vertices, triangles, tsurf_data)

    assert surface.gxf_info == {}


def test_rejects_mixed_tsurf_and_gxf_keys():
    vertices = _square_vertices()
    triangles = _square_triangles()
    mixed = {
        TSurfInfoKeyType.HEADER: TSurfHeader(name="test"),
        GxfInfoKeyType.DUMMYTYPE_1: object(),
    }

    with pytest.raises(ValueError, match="Cannot mix TSurf and GXF info keys"):
        TriangulatedSurface(vertices, triangles, mixed)


def test_to_tsurf_file_round_trip(tmp_path: Path):
    vertices = _square_vertices()
    triangles = _square_triangles()
    header = TSurfHeader(name="round_trip_test")
    coordsys = TSurfCoordSys(
        name="Default",
        axis_name=["X", "Y", "Z"],
        axis_unit=["m", "m", "m"],
        zpositive="Depth",
    )
    info = {
        TSurfInfoKeyType.HEADER: header,
        TSurfInfoKeyType.COORDSYS: coordsys,
    }

    surface = TriangulatedSurface(vertices, triangles, info)
    filepath = tmp_path / "test.ts"
    surface.to_file(filepath, TriangulatedSurfaceFileFormat.TSURF)

    loaded = TriangulatedSurface.from_file(
        filepath, TriangulatedSurfaceFileFormat.TSURF
    )

    np.testing.assert_array_almost_equal(loaded.vertices, vertices)
    assert loaded.num_vertices == surface.num_vertices
    assert loaded.num_triangles == surface.num_triangles
    assert loaded.tsurf_info[TSurfInfoKeyType.HEADER].name == "round_trip_test"


def test_to_gxf_file_not_implemented():
    vertices = _square_vertices()
    triangles = _square_triangles()
    surface = TriangulatedSurface(vertices, triangles)

    with pytest.raises(NotImplementedError, match="GXF format is not yet supported"):
        surface.to_file("dummy.gxf", TriangulatedSurfaceFileFormat.GXF)


def test_to_tsurf_file_raises_if_missing_header(tmp_path: Path):
    vertices = _square_vertices()
    triangles = _square_triangles()
    surface = TriangulatedSurface(vertices, triangles, {})

    with pytest.raises(ValueError, match="Header information is required"):
        surface._to_tsurf_file(tmp_path / "test.ts")


def test_to_tsurf_file_raises_if_wrong_header_type(tmp_path: Path):
    vertices = _square_vertices()
    triangles = _square_triangles()
    # Force a bad header type past __init__ validation by using object.__setattr__
    info = {TSurfInfoKeyType.HEADER: TSurfHeader(name="ok")}
    surface = TriangulatedSurface(vertices, triangles, info)
    # Mutate the frozen field to inject a bad header
    bad_info = {TSurfInfoKeyType.HEADER: "not a TSurfHeader"}
    object.__setattr__(surface, "_info", bad_info)

    with pytest.raises(ValueError, match="not of type TSurfHeader"):
        surface._to_tsurf_file(tmp_path / "test.ts")


def test_copy_preserves_vertex_and_triangle_independence():
    vertices = _square_vertices()
    triangles = _square_triangles()
    info = {TSurfInfoKeyType.HEADER: TSurfHeader(name="original")}

    surface = TriangulatedSurface(vertices, triangles, info)
    surface_copy = surface.copy()

    # Mutate the copy's arrays
    surface_copy._vertices[0, 0] = 999.0
    surface_copy._triangles[0, 0] = 3

    # Originals unchanged
    assert surface.vertices[0, 0] == 0.0
    assert surface.triangles[0, 0] == 0


def test_triangulated_surface_is_frozen():
    vertices = _square_vertices()
    triangles = _square_triangles()
    surface = TriangulatedSurface(vertices, triangles)

    with pytest.raises(AttributeError):
        surface._vertices = np.array([[9.0, 9.0, 9.0]])
    with pytest.raises(AttributeError):
        surface._info = {}
