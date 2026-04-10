import numpy as np
import pytest
from hypothesis import given
from numpy.testing import assert_allclose

import xtgeo

# Reuse the grid generator strategy from the grid3d test suite
from tests.test_grid3d.grid_generator import xtgeo_grids
from xtgeo.interfaces.resinsight._grid import GridDataResInsight, GridReader, GridWriter
from xtgeo.interfaces.resinsight._rips_package import RipsInstanceType


def test_validate_array_size():
    """Test that GridDataResInsight raises ValueError when array sizes do not match
    expected dimensions."""

    with pytest.raises(
        ValueError, match="coordsv should have length 150, but got length 10"
    ):
        GridDataResInsight(
            name="TEST_GRID",
            nx=4,
            ny=4,
            nz=3,
            coordsv=np.zeros(10, dtype=np.float32),  # Incorrect size
            zcornsv=np.zeros(384, dtype=np.float32),  # Correct size for 4x4x3 grid
            actnumsv=np.zeros(48, dtype=np.int32),  # Correct size for 4x4x3 grid
            filesrc="test_grid.roff",
        )
    with pytest.raises(
        ValueError, match="zcornsv should have length 384, but got length 10"
    ):
        GridDataResInsight(
            name="TEST_GRID",
            nx=4,
            ny=4,
            nz=3,
            coordsv=np.zeros(150, dtype=np.float32),  # Correct size for 4x4x3 grid
            zcornsv=np.zeros(10, dtype=np.float32),  # Incorrect size
            actnumsv=np.zeros(48, dtype=np.int32),  # Correct size for 4x4x3 grid
            filesrc="test_grid.roff",
        )
    with pytest.raises(
        ValueError, match="actnumsv should have length 48, but got length 10"
    ):
        GridDataResInsight(
            name="TEST_GRID",
            nx=4,
            ny=4,
            nz=3,
            coordsv=np.zeros(150, dtype=np.float32),  # Correct size for 4x4x3 grid
            zcornsv=np.zeros(384, dtype=np.float32),  # Correct size for 4x4x3 grid
            actnumsv=np.zeros(10, dtype=np.int32),  # Incorrect size
            filesrc="test_grid.roff",
        )


# ---------------------------------------------------------------------------
# Roundtrip tests: GridDataResInsight ↔ XTGeo Grid (no ResInsight required)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dimension, increment, rotation, origin, filesrc",
    [
        ((2, 2, 2), (1.0, 1.0, 1.0), 0.0, (0.0, 0.0, 0.0), ""),
        ((4, 3, 2), (10.0, 10.0, 5.0), 0.0, (0.0, 0.0, 0.0), ""),
        ((10, 8, 5), (100.0, 100.0, 20.0), 0.0, (0.0, 0.0, 0.0), "/path/to/file.roff"),
        ((5, 4, 3), (50.0, 50.0, 10.0), 30.0, (100.0, 200.0, 1000.0), ""),
    ],
    ids=["small", "medium", "large-with-filesrc", "rotated"],
)
def test_roundtrip(
    dimension,
    increment: tuple[float, float, float],
    rotation: float,
    origin: tuple[float, float, float],
    filesrc: str,
):
    """XTGeo Grid → GridDataResInsight → XTGeo Grid roundtrip preserves
    dimensions, geometry (within float32 precision), flat array sizes/dtypes,
    and filesrc."""
    original = xtgeo.create_box_grid(
        dimension, increment=increment, rotation=rotation, origin=origin
    )
    nx, ny, nz = original.ncol, original.nrow, original.nlay
    data = GridDataResInsight.from_xtgeo_grid(original, name="TEST", filesrc=filesrc)

    # Intermediate flat array sizes and dtypes
    assert data.coordsv.size == (nx + 1) * (ny + 1) * 6
    assert data.zcornsv.size == nx * ny * nz * 8
    assert data.actnumsv.size == nx * ny * nz
    assert data.coordsv.dtype == np.float64
    assert data.zcornsv.dtype == np.float32
    assert data.actnumsv.dtype == np.int32

    restored = data.to_xtgeo_grid()

    assert restored.ncol == nx
    assert restored.nrow == ny
    assert restored.nlay == nz
    assert restored.filesrc == filesrc

    original._set_xtgformat2()
    restored._set_xtgformat2()

    # atol=1e-2 covers float64→float32→float64 precision loss, including rotated grids
    assert_allclose(original._coordsv, restored._coordsv, atol=1e-2)
    assert_allclose(original._zcornsv, restored._zcornsv, atol=1e-2)
    assert np.array_equal(original._actnumsv, restored._actnumsv)


def test_roundtrip_inactive_cells():
    """Inactive cells survive the GridDataResInsight roundtrip without change."""
    original = xtgeo.create_box_grid((4, 3, 2))
    actnum = original.get_actnum()
    actnum.values[0, 0, 0] = 0
    actnum.values[2, 1, 1] = 0
    original.set_actnum(actnum)

    data = GridDataResInsight.from_xtgeo_grid(original, name="INACTIVE", filesrc="")
    restored = data.to_xtgeo_grid()

    assert np.array_equal(
        original.get_actnum().values,
        restored.get_actnum().values,
    )


@given(xtgeo_grids)
def test_roundtrip_hypothesis(grid: xtgeo.Grid):
    """Property-based test: any xtgeo box grid round-trips through
    GridDataResInsight without losing dimensions, geometry, or actnum."""
    data = GridDataResInsight.from_xtgeo_grid(grid, name="HYPO", filesrc="")
    restored = data.to_xtgeo_grid()

    assert restored.ncol == grid.ncol
    assert restored.nrow == grid.nrow
    assert restored.nlay == grid.nlay

    grid._set_xtgformat2()
    restored._set_xtgformat2()

    assert_allclose(grid._coordsv, restored._coordsv, atol=1e-2)
    assert_allclose(grid._zcornsv, restored._zcornsv, atol=1e-2)
    assert np.array_equal(grid._actnumsv, restored._actnumsv)


@pytest.mark.requires_resinsight
def test_reader_init(resinsight_instance):
    """Test that GridReader can load grid metadata from ResInsight cases."""

    reader = GridReader(resinsight_instance)
    data = reader.load("EXAMPLE")
    assert data.name == "EXAMPLE", "Should load the grid with the correct name"
    assert data.nx == 4, "Should load the correct nx value"
    assert data.ny == 4, "Should load the correct ny value"
    assert data.nz == 3, "Should load the correct nz value"
    assert data.filesrc.endswith("emerald.roff"), "Should load the correct file source"


@pytest.mark.requires_resinsight
def test_reader_select_first_matching_case(resinsight_instance):
    """Test that GridReader selects the first matching case when find_last is False."""

    reader = GridReader(resinsight_instance)
    data = reader.load("EXAMPLE", find_last=False)
    assert data.name == "EXAMPLE", "Should load the grid with the correct name"
    assert data.nx == 92, "Should load the correct nx value"
    assert data.ny == 146, "Should load the correct ny value"
    assert data.nz == 67, "Should load the correct nz value"
    assert data.filesrc.endswith("geogrid.roff"), (
        "Should load the first matching case when find_last is False"
    )


@pytest.mark.requires_resinsight
def test_reader_no_matching_case(resinsight_instance):
    """Test that GridReader raises an error when no matching case is found."""
    reader = GridReader(resinsight_instance)
    with pytest.raises(
        RuntimeError, match="Cannot find any case with name 'NON_EXISTENT_CASE'"
    ):
        reader.load("NON_EXISTENT_CASE")


@pytest.mark.requires_resinsight
def test_writer_roundtrip_new_case(resinsight_instance):
    reader = GridReader(resinsight_instance)
    data = reader.load("EXAMPLE")

    writer = GridWriter(resinsight_instance)
    writer.save(data, gname="NEW_CASE")

    reloaded_data = reader.load("NEW_CASE")
    assert reloaded_data.name == "NEW_CASE", "Should reload the grid with the new name"
    assert reloaded_data.nx == data.nx, "Should reload the correct nx value"
    assert reloaded_data.ny == data.ny, "Should reload the correct ny value"
    assert reloaded_data.nz == data.nz, "Should reload the correct nz value"
    assert reloaded_data.filesrc == data.filesrc, (
        "Should reload the correct file source"
    )

    assert np.array_equal(reloaded_data.coordsv, data.coordsv), (
        "Should reload the correct coordsv array"
    )
    assert np.array_equal(reloaded_data.zcornsv, data.zcornsv), (
        "Should reload the correct zcornsv array"
    )
    assert np.array_equal(reloaded_data.actnumsv, data.actnumsv), (
        "Should reload the correct actnumsv array"
    )


@pytest.mark.requires_resinsight
def test_writer_replace_case(resinsight_instance: RipsInstanceType):
    grid = xtgeo.create_box_grid((2, 2, 2), increment=(2.0, 2.0, 2.0))

    writer = GridWriter(resinsight_instance)
    data = GridDataResInsight.from_xtgeo_grid(grid, name="TEST_GRID", filesrc="")

    writer.save(data, gname="SIMPLE", find_last=False)

    reloaded_data = GridReader(resinsight_instance).load("SIMPLE", find_last=False)
    assert reloaded_data.name == "SIMPLE", "Should reload the grid with the new name"
    assert reloaded_data.nx == data.nx, "Should reload the correct nx value"
    assert reloaded_data.ny == data.ny, "Should reload the correct ny value"
    assert reloaded_data.nz == data.nz, "Should reload the correct nz value"
    assert reloaded_data.filesrc == data.filesrc, (
        "Should reload the correct file source"
    )
    assert np.array_equal(reloaded_data.coordsv, data.coordsv), (
        "Should reload the correct coordsv array"
    )
    assert np.array_equal(reloaded_data.zcornsv, data.zcornsv), (
        "Should reload the correct zcornsv array"
    )
    assert np.array_equal(reloaded_data.actnumsv, data.actnumsv), (
        "Should reload the correct actnumsv array"
    )

    # Create a new grid with different dimensions and save it with the same case name
    grid2 = xtgeo.create_box_grid((3, 3, 3))
    data2 = GridDataResInsight.from_xtgeo_grid(grid2, name="TEST_GRID_2", filesrc="")
    writer.save(data2, gname="SIMPLE", find_last=False)

    reloaded_data2 = GridReader(resinsight_instance).load("SIMPLE", find_last=False)
    assert reloaded_data2.name == "SIMPLE", "Should reload the grid with the new name"
    assert reloaded_data2.nx == data2.nx, "Should reload the correct nx value"
    assert reloaded_data2.ny == data2.ny, "Should reload the correct ny value"
    assert reloaded_data2.nz == data2.nz, "Should reload the correct nz value"
    assert reloaded_data2.filesrc == data2.filesrc, (
        "Should reload the correct file source"
    )
    assert np.array_equal(reloaded_data2.coordsv, data2.coordsv), (
        "Should reload the correct coordsv array"
    )
    assert np.array_equal(reloaded_data2.zcornsv, data2.zcornsv), (
        "Should reload the correct zcornsv array"
    )
    assert np.array_equal(reloaded_data2.actnumsv, data2.actnumsv), (
        "Should reload the correct actnumsv array"
    )


@pytest.mark.requires_resinsight
def test_writer_roundtrip_replace_case(resinsight_instance):
    reader = GridReader(resinsight_instance)
    data = reader.load("EXAMPLE")

    writer = GridWriter(resinsight_instance)
    writer.save(data, gname="EXAMPLE")

    reloaded_data = reader.load("EXAMPLE")
    assert reloaded_data.name == "EXAMPLE", (
        "Should reload the grid with the correct name"
    )
    assert reloaded_data.nx == data.nx, "Should reload the correct nx value"
    assert reloaded_data.ny == data.ny, "Should reload the correct ny value"
    assert reloaded_data.nz == data.nz, "Should reload the correct nz value"
    assert reloaded_data.filesrc == data.filesrc, (
        "Should reload the correct file source"
    )
    assert np.array_equal(reloaded_data.coordsv, data.coordsv), (
        "Should reload the correct coordsv array"
    )
    assert np.array_equal(reloaded_data.zcornsv, data.zcornsv), (
        "Should reload the correct zcornsv array"
    )
    assert np.array_equal(reloaded_data.actnumsv, data.actnumsv), (
        "Should reload the correct actnumsv array"
    )
