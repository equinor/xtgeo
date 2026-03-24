import numpy as np
import pytest

import xtgeo
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


@pytest.mark.requires_resinsight
def test_reader_init(resinsight_instance):
    """Test that GridReader can load grid metadata from ResInsight cases."""

    reader = GridReader(resinsight_instance)
    data = reader.load("EXAMPLE")
    assert data.name == "EXAMPLE", "Should load the grid with the correct name"
    assert data.nx == 4, "Should load the correct nx value"
    assert data.ny == 4, "Should load the correct ny value"
    assert data.nz == 3, "Should load the correct nz value"
    assert data.filesrc.endswith("eme/1/emerald.roff"), (
        "Should load the correct file source"
    )


@pytest.mark.requires_resinsight
def test_reader_select_first_matching_case(resinsight_instance):
    """Test that GridReader selects the first matching case when find_last is False."""

    reader = GridReader(resinsight_instance)
    data = reader.load("EXAMPLE", find_last=False)
    assert data.name == "EXAMPLE", "Should load the grid with the correct name"
    assert data.nx == 92, "Should load the correct nx value"
    assert data.ny == 146, "Should load the correct ny value"
    assert data.nz == 67, "Should load the correct nz value"
    assert data.filesrc.endswith("drogon/2/geogrid.roff"), (
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
    # Coordinate roundtrip is intentionally not asserted yet. ResInsight and XTGeo
    # currently disagree on how the bottom pillar coordinates are represented for
    # XTGeo-created grids, and the planned fix belongs in XTGeo rather than in a
    # temporary ResInsight-specific conversion layer.
    # assert np.array_equal(reloaded_data.coordsv, data.coordsv), (
    #     "Should reload the correct coordsv array"
    # )
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
    # Coordinate roundtrip is intentionally not asserted yet. ResInsight and XTGeo
    # currently disagree on how the bottom pillar coordinates are represented for
    # XTGeo-created grids, and the planned fix belongs in XTGeo rather than in a
    # temporary ResInsight-specific conversion layer.
    # assert np.array_equal(reloaded_data.coordsv, data.coordsv), (
    #     "Should reload the correct coordsv array"
    # )
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
    # Coordinate roundtrip is intentionally not asserted yet. ResInsight and XTGeo
    # currently disagree on how the bottom pillar coordinates are represented for
    # XTGeo-created grids, and the planned fix belongs in XTGeo rather than in a
    # temporary ResInsight-specific conversion layer.
    # assert np.array_equal(reloaded_data2.coordsv, data2.coordsv), (
    #     "Should reload the correct coordsv array"
    # )
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
