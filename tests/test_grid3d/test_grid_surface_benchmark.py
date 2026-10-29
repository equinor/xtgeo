import pathlib

import pytest

import xtgeo

REEK_GRID = pathlib.Path("3dgrids/reek/reek_geo_grid.roff")


def _make_surfaces(grid):
    top = xtgeo.surface_from_grid3d(grid)
    top.fill()
    top.values = 1650
    base = top.copy()
    base.values = 1700
    return top, base


def _run_property(grid, top, base, operation):
    if operation == "between":
        return grid.get_property_between_surfaces(top, base)
    if operation == "below":
        return grid.get_property_between_surfaces(top=top)
    return grid.get_property_between_surfaces(base=base)


@pytest.mark.parametrize("operation", ["between", "below", "above"])
@pytest.mark.benchmark(group="grid_surface_property.reek")
def test_benchmark_grid_surface_properties(benchmark, operation, testdata_path):
    grid = xtgeo.grid_from_file(testdata_path / REEK_GRID)
    top, base = _make_surfaces(grid)

    result = benchmark(_run_property, grid, top, base, operation)

    assert result.values.shape == grid.dimensions


@pytest.mark.bigtest
@pytest.mark.parametrize("operation", ["between", "below", "above"])
@pytest.mark.benchmark(group="grid_surface_property.synthetic")
def test_benchmark_grid_surface_properties_synthetic(benchmark, operation):
    grid = xtgeo.create_box_grid(
        (120, 100, 60), origin=(0, 0, 1500), increment=(100, 100, 5)
    )
    top, base = _make_surfaces(grid)

    result = benchmark(_run_property, grid, top, base, operation)

    assert result.values.shape == grid.dimensions
    expected = {"between": 120000, "below": 360000, "above": 480000}
    assert result.values.sum() == expected[operation]
