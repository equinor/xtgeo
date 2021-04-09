import pytest

from xtgeo import Grid


@pytest.fixture
def setup_tmpdir(tmpdir):
    with tmpdir.as_cwd():
        yield


def assert_equal_to_init(init, result):
    init = init.copy()
    result_dict = {key: getattr(result, key) for key in init.keys()}
    assert result_dict == pytest.approx(init)


@pytest.mark.usefixtures("setup_tmpdir")
@pytest.mark.parametrize("fformat", ["egrid", "grdecl", "bgrdecl", "roff"])
def test_simple_io(fformat):
    grid = Grid(10, 10, 10, name="my_file")
    grid.create_box()
    init = {
        key: getattr(grid, key)
        for key in {
            "ncol",
            "nrow",
            "nlay",
            # "_coordsv",
            # "_zcornsv",
            # "_actnumsv",
            # "_xtgformat",  # how to handle?
            "subgrids",
            "dualperm",
            "dualporo",
            # "dualactnum",
            "props",
            "name",
        }
    }
    grid.to_file("my_file", fformat=fformat)
    grid_from_file = Grid.read_file("my_file", fformat=fformat)
    assert_equal_to_init(init, grid_from_file)
    grid._coordsv.flatten() == grid_from_file._coordsv.flatten()
    grid._zcornsv.flatten() == grid_from_file._zcornsv.flatten()
    grid._actnumsv.flatten() == grid_from_file._actnumsv.flatten()
