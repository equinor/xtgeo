import pytest
from packaging import version

import xtgeo
from xtgeo import version as xtgeo_version


@pytest.fixture
def any_cube():
    return xtgeo.Cube()


@pytest.fixture
def any_cube_file(any_cube, tmp_path):
    any_cube.to_file(tmp_path / "cube.segy")
    return tmp_path / "cube.segy"


def test_cube_from_file_warns(any_cube, any_cube_file):
    if version.parse(xtgeo_version) < version.parse("2.16"):
        pytest.skip()

    with pytest.warns(DeprecationWarning, match="from_file is deprecated"):
        any_cube.from_file(any_cube_file)


def test_cube_from_file_engine_warns(any_cube, any_cube_file):
    with pytest.warns(DeprecationWarning, match="The engine parameter"):
        any_cube.from_file(any_cube_file, engine="segyio")


def test_cube_file_in_init_warns(any_cube_file):
    with pytest.warns(DeprecationWarning, match="directly from file"):
        xtgeo.Cube(any_cube_file)


@pytest.mark.parametrize(
    "missing_arg, expected_warning",
    [
        ("ncol", "ncol is a required argument"),
        ("nrow", "nrow is a required argument"),
        ("nlay", "nlay is a required argument"),
        ("xinc", "xinc is a required argument"),
        ("yinc", "yinc is a required argument"),
        ("zinc", "zinc is a required argument"),
    ],
)
def test_default_init_deprecation(missing_arg, expected_warning):
    input_args = {
        "ncol": 10,
        "nrow": 10,
        "nlay": 2,
        "xinc": 10.0,
        "yinc": 10.0,
        "zinc": 1.0,
    }
    input_args.pop(missing_arg)
    with pytest.warns(DeprecationWarning, match=expected_warning) as record:
        xtgeo.Cube(**input_args)
        assert len(record) == 1
