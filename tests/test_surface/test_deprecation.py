import functools

import deprecation
import pytest
import xtgeo
from packaging import version
from xtgeo.common.version import __version__ as xtgeo_version


def fail_if_not_removed(version_limit, msg=None):
    if msg is None:
        msg = "This method has reached its deprecation limit and must be removed"

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func(*args, **kwargs)
            if version.parse(xtgeo_version) > version.parse(version_limit):
                pytest.fail(msg)

        return wrapper

    return decorator


@fail_if_not_removed(version_limit="4")
@pytest.mark.parametrize(
    "missing_arg, expected_warning",
    [
        ("ncol", "ncol is a required argument"),
        ("nrow", "nrow is a required argument"),
        ("xinc", "xinc is a required argument"),
        ("yinc", "yinc is a required argument"),
    ],
)
def test_default_init_deprecation(missing_arg, expected_warning):
    input_args = {"ncol": 10, "nrow": 10, "xinc": 10.0, "yinc": 10.0}
    input_args.pop(missing_arg)
    with pytest.warns(DeprecationWarning, match=expected_warning) as record:
        xtgeo.RegularSurface(**input_args)
        assert len(record) == 1


@fail_if_not_removed(version_limit="4")
def test_default_values_deprecation():
    with pytest.warns(DeprecationWarning, match="Default values") as record:
        xtgeo.RegularSurface(**{"ncol": 5, "nrow": 3, "xinc": 25.0, "yinc": 25.0})
        assert len(record) == 1


@deprecation.fail_if_not_removed
@pytest.mark.usefixtures("setup_tmpdir")
def test_from_file_deprecation(default_surface):
    surface = xtgeo.RegularSurface(**default_surface)
    surface.to_file("my_file")
    surface.from_file("my_file")


@deprecation.fail_if_not_removed
def test_from_grid3d_deprecation(default_surface):
    mygrid = xtgeo.Grid()
    surface = xtgeo.RegularSurface(**default_surface)
    surface.from_grid3d(mygrid)


@fail_if_not_removed(
    version_limit="4",
    msg="Creating directly from file has passed deprecation period and must be removed",
)
@pytest.mark.usefixtures("setup_tmpdir")
def test_init_from_file_deprecation(default_surface):
    surface = xtgeo.RegularSurface(**default_surface)
    surface.to_file("my_file")
    with pytest.warns(
        DeprecationWarning, match="Initializing directly from file name is deprecated"
    ) as record:
        xtgeo.RegularSurface("my_file")
        assert len(record) == 1
