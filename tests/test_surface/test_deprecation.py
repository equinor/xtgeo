import functools

import pytest
import deprecation
from packaging import version

import xtgeo


def fail_if_not_removed(version_limit, msg=None):
    if msg is None:
        msg = "This method has reached its deprecation limit and must be removed"

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func(*args, **kwargs)
            if version.parse(xtgeo.version) > version.parse(version_limit):
                pytest.fail(msg)

        return wrapper

    return decorator


@fail_if_not_removed(version_limit="4")
def test_default_init_deprecation():
    with pytest.warns(
        DeprecationWarning, match="ncol is a required argument"
    ) as record:
        xtgeo.RegularSurface()
        assert len(record) == 4


@fail_if_not_removed(version_limit="3", msg="nx as input has passed deprecation period")
def test_nx_deprecation(default_surface):
    default_surface.pop("ncol")
    default_surface["nx"] = 5
    xtgeo.RegularSurface(**default_surface)


@fail_if_not_removed(version_limit="3", msg="ny as input has passed deprecation period")
def test_ny_deprecation(default_surface):
    default_surface.pop("nrow")
    default_surface["ny"] = 3
    xtgeo.RegularSurface(**default_surface)


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
