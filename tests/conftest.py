"""Conftest functions"""

import functools
import os
import pathlib
import warnings

import pandas as pd
import pytest
from hypothesis import HealthCheck, settings
from packaging.version import parse as versionparse

settings.register_profile(
    "no_timeouts",
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile("no_timeouts")


def pytest_configure(config):
    # Ensure xtgeo-testdata is present where expected before running
    testdatapath = os.environ.get("XTG_TESTPATH", config.getoption("--testdatapath"))
    xtg_testdata = pathlib.Path(testdatapath)
    if not xtg_testdata.is_dir():
        raise RuntimeError(
            f"xtgeo-testdata path {testdatapath} does not exist! Clone it from "
            "https://github.com/equinor/xtgeo-testdata. The preferred location "
            " is ../xtgeo-testdata."
        )


def pytest_addoption(parser):
    parser.addoption(
        "--testdatapath",
        help="Path to xtgeo-testdata, defaults to ../xtgeo-testdata"
        "and is overriden by the XTG_TESTPATH environment variable."
        "Experimental feature, not all tests obey this option.",
        action="store",
        default="../xtgeo-testdata",
    )
    parser.addoption(
        "--generate-plots",
        help="whether to generate plot files. The files are written to the"
        "pytest tmpfolder. In order to inspect whether plots are correctly"
        "generated, the files must be manually inspected.",
        action="store_true",
        default=False,
    )


def in_roxar_env():
    """Helper function to check if running in Roxar/RMS environment"""
    return any(env in os.environ for env in ["ROXENV", "RMSVENV_RELEASE"])


def pytest_runtest_setup(item):
    """Called for each test."""

    markers = [value.name for value in item.iter_markers()]

    # pytest.mark.bigtest
    if "bigtest" in markers and "XTG_BIGTEST" not in os.environ:
        pytest.skip("Skip big test (no env variable XTG_BIGTEST)")

    # pytest.mark.requires_roxar:
    if "requires_roxar" in markers and not in_roxar_env():
        pytest.skip("Skip test if outside RMSVENV_RELEASE (former ROXENV)")

    # pytest.mark.requires_opm:
    if "requires_opm" in markers and "HAS_OPM" not in os.environ:
        pytest.skip("Skip as requires OPM")


@pytest.fixture(scope="session")
def testdata_path(request):
    # Prefer 'XTG_TESTPATH' environment variable, fallback to the pytest --testdatapath
    # environment variable, which defaults to '../xtgeo-testdata'
    return os.environ.get("XTG_TESTPATH", request.config.getoption("--testdatapath"))


@pytest.fixture()
def tmp_path_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


@pytest.fixture(name="show_plot")
def fixture_xtgshow():
    """For eventual plotting, to be uses in an if sence inside a test."""
    if in_roxar_env():
        pytest.skip("Skip plotting tests in roxar environment")
    return any(word in os.environ for word in ["XTGSHOW", "XTG_SHOW"])


@pytest.fixture(name="generate_plot")
def fixture_generate_plot(request):
    if in_roxar_env():
        pytest.skip("Skip plotting tests in roxar environment")
    return request.config.getoption("--generate-plots")


@pytest.fixture(autouse=True)
def add_testfile_paths(doctest_namespace, testdata_path, tmp_path):
    doctest_namespace["surface_dir"] = testdata_path + "/surfaces/reek/1/"
    doctest_namespace["reek_dir"] = testdata_path + "/3dgrids/reek/"
    doctest_namespace["emerald_dir"] = testdata_path + "/3dgrids/eme/1/"
    doctest_namespace["cube_dir"] = testdata_path + "/cubes/etc/"
    doctest_namespace["well_dir"] = testdata_path + "/wells/reek/1/"
    doctest_namespace["points_dir"] = testdata_path + "/points/reek/1/"
    doctest_namespace["outdir"] = str(tmp_path)


class Helpers:
    @staticmethod
    def df2csv(dataframe, index=True):
        """Combat Pandas change 1.4 -> 1.5; avoid FutureWarning on line_terminator."""
        if versionparse(pd.__version__) < versionparse("1.5"):
            return dataframe.to_csv(line_terminator="\n", index=index)
        return dataframe.to_csv(lineterminator="\n", index=index)


@pytest.fixture
def helpers():
    return Helpers


def suppress_xtgeo_warnings(*warning_types):
    """Decorator to suppress specific warning types during test execution."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with warnings.catch_warnings():
                for warning_type in warning_types:
                    warnings.simplefilter("ignore", warning_type)
                return func(*args, **kwargs)

        return wrapper

    return decorator
