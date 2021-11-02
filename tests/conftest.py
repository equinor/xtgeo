"""Setup common stuff for pytests."""
import os

import pytest
from hypothesis import HealthCheck, settings

settings.register_profile(
    "ci", max_examples=1000, deadline=None, suppress_health_check=[HealthCheck.too_slow]
)
settings.register_profile(
    "ci-fast",
    max_examples=10,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)


@pytest.fixture
def setup_tmpdir(tmpdir):
    with tmpdir.as_cwd():
        yield


def pytest_runtest_setup(item):
    """Called for each test."""

    markers = [value.name for value in item.iter_markers()]

    # pytest.mark.bigtest
    if "bigtest" in markers:
        if "XTG_BIGTEST" not in os.environ:
            pytest.skip("Skip big test (no env variable XTG_BIGTEST)")

    # pytest.mark.requires_roxar:
    if "requires_roxar" in markers:
        if "ROXENV" not in os.environ:
            pytest.skip("Skip test if outside ROXENV (env variable ROXENV is present)")

    # pytest.mark.requires_opm:
    if "requires_opm" in markers:
        if "HAS_OPM" not in os.environ:
            pytest.skip("Skip as requires OPM")


@pytest.fixture(name="show_plot")
def fixture_xtgshow():
    """For eventual plotting, to be uses in an if sence inside a test."""
    if "ROXENV" in os.environ:
        pytest.skip("Skip plotting tests in roxar environment")
    if any(word in os.environ for word in ["XTGSHOW", "XTG_SHOW"]):
        return True
    return False


def pytest_addoption(parser):
    parser.addoption(
        "--testdatapath",
        help="path to xtgeo-testdata, defaults to ../xtgeo-testdata"
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


@pytest.fixture(name="generate_plot")
def fixture_generate_plot(request):
    if "ROXENV" in os.environ:
        pytest.skip("Skip plotting tests in roxar environment")
    return request.config.getoption("--generate-plots")


@pytest.fixture()
def testpath(request):
    testdatapath = request.config.getoption("--testdatapath")
    environ_path = os.environ.get("XTG_TESTPATH", None)
    if environ_path:
        testdatapath = environ_path

    return testdatapath
