"""Setup common stuff for pytests."""
import os
import platform

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

ALLPLATF = set("darwin linux windows".split())
SKIPPED = set("skipdarwin skiplinux skipwindows".split())


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

    # pytest.mark.skipifroxar:
    if "skipifroxar" in markers:
        if "ROXENV" in os.environ:
            pytest.skip("Skip test in ROXENV (env variable ROXENV is present)")

    # pytest.mark.skipunlessroxar:
    if "skipunlessroxar" in markers:
        if "ROXENV" not in os.environ:
            pytest.skip("Skip test if outside ROXENV (env variable ROXENV is present)")

    # pytest.mark.requires_opm:
    if "requires_opm" in markers:
        if "HAS_OPM" not in os.environ:
            pytest.skip("Skip as requires OPM")

    # pytest.mark.linux ...
    supported = ALLPLATF.intersection(mark.name for mark in item.iter_markers())
    plat = platform.system().lower()
    if supported and plat not in supported:
        pytest.skip("cannot run on platform {}".format(plat))

    # pytest.mark.skiplinux ...
    skipped = SKIPPED.intersection(mark.name for mark in item.iter_markers())
    plat = "skip" + platform.system().lower()
    if skipped and plat in skipped:
        pytest.skip("cannot run on platform {}".format(plat))


def assert_equal(this, that, txt=""):
    """Assert equal wrapper function."""
    assert this == that, txt


def assert_almostequal(this, that, tol, txt=""):
    """Assert almost equal wrapper function."""
    assert this == pytest.approx(that, abs=tol), txt


@pytest.fixture(name="show_plot")
def fixture_xtgshow():
    """For eventual plotting, to be uses in an if sence inside a test."""
    if any(word in os.environ for word in ["XTGSHOW", "XTG_SHOW"]):
        return True
    return False


@pytest.fixture(name="demo")
def fixture_demo():
    """Fixture demo for later usage.

    In the test script run like:

    def test_whatever(demo):
        demo
    """
    print("THIS IS A DEMO")


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
    return request.config.getoption("--generate-plots")


@pytest.fixture()
def testpath(request):
    testdatapath = request.config.getoption("--testdatapath")
    environ_path = os.environ.get("XTG_TESTPATH", None)
    if environ_path:
        testdatapath = environ_path

    return testdatapath
