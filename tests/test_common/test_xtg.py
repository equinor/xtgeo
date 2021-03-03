# -*- coding: utf-8 -*-
import os
import warnings
import time
import platform

import pytest

from xtgeo.common import XTGeoDialog

# pylint: disable=invalid-name

xtg = XTGeoDialog()
logger = xtg.basiclogger(__name__)

if not xtg.testsetup():
    raise SystemExit

td = xtg.tmpdir
testpath = xtg.testpathobj


# =============================================================================
# Some useful functions
# =============================================================================


def assert_equal(this, that, txt=""):
    """Assert equal wrapper function."""
    logger.debug("Test if values are equal...")
    assert this == that, txt


def assert_almostequal(this, that, tol, txt=""):
    """Assert almost equal wrapper function."""
    logger.debug("Test if values are almost equal...")
    assert this == pytest.approx(that, abs=tol), txt


# SKIP IF TRAVIS --------------------------------------------------------------
qtravis = False
if "TRAVISRUN" in os.environ:
    qtravis = True

skipiftravis = pytest.mark.skipif(qtravis, reason="Test skipped on Travis")

# EQUINOR ONLY ----------------------------------------------------------------
qequinor = False
if "KOMODO_RELEASE" in os.environ:
    qequinor = True

equinor = pytest.mark.skipif(not qequinor, reason="Equinor internal test set")

# BIG TESTS -------------------------------------------------------------------
nobigtests = True
if "XTG_BIGTEST" in os.environ:
    nobigtests = False

bigtest = pytest.mark.skipif(nobigtests, reason="Run time demanding test")

# SEGYIO ----------------------------------------------------------------------
no_segyio = False
try:
    import segyio  # noqa # pylint: disable=unused-import
except ImportError:
    no_segyio = True

if no_segyio:
    warnings.warn('"segyio" library not found')

skipsegyio = pytest.mark.skipif(no_segyio, reason="Skip test with segyio")

# Roxar python-----------------------------------------------------------------
# Routines using matplotlib shall not ran if ROXENV=1
# use the @skipifroxar decorator

roxar = False
if "ROXENV" in os.environ:
    roxenv = str(os.environ.get("ROXENV"))
    roxar = True
    print(roxenv)
    warnings.warn("Roxar is present")


skipifroxar = pytest.mark.skipif(roxar, reason="Skip test in Roxar python")

skipunlessroxar = pytest.mark.skipif(True, reason="Skip if NOT Roxar python OLD")
roxapilicenseneeded = pytest.mark.skipif(not roxar, reason="Skip if NOT Roxar python")

skipplot = False
if "ROXAR_RMS_ROOT" in os.environ:
    skipplot = True

plotskipifroxar = pytest.mark.skipif(
    skipplot, reason="Skip test in as " "Roxar has matplotlib issues"
)

skipwindows = False
if "WINDOWS" in platform.system().upper():
    skipwindows = True

skipifwindows = pytest.mark.skipif(skipwindows, reason="Skip test for Windows")

skipmac = False
if "DARWIN" in platform.system().upper():
    skipmac = True

skipifwindows = pytest.mark.skipif(skipwindows, reason="Skip test for Windows")
skipifmac = pytest.mark.skipif(skipmac, reason="Skip test for MacOS")

# =============================================================================
# Do tests
# =============================================================================


@pytest.fixture()
def mylogger():
    # need to do it like this...
    mlog = xtg.basiclogger(__name__, logginglevel="DEBUG")
    return mlog


def test_info_logger(mylogger, caplog):
    """Test basic logger behaviour, will capture output to stdin"""

    mylogger.info("This is a test")
    #    assert 'This is a test' in caplog.text[0]

    logger.warning("This is a warning")


#    assert 'This is a warning' in caplog.text[0]


def test_difftimelogger():
    # need to do it like this...

    for level in (0, 1, 2, 20, 1):
        mlogger = xtg.basiclogger(__name__, logginglevel="INFO", loggingformat=level)
        mlogger.info("String 1")
        time.sleep(0.3)
        mlogger.info("String 2")
        del mlogger


def test_more_logging_tests(caplog):
    """Testing on the logging levels, see that ENV variable will override
    the basiclogger setting.
    """

    os.environ["XTG_LOGGING_LEVEL"] = "INFO"

    xtgmore = XTGeoDialog()  # another instance
    locallogger = xtgmore.basiclogger(__name__, logginglevel="WARNING")
    locallogger.debug("Display debug")
    locallogger.info("Display info")
    locallogger.warning("Display warning")
    locallogger.critical("Display critical")

    os.environ["XTG_LOGGING_LEVEL"] = "CRITICAL"


def test_timer(capsys):
    """Test the timer function"""

    time1 = xtg.timer()
    for inum in range(100000):
        inum += 1

    xtg.say("Used time was {}".format(xtg.timer(time1)))
    # captured = capsys.readouterr()
    # assert 'Used time was' in captured[0]
    # # repeat to see on screen
    # xtg.say('')
    # xtg.warn('Used time was {}'.format(xtg.timer(time1)))


def test_print_xtgeo_header():
    """Test writing an app header."""
    xtg.print_xtgeo_header("MYAPP", "0.99", info="Beta release (be careful)")


def test_user_msg():
    """Testing user messages"""

    xtg.say("")
    xtg.say("This is a message")
    xtg.warn("This is a warning")
    xtg.warning("This is also a warning")
    xtg.error("This is an error")
    xtg.critical("This is a critical error", sysexit=False)
