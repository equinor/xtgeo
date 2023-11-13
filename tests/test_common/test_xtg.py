# -*- coding: utf-8 -*-
import os
import time

import pytest

from xtgeo.common import XTGeoDialog, timer

# pylint: disable=invalid-name

xtg = XTGeoDialog()
logger = xtg.basiclogger(__name__)

if not xtg.testsetup():
    raise SystemExit

testpath = xtg.testpathobj


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


def test_timer():
    """Test the timer function"""
    with timer() as elapsed:
        assert pytest.approx(0, abs=0.001) == elapsed().total_seconds()
        time.sleep(0.001)
        assert pytest.approx(0.001, abs=0.001) == elapsed().total_seconds()

    # After ctx. exit time stops.
    assert pytest.approx(0.001, abs=0.001) == elapsed().total_seconds()
    time.sleep(0.001)
    assert pytest.approx(0.001, abs=0.001) == elapsed().total_seconds()


def test_user_msg():
    """Testing user messages"""

    xtg.say("")
    xtg.say("This is a message")
    xtg.warn("This is a warning")
    xtg.warning("This is also a warning")
    xtg.error("This is an error")
    xtg.critical("This is a critical error")
