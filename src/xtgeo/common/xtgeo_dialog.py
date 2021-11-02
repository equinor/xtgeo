# -*- coding: utf-8 -*-
"""
Module for basic XTGeo dialog, basic interaction with user,
including logging for debugging.

Logging is enabled by setting a environment variable::

  export XTG_LOGGING_LEVEL=INFO   # if bash; will set logging to INFO level
  setenv XTG_LOGGING_LEVEL INFO   # if tcsh; will set logging to INFO level

Other levels are DEBUG and CRITICAL. CRITICAL is default (cf. Pythons logging)

Usage of logging in scripts::

  import xtgeo
  xtg = xtgeo.common.XTGeoDialog()
  logger = xtg.basiclogger(__name__)
  logger.info('This is logging of %s', something)

Other than logging, there is also a template for user interaction, which shall
be used in client scripts::

  xtg.echo('This is a message')
  xtg.warn('This is a warning')
  xtg.error('This is an error, will continue')
  xtg.critical('This is a big error, will exit')

In addition there are other classes:

* XTGShowProgress()

* XTGDescription()

"""


import os
import sys
import re
from datetime import datetime as dtime
import getpass
import platform
import inspect
import logging
import warnings
import timeit
import pathlib

import xtgeo

DEBUG = 0
MLS = 10000000.0


HEADER = "\033[1;96m"
OKBLUE = "\033[94m"
OKGREEN = "\033[92m"
WARN = "\033[93;43m"
ERROR = "\033[93;41m"
CRITICAL = "\033[1;91m"
ENDC = "\033[0m"
BOLD = "\033[1m"
UNDERLINE = "\033[4m"


def _printdebug(*args):
    """local unction to print debugging while initializing logging"""

    if DEBUG:
        print("XTG DEBUG:", *args)


class XTGShowProgress(object):
    """Class for showing progress of a computation to the terminal.

    Example::

        # assuming 30 steps in calculation
        theprogress = XTGShowProgress(30, info='Compute stuff')
        for i in range(30):
            do_slow_computation()
            theprogress.flush(i)
        theprogress.finished()
    """

    def __init__(self, maxiter, info="", leadtext="", skip=1, show=True):
        self._max = maxiter
        self._info = info
        self._show = show
        self._leadtext = leadtext
        self._skip = skip
        self._next = 0

    def flush(self, step):
        if not self._show:
            return
        progress = int(float(step) / float(self._max) * 100.0)
        if progress >= self._next:
            print("{0}{1}% {2}".format(self._leadtext, progress, self._info))
            self._next += self._skip

    def finished(self):
        if not self._show:
            return
        print("{0}{1}% {2}".format(self._leadtext, 100, self._info))


class XTGDescription(object):
    """Class for making desciptions of object instances"""

    def __init__(self):
        self._txt = []

    def title(self, atitle):
        fmt = "=" * 99
        self._txt.append(fmt)
        fmt = "{}".format(atitle)
        self._txt.append(fmt)
        fmt = "=" * 99
        self._txt.append(fmt)

    def txt(self, *atxt):
        atxt = list(atxt)
        fmt = self._smartfmt(atxt)
        self._txt.append(fmt)

    def flush(self):
        fmt = "=" * 99
        self._txt.append(fmt)

        for line in self._txt:
            print(line)

    def astext(self):
        thetext = ""
        fmt = "=" * 99
        self._txt.append(fmt)

        for line in self._txt:
            thetext += line + "\n"

        return thetext[:-1]  # skip last \n

    @staticmethod
    def _smartfmt(atxt):
        alen = len(atxt)
        atxt.insert(1, "=>")
        if alen == 1:
            fmt = "{:40s}".format(*atxt)
        elif alen == 2:
            fmt = "{:40s} {:>2s} {}".format(*atxt)
        elif alen == 3:
            fmt = "{:40s} {:>2s} {}  {}".format(*atxt)
        elif alen == 4:
            fmt = "{:40s} {:>2s} {}  {}  {}".format(*atxt)
        elif alen == 5:
            fmt = "{:40s} {:>2s} {}  {}  {}  {}".format(*atxt)
        elif alen == 6:
            fmt = "{:40s} {:>2s} {}  {}  {}  {}  {}".format(*atxt)
        elif alen == 7:
            fmt = "{:40s} {:>2s} {}  {}  {}  {}  {}  {}".format(*atxt)
        else:
            fmt = "{:40s} {:>2s} {}  {}  {}  {}  {}  {}  {}".format(*atxt)
        return fmt


class _TimeFilter(logging.Filter):  # pylint: disable=too-few-public-methods
    """handling difftimes in logging..."""

    # cf https://stackoverflow.com/questions/31521859/
    # \python-logging-module-time-since-last-log

    def filter(self, record):
        # pylint: disable=access-member-before-definition
        # pylint: disable=attribute-defined-outside-init
        try:
            last = self.last
        except AttributeError:
            last = record.relativeCreated

        dlt = dtime.fromtimestamp(
            record.relativeCreated / 1000.0
        ) - dtime.fromtimestamp(last / 1000.0)

        record.relative = "{0:7.3f}".format(dlt.seconds + dlt.microseconds / MLS)

        self.last = record.relativeCreated
        return True


class _Formatter(logging.Formatter):
    """Override record.pathname to truncate strings"""

    # https://stackoverflow.com/questions/14429724/
    # python-logging-how-do-i-truncate-the-pathname-to-just-the-last-few-characters
    def format(self, record):

        filename = "unset_filename"

        if "pathname" in record.__dict__.keys():
            # truncate the pathname
            filename = record.pathname
            if len(filename) > 40:
                filename = re.sub(r".*src/", "", filename)
            record.pathname = filename

        return super().format(record)


class XTGeoDialog(object):  # pylint: disable=too-many-public-methods
    """System for handling dialogs and messages in XTGeo.

    This module cooperates with Python logging module.

    """

    def __init__(self):
        """Initializing XTGeoDialog."""
        self._callclass = None
        self._caller = None
        self._rootlogger = logging.getLogger()
        self._lformat = None
        self._lformatlevel = 1
        self._logginglevel = "CRITICAL"
        self._logginglevel_fromenv = None
        self._loggingname = ""
        self._test_env = True
        self._testpath = "../xtgeo-testdata"
        self._showrtwarnings = True

        # a string, for Python logging:
        self._logginglevel_fromenv = os.environ.get("XTG_LOGGING_LEVEL", None)

        # a number, for format, 1 is simple, 2 is more info etc
        loggingformat = os.environ.get("XTG_LOGGING_FORMAT")

        _printdebug("Logging format is", loggingformat)

        if self._logginglevel_fromenv:
            self.logginglevel = self._logginglevel_fromenv

        if loggingformat is not None:
            self._lformatlevel = int(loggingformat)

        if "XTG_TESTPATH" in os.environ:
            self._testpath = os.environ.get("XTG_TESTPATH")

    @property
    def testpathobj(self):
        """Return testpath as pathlib.Path object."""
        return pathlib.Path(self._testpath)

    @property
    def testpath(self):
        """Return or setting up testpath."""
        return self._testpath

    @testpath.setter
    def testpath(self, newtestpath):

        if not os.path.isdir(newtestpath):
            raise RuntimeError(
                "Proposed test path is not valid: {}".format(newtestpath)
            )

        self._testpath = newtestpath

    @property
    def logginglevel(self):
        """Set or return a logging level property, e.g. logging.CRITICAL"""

        return self._logginglevel

    @logginglevel.setter
    def logginglevel(self, level):
        # pylint: disable=pointless-statement

        validlevels = ("INFO", "WARNING", "DEBUG", "CRITICAL")
        if level in validlevels:
            self._logginglevel = level
        else:
            raise ValueError(
                "Invalid level given, must be " "in {}".format(validlevels)
            )

    @property
    def numericallogginglevel(self):
        """Return a numerical logging level (read only)"""
        llo = logging.CRITICAL
        if self._logginglevel == "INFO":
            llo = logging.INFO
        elif self._logginglevel == "WARNING":
            llo = logging.WARNING
        elif self._logginglevel == "DEBUG":
            llo = logging.DEBUG

        return llo

    @property
    def loggingformatlevel(self):
        return self._lformatlevel

    @property
    def loggingformat(self):
        """Returns the format string to be used in logging"""

        _printdebug("Logging format is", self._lformatlevel)

        if self._lformatlevel <= 1:
            fmt = logging.Formatter(fmt="%(levelname)8s: (%(relative)ss) \t%(message)s")

        elif self._lformatlevel == 2:
            fmt = _Formatter(
                fmt="%(levelname)8s (%(relative)ss) %(pathname)44s "
                "[%(funcName)40s()] %(lineno)4d >> \t%(message)s"
            )

        else:
            fmt = logging.Formatter(
                fmt="%(asctime)s Line: %(lineno)4d %(name)44s "
                "(Delta=%(relative)ss) "
                "[%(funcName)40s()]"
                "%(levelname)8s:"
                "\t%(message)s"
            )

        log = self._rootlogger
        _tmp1 = [hndl.addFilter(_TimeFilter()) for hndl in log.handlers]
        _tmp2 = [hndl.setFormatter(fmt) for hndl in log.handlers]

        _printdebug("TMP1:", _tmp1)
        _printdebug("TMP2:", _tmp2)

        self._lformat = fmt._fmt  # private attribute in Formatter()
        return self._lformat

    @staticmethod
    def get_xtgeo_info(variant="clibinfo"):
        """Prints a banner for a XTGeo app to STDOUT.

        Args:
            variant (str): Variant of info

        Returns:
            info (str): A string with XTGeo system info

        """

        if variant == "clibinfo":
            return "XTGeo version {} (Python {} on {})".format(
                xtgeo.__version__,
                platform.python_version(),
                platform.system(),
            )

        return "Invalid"

    @staticmethod
    def print_xtgeo_header(appname, appversion, info=None):
        """Prints a banner for a XTGeo app to STDOUT.

        Args:
            appname (str): Name of application.
            appversion (str): Version of application on form '3.2.1'
            info (str, optional): More info, e.g. if beta release

        Example::

            xtg.print_xtgeo_header('myapp', '0.2.1', info='Beta release!')
        """

        cur_version = "Python " + str(sys.version_info[0]) + "."
        cur_version += str(sys.version_info[1]) + "." + str(sys.version_info[2])

        app = appname + ", version " + str(appversion)
        if info:
            app = app + " (" + info + ")"
        print("")
        print(HEADER)
        print("#" * 79)
        print("#{}#".format(app.center(77)))
        print("#" * 79)
        nowtime = dtime.now().strftime("%Y-%m-%d %H:%M:%S")
        ver = "Using XTGeo version " + xtgeo.__version__
        cur_version += " @ {} on {} by {}".format(
            nowtime, platform.node(), getpass.getuser()
        )
        print("#{}#".format(ver.center(77)))
        print("#{}#".format(cur_version.center(77)))
        print("#" * 79)
        print(ENDC)
        print("")

    def basiclogger(self, name, logginglevel=None, loggingformat=None, info=False):
        """Initiate the logger by some default settings."""

        if logginglevel is not None and self._logginglevel_fromenv is None:
            self.logginglevel = logginglevel

        if loggingformat is not None and isinstance(loggingformat, int):
            self._lformatlevel = loggingformat

        logging.basicConfig(stream=sys.stdout)
        fmt = self.loggingformat
        self._loggingname = name
        if info:
            print(
                "Logginglevel is {}, formatlevel is {}, and format is {}".format(
                    self.logginglevel, self._lformatlevel, fmt
                )
            )
        self._rootlogger.setLevel(self.numericallogginglevel)

        logging.captureWarnings(True)

        return logging.getLogger(self._loggingname)

    @staticmethod
    def functionlogger(name):
        """Get the logger for functions (not top level)."""

        logger = logging.getLogger(name)
        logger.addHandler(logging.NullHandler())
        return logger

    def testsetup(self):
        """Basic setup for XTGeo testing (private; only relevant for tests)"""

        tstpath = os.environ.get("XTG_TESTPATH", "../xtgeo-testdata")
        if not os.path.isdir(tstpath):
            raise RuntimeError("Test path is not valid: {}".format(tstpath))

        self._test_env = True
        self._testpath = tstpath

        return True

    @staticmethod
    def timer(*args):
        """Without args; return the time, with a time as arg return the
        difference.
        """
        time1 = timeit.default_timer()

        if args:
            return time1 - args[0]

        return time1

    def show_runtimewarnings(self, flag=True):
        """Show warnings issued by xtg.warn, if flag is True."""
        self._showrtwarnings = flag

    def insane(self, string):
        level = 4
        idx = 0

        caller = sys._getframe(1).f_code.co_name
        frame = inspect.stack()[1][0]
        self.get_callerinfo(caller, frame)

        self._output(idx, level, string)

    def trace(self, string):
        level = 3
        idx = 0

        caller = sys._getframe(1).f_code.co_name
        frame = inspect.stack()[1][0]
        self.get_callerinfo(caller, frame)

        self._output(idx, level, string)

    def debug(self, string):
        level = 2
        idx = 0

        caller = sys._getframe(1).f_code.co_name
        frame = inspect.stack()[1][0]
        self.get_callerinfo(caller, frame)

        self._output(idx, level, string)

    def speak(self, string):
        level = 1
        idx = 1

        caller = sys._getframe(1).f_code.co_name
        frame = inspect.stack()[1][0]
        self.get_callerinfo(caller, frame)

        self._output(idx, level, string)

    info = speak

    def say(self, string):
        level = -5
        idx = 3

        caller = sys._getframe(1).f_code.co_name
        frame = inspect.stack()[1][0]
        self.get_callerinfo(caller, frame)

        self._output(idx, level, string)

    def warn(self, string):
        """Show warnings at Runtime (pure user info/warns)."""
        level = 0
        idx = 6

        if self._showrtwarnings:
            caller = sys._getframe(1).f_code.co_name
            frame = inspect.stack()[1][0]
            self.get_callerinfo(caller, frame)

            self._output(idx, level, string)

    warning = warn

    @staticmethod
    def warndeprecated(string):
        """Show Deprecation warnings using Python warnings"""

        warnings.simplefilter("default", DeprecationWarning)
        warnings.warn(string, DeprecationWarning, stacklevel=2)

    @staticmethod
    def warnuser(string):
        """Show User warnings, using Python warnings"""

        warnings.simplefilter("default", UserWarning)
        warnings.warn(string, UserWarning, stacklevel=2)

    def error(self, string):
        level = -8
        idx = 8

        caller = sys._getframe(1).f_code.co_name
        frame = inspect.stack()[1][0]
        self.get_callerinfo(caller, frame)

        self._output(idx, level, string)

    def critical(self, string, sysexit=False):
        level = -9
        idx = 9

        caller = sys._getframe(1).f_code.co_name
        frame = inspect.stack()[1][0]
        self.get_callerinfo(caller, frame)

        self._output(idx, level, string)

    def get_callerinfo(self, caller, frame):
        the_class = self._get_class_from_frame(frame)

        # just keep the last class element
        x = str(the_class)
        x = x.split(".")
        the_class = x[-1]

        self._caller = caller
        self._callclass = the_class

        return (self._caller, self._callclass)

    # =============================================================================
    # Private routines
    # =============================================================================

    @staticmethod
    def _get_class_from_frame(fr):
        # pylint: disable=deprecated-method
        args, _, _, value_dict = inspect.getargvalues(fr)

        # we check the first parameter for the frame function is
        # named 'self'
        if args and args[0] == "self":
            instance = value_dict.get("self", None)
            if instance:
                # return its class
                return getattr(instance, "__class__", None)
        # return None otherwise
        return None

    def _output(self, idx, level, string):

        prefix = ""
        endfix = ""

        if idx == 0:
            prefix = "++"
        elif idx == 1:
            prefix = "**"
        elif idx == 3:
            prefix = ">>"
        elif idx == 6:
            prefix = WARN + "##"
            endfix = ENDC
        elif idx == 8:
            prefix = ERROR + "!#"
            endfix = ENDC
        elif idx == 9:
            prefix = CRITICAL + "!!"
            endfix = ENDC

        ulevel = str(level)
        if level == -5:
            ulevel = "M"
        if level == -8:
            ulevel = "E"
        if level == -9:
            ulevel = "W"
        print(
            "{0} <{1}> [{2:23s}->{3:>33s}] {4}{5}".format(
                prefix, ulevel, self._callclass, self._caller, string, endfix
            )
        )
