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

from __future__ import annotations

import getpass
import inspect
import logging
import os
import pathlib
import platform
import re
import sys
import timeit
import warnings
from datetime import datetime as dtime
from typing import Any, Literal

import xtgeo
from xtgeo.common.log import null_logger

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


def _printdebug(*args: Any) -> None:
    """local unction to print debugging while initializing logging"""

    if DEBUG:
        print("XTG DEBUG:", *args)


class XTGShowProgress:
    """Class for showing progress of a computation to the terminal.

    Example::

        # assuming 30 steps in calculation
        theprogress = XTGShowProgress(30, info='Compute stuff')
        for i in range(30):
            do_slow_computation()
            theprogress.flush(i)
        theprogress.finished()
    """

    def __init__(
        self,
        maxiter: int,
        info: str = "",
        leadtext: str = "",
        skip: int = 1,
        show: bool = True,
    ):
        self._max = maxiter
        self._info = info
        self._show = show
        self._leadtext = leadtext
        self._skip = skip
        self._next = 0

    def flush(self, step: int) -> None:
        if not self._show:
            return
        progress = int(float(step) / float(self._max) * 100.0)
        if progress >= self._next:
            print(f"{self._leadtext}{progress}% {self._info}")
            self._next += self._skip

    def finished(self) -> None:
        if not self._show:
            return
        print(f"{self._leadtext}{100}% {self._info}")


class XTGDescription:
    """Class for making desciptions of object instances"""

    def __init__(self) -> None:
        self._txt: list[str] = []

    def title(self, atitle: str) -> None:
        fmt = "=" * 99
        self._txt.append(fmt)
        fmt = f"{atitle}"
        self._txt.append(fmt)
        fmt = "=" * 99
        self._txt.append(fmt)

    def txt(self, *atxt: Any) -> None:
        fmt = self._smartfmt(list(atxt))
        self._txt.append(fmt)

    def flush(self) -> None:
        fmt = "=" * 99
        self._txt.append(fmt)

        for line in self._txt:
            print(line)

    def astext(self) -> str:
        thetext = ""
        fmt = "=" * 99
        self._txt.append(fmt)

        for line in self._txt:
            thetext += line + "\n"

        return thetext[:-1]  # skip last \n

    @staticmethod
    def _smartfmt(atxt: list[str]) -> str:
        # pylint: disable=consider-using-f-string  # f-string does not work with starred
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

    def filter(self, record: logging.LogRecord) -> bool:
        # pylint: disable=access-member-before-definition
        # pylint: disable=attribute-defined-outside-init
        try:
            last: float = self.last  # type: ignore
        except AttributeError:
            last = record.relativeCreated

        dlt = dtime.fromtimestamp(
            record.relativeCreated / 1000.0
        ) - dtime.fromtimestamp(last / 1000.0)

        record.relative = f"{dlt.seconds + dlt.microseconds / MLS:7.3f}"

        self.last = record.relativeCreated
        return True


class _Formatter(logging.Formatter):
    """Override record.pathname to truncate strings"""

    # https://stackoverflow.com/questions/14429724/
    # python-logging-how-do-i-truncate-the-pathname-to-just-the-last-few-characters
    def format(self, record: logging.LogRecord) -> str:
        filename = "unset_filename"

        if "pathname" in record.__dict__.keys():
            # truncate the pathname
            filename = record.pathname
            if len(filename) > 40:
                filename = re.sub(r".*src/", "", filename)
            record.pathname = filename

        return super().format(record)


class XTGeoDialog:  # pylint: disable=too-many-public-methods
    """System for handling dialogs and messages in XTGeo.

    This module cooperates with Python logging module.

    """

    def __init__(self) -> None:
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
        self._testpath = os.environ.get("XTG_TESTPATH", "../xtgeo-testdata")
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

    @property
    def testpathobj(self) -> pathlib.Path:
        """Return testpath as pathlib.Path object."""
        return pathlib.Path(self._testpath)

    @property
    def testpath(self) -> str:
        """Return or setting up testpath."""
        return self._testpath

    @testpath.setter
    def testpath(self, newtestpath: str) -> None:
        if not os.path.isdir(newtestpath):
            raise RuntimeError(f"Proposed test path is not valid: {newtestpath}")

        self._testpath = newtestpath

    @property
    def logginglevel(self) -> str:
        """Set or return a logging level property, e.g. logging.CRITICAL"""

        return self._logginglevel

    @logginglevel.setter
    def logginglevel(self, level: str) -> None:
        # pylint: disable=pointless-statement

        validlevels = ("INFO", "WARNING", "DEBUG", "CRITICAL")
        if level in validlevels:
            self._logginglevel = level
        else:
            raise ValueError(
                f"Invalid level given, must be one of: {', '.join(validlevels)}"
            )

    @property
    def numericallogginglevel(self) -> int:
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
    def loggingformatlevel(self) -> int:
        return self._lformatlevel

    @property
    def loggingformat(self) -> str | None:
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
        for h in log.handlers:
            h.addFilter(_TimeFilter())
            h.setFormatter(fmt)

        self._lformat = fmt._fmt  # private attribute in Formatter()
        return self._lformat

    @staticmethod
    def get_xtgeo_info(variant: Literal["clibinfo"] = "clibinfo") -> str:
        """Prints a banner for a XTGeo app to STDOUT.

        Args:
            variant (str): Variant of info

        Returns:
            info (str): A string with XTGeo system info

        """

        if variant == "clibinfo":
            return (
                f"XTGeo version {xtgeo.__version__} (Python "
                f"{platform.python_version()} on {platform.system()})"
            )

        return "Invalid"

    @staticmethod
    def print_xtgeo_header(
        appname: str,
        appversion: str | None,
        info: str | None = None,
    ) -> None:
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
        print(f"#{app.center(77)}#")
        print("#" * 79)
        nowtime = dtime.now().strftime("%Y-%m-%d %H:%M:%S")
        ver = "Using XTGeo version " + xtgeo.__version__
        cur_version += f" @ {nowtime} on {platform.node()} by {getpass.getuser()}"
        print(f"#{ver.center(77)}#")
        print(f"#{cur_version.center(77)}#")
        print("#" * 79)
        print(ENDC)
        print("")

    def basiclogger(
        self,
        name: str,
        logginglevel: str | None = None,
        loggingformat: int | None = None,
        info: bool = False,
    ) -> logging.Logger:
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
                f"Logginglevel is {self.logginglevel}, formatlevel is "
                f"{self._lformatlevel}, and format is {fmt}"
            )
        self._rootlogger.setLevel(self.numericallogginglevel)

        logging.captureWarnings(True)

        return logging.getLogger(self._loggingname)

    @staticmethod
    def functionlogger(name: str) -> logging.Logger:
        """
        Deprecated: Get the logger for functions (not top level).

        This method is deprecated and will be removed in a future version.
        Use the `null_logger` function instead for creating loggers with a NullHandler.

        Args:
            name (str): The name of the logger.

        Returns:
            logging.Logger: A logger object with a NullHandler.

        Example:
            # Deprecated usage
            logger = XTGeoDialog.functionlogger(__name__)

            # Recommended usage
            logger = null_logger(__name__)
        """

        warnings.warn(
            "functionlogger is deprecated and will be removed in a future version. "
            "Use null_logger instead.",
            DeprecationWarning,
        )
        return null_logger(name)

    def testsetup(self) -> bool:
        """Basic setup for XTGeo testing (private; only relevant for tests)"""

        tstpath = os.environ.get("XTG_TESTPATH", "../xtgeo-testdata")
        if not os.path.isdir(tstpath):
            raise RuntimeError(f"Test path is not valid: {tstpath}")

        self._test_env = True
        self._testpath = tstpath

        return True

    @staticmethod
    def timer(*args: float) -> float:
        """Without args; return the time, with a time as arg return the
        difference.
        """
        time1 = timeit.default_timer()

        if args:
            return time1 - args[0]

        return time1

    def show_runtimewarnings(self, flag: bool = True) -> None:
        """Show warnings issued by xtg.warn, if flag is True."""
        self._showrtwarnings = flag

    def insane(self, string: str) -> None:
        level = 4
        idx = 0

        caller = sys._getframe(1).f_code.co_name
        frame = inspect.stack()[1][0]
        self.get_callerinfo(caller, frame)

        self._output(idx, level, string)

    def trace(self, string: str) -> None:
        level = 3
        idx = 0

        caller = sys._getframe(1).f_code.co_name
        frame = inspect.stack()[1][0]
        self.get_callerinfo(caller, frame)

        self._output(idx, level, string)

    def debug(self, string: str) -> None:
        level = 2
        idx = 0

        caller = sys._getframe(1).f_code.co_name
        frame = inspect.stack()[1][0]
        self.get_callerinfo(caller, frame)

        self._output(idx, level, string)

    def speak(self, string: str) -> None:
        level = 1
        idx = 1

        caller = sys._getframe(1).f_code.co_name
        frame = inspect.stack()[1][0]
        self.get_callerinfo(caller, frame)

        self._output(idx, level, string)

    info = speak

    def say(self, string: str) -> None:
        level = -5
        idx = 3

        caller = sys._getframe(1).f_code.co_name
        frame = inspect.stack()[1][0]
        self.get_callerinfo(caller, frame)

        self._output(idx, level, string)

    def warn(self, string: str) -> None:
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
    def warndeprecated(string: str) -> None:
        """Show Deprecation warnings using Python warnings"""

        warnings.simplefilter("default", DeprecationWarning)
        warnings.warn(string, DeprecationWarning, stacklevel=2)

    @staticmethod
    def warnuser(string: str) -> None:
        """Show User warnings, using Python warnings"""

        warnings.simplefilter("default", UserWarning)
        warnings.warn(string, UserWarning, stacklevel=2)

    def error(self, string: str) -> None:
        level = -8
        idx = 8

        caller = sys._getframe(1).f_code.co_name
        frame = inspect.stack()[1][0]
        self.get_callerinfo(caller, frame)

        self._output(idx, level, string)

    def critical(self, string: str) -> None:
        level = -9
        idx = 9

        caller = sys._getframe(1).f_code.co_name
        frame = inspect.stack()[1][0]
        self.get_callerinfo(caller, frame)

        self._output(idx, level, string)

    def get_callerinfo(self, caller: Any, frame: Any) -> tuple[Any, str]:
        the_class = self._get_class_from_frame(frame)

        # just keep the last class element
        x = str(the_class).split(".")
        the_class = x[-1]

        self._caller = caller
        self._callclass = the_class

        return (self._caller, self._callclass)

    # =============================================================================
    # Private routines
    # =============================================================================

    @staticmethod
    def _get_class_from_frame(fr: Any) -> Any:
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

    def _output(self, idx: int, level: int, string: str) -> None:
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
            f"{prefix} <{ulevel}> [{self._callclass:23s}-> "
            f"{self._caller:>33s}] {string}{endfix}"
        )
