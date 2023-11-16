# -*- coding: utf-8 -*-
"""
Module for basic XTGeo dialog, basic interaction with user,
including logging for debugging.

Logging is enabled by setting a environment variable::

  export XTG_LOGGING_LEVEL=INFO   # if bash; will set logging to INFO level
  setenv XTG_LOGGING_LEVEL INFO   # if tcsh; will set logging to INFO level

Other levels are DEBUG and CRITICAL. CRITICAL is default (cf. Pythons logging)

Usage of logging in scripts::

  from xtgeo.common import logger
  logger.info('This is logging of %s', something)

Other than logging, there is also a template for user interaction, which shall
be used in client scripts::

  xtg.echo('This is a message')
  xtg.warning('This is a warning')
  xtg.error('This is an error, will continue')
  xtg.critical('This is a big error, will exit')

In addition there are other classes:

* XTGShowProgress()

"""

from __future__ import annotations

import contextlib
import datetime
import logging
import os
import pathlib
import platform
import warnings
from typing import Callable, Final, Iterator

import tqdm

import xtgeo

logging.basicConfig(
    level=logging.getLevelName(
        os.environ.get(
            "XTG_LOGGING_LEVEL",
            logging.INFO,
        )
    ),
    format=os.environ.get(
        "XTG_LOGGING_FORMAT",
        "%(levelname)s - %(asctime)s - %(filename)s:%(lineno)d - %(message)s",
    ),
)


def get_xtgeo_info() -> str:
    """Prints a banner for a XTGeo app to STDOUT.

    Args:
        variant (str): Variant of info

    Returns:
        info (str): A string with XTGeo system info

    """

    return (
        f"XTGeo version {xtgeo.__version__} (Python "
        f"{platform.python_version()} on {platform.system()})"
    )


@contextlib.contextmanager
def timer() -> Iterator[Callable[[], datetime.timedelta]]:
    """
    A contextmanger tha tracks elapsed time inside it, after exit it gives
    the total time for the context.

    Ex.
    with timer() as elapsed:
        print(elapsed()) -> 0
        time.seep(1)
        print(elapsed()) -> 1
    print(elapsed()) -> 1
    time.seep(1)
    print(elapsed()) -> 1
    """

    enter = datetime.datetime.now()
    done: datetime.datetime | None = None
    try:
        yield lambda: (done or datetime.datetime.now()) - enter
    finally:
        done = datetime.datetime.now()


def _xtgeo_root_folder() -> pathlib.Path:
    return pathlib.Path("./").resolve()


def _testdatafolder() -> pathlib.Path:
    testpath = pathlib.Path(
        os.environ.get(
            "XTG_TESTPATH",
            _xtgeo_root_folder().parent / "xtgeo-testdata",
        )
    )
    if testpath.is_dir():
        return testpath
    raise RuntimeError(f"XTG_TESTPATH({testpath}) must be a directory.")


class XTGShowProgress(tqdm.tqdm):
    ...


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

    def txt(self, *atxt: str) -> None:
        fmt = self._smartfmt(list(atxt))
        self._txt.append(fmt)

    def flush(self) -> None:
        self._txt.append("=" * 99)
        print("\n".join(self._txt))

    def astext(self) -> str:
        self._txt.append("=" * 99)
        return "\n".join(self._txt)

    @staticmethod
    def _smartfmt(atxt: list[str]) -> str:
        # pylint: disable=consider-using-f-string  # f-string does not work with starred
        alen = len(atxt)
        atxt.insert(1, "=>")
        if alen == 1:
            return "{:40s}".format(*atxt)
        elif alen == 2:
            return "{:40s} {:>2s} {}".format(*atxt)
        elif alen == 3:
            return "{:40s} {:>2s} {}  {}".format(*atxt)
        elif alen == 4:
            return "{:40s} {:>2s} {}  {}  {}".format(*atxt)
        elif alen == 5:
            return "{:40s} {:>2s} {}  {}  {}  {}".format(*atxt)
        elif alen == 6:
            return "{:40s} {:>2s} {}  {}  {}  {}  {}".format(*atxt)
        elif alen == 7:
            return "{:40s} {:>2s} {}  {}  {}  {}  {}  {}".format(*atxt)
        else:
            return "{:40s} {:>2s} {}  {}  {}  {}  {}  {}  {}".format(*atxt)


def warnuser(string: str) -> None:
    """Show User warnings, using Python warnings"""
    warnings.simplefilter("default", UserWarning)
    warnings.warn(string, UserWarning, stacklevel=2)


def warndeprecated(string: str) -> None:
    """Show Deprecation warnings using Python warnings"""
    warnings.simplefilter("default", DeprecationWarning)
    warnings.warn(string, DeprecationWarning, stacklevel=2)


logger: Final = logging.getLogger(name="XTGeo")
XTGeoDialog: Final = logger  # Alias

testdatafolder: Final = _testdatafolder()
