# -*- coding: utf-8 -*-
"""Module for XTGeo defined Excpetions.

This module implements a number of Python exceptions you can raise from
within your views to trigger a special Exception.

Usage Example
-------------
::
    import xtgeo
    from xtgeo.common.exceptions import DateNotFoundError
"""


class DateNotFoundError(ValueError):
    """Invalid date in restart import (date not found)"""


class KeywordFoundNoDateError(ValueError):
    """Keyword found in restart, but not at the given date."""


class KeywordNotFoundError(ValueError):
    """Keyword not found in input (restart, init, roff)."""
