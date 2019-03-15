# -*- coding: utf-8 -*-
"""Module for XTGeo defined Exceptions.

This module implements a number of Python exceptions you can raise from
within your views to trigger a special Exception. Alternatively, they can
be catched by the standard ValueError which is the base class.

These exceptions will be present on top xtgeo level as e.g.::

  try:
      a_function
  except xtgeo.WellNotFoundError:
      an_action

"""


class DateNotFoundError(ValueError):
    """Invalid date in restart import (date not found) (ValueError)"""


class KeywordFoundNoDateError(ValueError):
    """Keyword found in restart, but not at the given date (ValueError)"""


class KeywordNotFoundError(ValueError):
    """Keyword not found in input (restart, init, roff) (ValueError)"""


class WellNotFoundError(ValueError):
    """Well is not found in the request (ValueError)"""


class GridNotFoundError(ValueError):
    """3D grid is not found in the request (ValueError)"""


class BlockedWellsNotFoundError(ValueError):
    """Blocked Wells icon is not found in the request (ValueError)"""
